# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Zero-dependency download shim for standalone ``models/*/export.py`` scripts.

Standalone export scripts run via PEP 723 inline dependencies and may pin
``transformers==4.x`` — incompatible with the ``coreai-models`` package
(which requires ``transformers>=5.x``). So they cannot import
``coreai_models._download``. This shim is a self-contained subset that
replicates the same backend-selection + id-mapping logic *without* importing
``coreai_models``.

Usage (from any ``models/<name>/export.py``):

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _download_shim import resolve_model_path

    local_path = resolve_model_path(model_name)
    model = transformers.XXX.from_pretrained(local_path)

Keep this module in sync with ``python/src/coreai_models/_download.py`` —
specifically the env-var precedence, the invalid-value ``ValueError``, and
the ``_HF_TO_MODELSCOPE`` table. The test suite
(``test_shim_mapping_table_matches_main_module``) enforces the table sync.

This is a *minimal subset* of the main module: ``resolve_model_path`` here
does not support ``local_files_only``, ``token``, or ``backend`` kwargs.
Standalone scripts that need those should use the main module directly.
"""

from __future__ import annotations

import os
from pathlib import Path

# HuggingFace → ModelScope model-id mapping (mirror of coreai_models._download).
# Each entry was verified to exist on ModelScope. A few HF ids are bare names
# (e.g. ``roberta-base``) — the mapping supplies the ModelScope org prefix.
_HF_TO_MODELSCOPE: dict[str, str] = {
    "openai/whisper-large-v3": "AI-ModelScope/whisper-large-v3",
    "openai/whisper-large-v3-turbo": "AI-ModelScope/whisper-large-v3-turbo",
    "google-t5/t5-small": "AI-ModelScope/t5-small",
    "google-t5/t5-base": "AI-ModelScope/t5-base",
    "roberta-base": "AI-ModelScope/roberta-base",
    "openai/clip-vit-base-patch32": "openai-mirror/clip-vit-base-patch32",
    "openai/gpt-oss-20b": "openai-mirror/gpt-oss-20b",
    "nvidia/parakeet-tdt-0.6b-v3": "nv-community/parakeet-tdt-0.6b-v3",
}


def _resolve_backend() -> str:
    """Resolve the download backend from environment variables.

    Precedence: COREAI_DOWNLOAD_BACKEND > huggingface.

    Kept in sync with ``coreai_models._download.resolve_backend`` — including
    raising ``ValueError`` on an invalid ``COREAI_DOWNLOAD_BACKEND`` value so
    that standalone scripts and the package-level pipeline fail the same way
    on a typo.
    """
    env = os.environ.get("COREAI_DOWNLOAD_BACKEND", "").strip().lower()
    if env:
        if env not in ("huggingface", "modelscope"):
            raise ValueError(
                f"COREAI_DOWNLOAD_BACKEND={env!r} is invalid. "
                "Expected 'huggingface' or 'modelscope'."
            )
        return env
    return "huggingface"


def _map_to_modelscope_id(hf_id: str) -> str:
    return _HF_TO_MODELSCOPE.get(hf_id, hf_id)


def resolve_model_path(
    model_id: str,
    *,
    allow_patterns: list[str] | str | None = None,
    cache_dir: str | Path | None = None,
) -> str:
    """Pre-download a model snapshot and return its local path.

    Routes to HuggingFace (default) or ModelScope based on
    ``COREAI_DOWNLOAD_BACKEND`` env var.
    The returned path is suitable for ``transformers.from_pretrained(path)``.
    """
    backend = _resolve_backend()

    if backend == "modelscope":
        try:
            from modelscope import snapshot_download as _ms_snapshot
        except ImportError as exc:
            raise ImportError(
                "The ModelScope download backend requires the 'modelscope' package. "
                "Install it with: pip install modelscope  (or: uv pip install modelscope)"
            ) from exc
        ms_id = _map_to_modelscope_id(model_id)
        kwargs: dict = {"allow_patterns": allow_patterns}
        if cache_dir is not None:
            kwargs["cache_dir"] = str(cache_dir)
        return _ms_snapshot(ms_id, **kwargs)

    import huggingface_hub

    kwargs = {"allow_patterns": allow_patterns}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    return huggingface_hub.snapshot_download(model_id, **kwargs)
