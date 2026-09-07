# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Unified model-download abstraction.

This module is the single entry point for fetching model snapshots. It can
route to either the HuggingFace Hub (default) or the ModelScope hub, so the
rest of the codebase never imports ``huggingface_hub`` or ``modelscope``
directly for downloads.

Backend selection (in priority order):

1. Explicit ``backend`` argument to :func:`download_snapshot` /
   :func:`resolve_model_path`.
2. ``COREAI_DOWNLOAD_BACKEND`` environment variable (``huggingface`` |
   ``modelscope``).
3. Default: HuggingFace.

``modelscope`` is an *optional* dependency — it is only imported when the
ModelScope backend is actually selected, so existing users who never opt in
pay no cost and need not install it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

DownloadBackend = Literal["huggingface", "modelscope"]

# HuggingFace → ModelScope model-id mapping.
#
# Both hubs use the ``org/name`` convention, but some models live under a
# different namespace on ModelScope (commonly ``AI-ModelScope/<name>``, the
# official mirror namespace). When the ModelScope backend is active, a
# HuggingFace id is looked up here; if a mapping exists it is used,
# otherwise the original id is passed through unchanged (best-effort
# same-name download).
#
# A few HuggingFace ids are bare names without an ``org/`` prefix (e.g.
# ``roberta-base``). These are valid keys — the mapping supplies the
# ModelScope org prefix (``AI-ModelScope/roberta-base``) that the bare
# name lacks.
#
# Only *verified* mappings are recorded — each entry was confirmed to exist
# on ModelScope via ``modelscope.hub.api.HubApi().get_model``. An unverified
# entry would silently produce a wrong-id 404. Models NOT in this table are
# passed through by id and must exist on ModelScope under the same ``org/name``
# for the download to succeed; see models/README.md for the list of models
# whose ModelScope availability has been verified by same-name passthrough.
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


def _map_to_modelscope_id(hf_id: str) -> str:
    """Return the ModelScope equivalent of a HuggingFace model id.

    Falls back to the original id when no mapping is known.
    """
    return _HF_TO_MODELSCOPE.get(hf_id, hf_id)


def resolve_backend(backend: DownloadBackend | None = None) -> DownloadBackend:
    """Return the effective download backend.

    Precedence: explicit arg > COREAI_DOWNLOAD_BACKEND > huggingface.
    """
    if backend is not None:
        if backend not in ("huggingface", "modelscope"):
            raise ValueError(
                f"Unknown download backend {backend!r}. Expected 'huggingface' or 'modelscope'."
            )
        return backend

    env = os.environ.get("COREAI_DOWNLOAD_BACKEND", "").strip().lower()
    if env:
        if env not in ("huggingface", "modelscope"):
            raise ValueError(
                f"COREAI_DOWNLOAD_BACKEND={env!r} is invalid. "
                "Expected 'huggingface' or 'modelscope'."
            )
        return env  # type: ignore[return-value]

    return "huggingface"


def _download_huggingface(
    model_id: str,
    *,
    allow_patterns: list[str] | str | None = None,
    local_files_only: bool = False,
    token: str | None = None,
    cache_dir: str | Path | None = None,
) -> str:
    import huggingface_hub

    kwargs: dict = {
        "allow_patterns": allow_patterns,
        "local_files_only": local_files_only,
    }
    if token is not None:
        kwargs["token"] = token
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    return huggingface_hub.snapshot_download(model_id, **kwargs)


def _download_modelscope(
    model_id: str,
    *,
    allow_patterns: list[str] | str | None = None,
    local_files_only: bool = False,
    token: str | None = None,
    cache_dir: str | Path | None = None,
) -> str:
    try:
        from modelscope import snapshot_download as _ms_snapshot
    except ImportError as exc:
        raise ImportError(
            "The ModelScope download backend requires the 'modelscope' package. "
            "Install it with: pip install modelscope  (or: uv pip install modelscope)"
        ) from exc

    # Map the HuggingFace id to its ModelScope equivalent (some models live
    # under a different namespace, e.g. AI-ModelScope/*). Falls back to the
    # original id when no mapping is known.
    ms_id = _map_to_modelscope_id(model_id)

    kwargs: dict = {
        "allow_patterns": allow_patterns,
        "local_files_only": local_files_only,
    }
    if token is not None:
        kwargs["token"] = token
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    # ModelScope's snapshot_download accepts ``model_id`` (or ``repo_id``) as
    # the first positional argument and returns a local path string, mirroring
    # huggingface_hub.snapshot_download closely.
    return _ms_snapshot(ms_id, **kwargs)


def download_snapshot(
    model_id: str,
    *,
    allow_patterns: list[str] | str | None = None,
    local_files_only: bool = False,
    token: str | None = None,
    cache_dir: str | Path | None = None,
    backend: DownloadBackend | None = None,
) -> str:
    """Download (or resolve from cache) a full model snapshot and return its
    local path string.

    Args:
        model_id: Model identifier (typically ``org/name``; bare names like
            ``roberta-base`` are also accepted). Works for both HF and ModelScope.
        allow_patterns: Glob patterns restricting which files to fetch.
        local_files_only: When True, only use the local cache; do not hit the network.
        token: Optional auth token (HF token or ModelScope SDK token).
        cache_dir: Optional cache root override.
        backend: Force a backend; otherwise resolved via :func:`resolve_backend`.

    Returns:
        Local filesystem path to the snapshot directory.
    """
    effective = resolve_backend(backend)
    fn = _download_modelscope if effective == "modelscope" else _download_huggingface
    return fn(
        model_id,
        allow_patterns=allow_patterns,
        local_files_only=local_files_only,
        token=token,
        cache_dir=cache_dir,
    )


def resolve_model_path(
    model_id: str,
    *,
    allow_patterns: list[str] | str | None = None,
    cache_dir: str | Path | None = None,
    backend: DownloadBackend | None = None,
) -> str:
    """Resolve a model id to a local snapshot path suitable for passing to
    ``transformers``/``diffusers`` ``from_pretrained(local_path)``.

    This is the bridge that lets ModelScope-backed downloads flow through the
    existing ``AutoConfig.from_pretrained(path)`` / ``AutoTokenizer.from_pretrained(path)``
    call sites: we pre-download the snapshot (only the files the caller needs,
    via ``allow_patterns``) and hand back the local directory.

    Pass the returned path where a HuggingFace model id currently goes.
    """
    return download_snapshot(
        model_id,
        allow_patterns=allow_patterns,
        cache_dir=cache_dir,
        backend=backend,
    )
