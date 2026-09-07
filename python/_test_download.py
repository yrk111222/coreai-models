import os, shutil


def remove_path(p):
    if os.path.islink(p):
        os.unlink(p)
    elif os.path.isdir(p):
        shutil.rmtree(p)
    elif os.path.exists(p):
        os.remove(p)


# Clear any existing cache for this model
cache_base = os.path.expanduser('~/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B')
remove_path(cache_base)

# --- Test 1: WITH allow_patterns (tokenizer files only) ---
from coreai_models._download import resolve_model_path
local_path_1 = resolve_model_path(
    'Qwen/Qwen3-0.6B',
    allow_patterns=['tokenizer*', 'vocab.json', 'merges.txt', '*.model', '*.txt', '*.jinja'],
)
files_1 = os.listdir(local_path_1)
total_size_1 = sum(os.path.getsize(os.path.join(local_path_1, f)) for f in files_1 if os.path.isfile(os.path.join(local_path_1, f)))
print('=== WITH allow_patterns (tokenizer only) ===')
print(f'Path: {local_path_1}')
print(f'Files ({len(files_1)}):', sorted(files_1))
print(f'Total size: {total_size_1:,} bytes ({total_size_1/1024:.1f} KB)')
print()

# --- Test 2: WITHOUT allow_patterns (full repo) ---
# Clear cache again
cache_base2 = os.path.expanduser('~/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B')
remove_path(cache_base2)

local_path_2 = resolve_model_path('Qwen/Qwen3-0.6B')
files_2 = os.listdir(local_path_2)
total_size_2 = sum(os.path.getsize(os.path.join(local_path_2, f)) for f in files_2 if os.path.isfile(os.path.join(local_path_2, f)))
print('=== WITHOUT allow_patterns (full repo) ===')
print(f'Path: {local_path_2}')
print(f'Files ({len(files_2)}):', sorted(files_2))
print(f'Total size: {total_size_2:,} bytes ({total_size_2/1024/1024:.1f} MB)')
print()

# --- Comparison ---
diff_files = set(files_2) - set(files_1)
print('=== DIFFERENCE ===')
print(f'Extra files downloaded without allow_patterns ({len(diff_files)}):', sorted(diff_files))
print(f'Size saved by allow_patterns: {total_size_2 - total_size_1:,} bytes ({(total_size_2 - total_size_1)/1024/1024:.1f} MB)')