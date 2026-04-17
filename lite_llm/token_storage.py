import glob
import os

import numpy as np


def existing_token_files(name: str, output_dir: str):
    paths = []
    for legacy_ext in (".npy", ".bin"):
        legacy_path = os.path.join(output_dir, f"{name}{legacy_ext}")
        if os.path.exists(legacy_path):
            paths.append(legacy_path)

    paths.extend(sorted(glob.glob(os.path.join(output_dir, f"{name}-*.npy"))))
    paths.extend(sorted(glob.glob(os.path.join(output_dir, f"{name}-*.bin"))))
    return paths


def count_tokens_in_file(path: str):
    if path.endswith(".npy"):
        return len(np.load(path, mmap_mode="r", allow_pickle=False))
    return os.path.getsize(path) // np.dtype(np.int32).itemsize


def next_shard_index(name: str, output_dir: str):
    shard_paths = glob.glob(os.path.join(output_dir, f"{name}-*.npy"))
    if not shard_paths:
        return 0

    shard_ids = []
    for path in shard_paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            shard_ids.append(int(stem.rsplit("-", 1)[-1]))
        except ValueError:
            continue
    return (max(shard_ids) + 1) if shard_ids else 0


def flush_token_shard(tokens, name: str, output_dir: str, shard_idx: int):
    out_path = os.path.join(output_dir, f"{name}-{shard_idx:05d}.npy")
    np.save(out_path, np.array(tokens, dtype=np.int32))
    return shard_idx + 1


def print_token_summary(output_dir: str):
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    total = 0
    for filename in sorted(os.listdir(output_dir)):
        if not (filename.endswith(".npy") or filename.endswith(".bin")):
            continue
        path = os.path.join(output_dir, filename)
        token_count = count_tokens_in_file(path)
        size_mb = os.path.getsize(path) / 1e6
        print(f"  {filename:45s}  {token_count:>12,} tokens  ({size_mb:.0f} MB)")
        total += token_count
    print(f"  {'TOTAL':45s}  {total:>12,} tokens  ({total / 1e9:.2f}B)")
