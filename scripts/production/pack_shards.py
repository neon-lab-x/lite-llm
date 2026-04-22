#!/usr/bin/env python3
"""Pack .npy token shards into ~10GB compressed archives.

Scans OUTPUT_DIR for .npy files, sorts deterministically, groups into
batches of ~TARGET_ARCHIVE_SIZE, and creates .tar.zst archives.

Supports incremental packing: skips files already in existing archives.
Re-running picks up where it left off.

Usage:
  python scripts/production/pack_shards.py
  python scripts/production/pack_shards.py --output-dir /data/tokenized
  python scripts/production/pack_shards.py --size 5g
  python scripts/production/pack_shards.py --dry-run
"""

import argparse
import fnmatch
import hashlib
import json
import os
import sys
import tarfile
import time

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUTPUT_DIR = "/root/autodl-fs/tokenized"
TARGET_ARCHIVE_SIZE = 10 * 1024 ** 3   # 10 GB
MANIFEST_NAME = "pack_manifest.json"    # tracks what's been packed
ARCHIVE_PREFIX = "shards_pack"


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt_bytes(n):
    if n >= 1024 ** 3:
        return f"{n / 1024 ** 3:.1f} GB"
    if n >= 1024 ** 2:
        return f"{n / 1024 ** 2:.0f} MB"
    return f"{n / 1024:.0f} KB"


# ---------------------------------------------------------------------------
# Manifest: tracks which files have been packed into which archive
# ---------------------------------------------------------------------------

def _manifest_path(output_dir):
    return os.path.join(output_dir, MANIFEST_NAME)


def load_manifest(output_dir):
    path = _manifest_path(output_dir)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"archives": {}, "packed_files": []}


def save_manifest(output_dir, manifest):
    path = _manifest_path(output_dir)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def find_npy_files(output_dir):
    """Find all .npy files, sorted deterministically by name."""
    files = []
    for f in os.listdir(output_dir):
        if f.endswith(".npy"):
            files.append(f)
    files.sort()
    return files


def compute_file_stable_id(filepath):
    """Stable identifier: filename + size. Avoids re-reading content hash."""
    return f"{os.path.basename(filepath)}:{os.path.getsize(filepath)}"


def plan_batches(npy_files, output_dir, target_size):
    """Group unpacked .npy files into batches of ~target_size bytes.

    Returns list of (archive_name, [filenames]) tuples.
    """
    manifest = load_manifest(output_dir)
    already_packed = set(manifest["packed_files"])

    unpacked = []
    for f in npy_files:
        stable_id = compute_file_stable_id(os.path.join(output_dir, f))
        if stable_id not in already_packed:
            unpacked.append(f)

    if not unpacked:
        return [], manifest

    # Group into batches by cumulative size
    batches = []
    current_batch = []
    current_size = 0

    for f in unpacked:
        fsize = os.path.getsize(os.path.join(output_dir, f))
        if current_size + fsize > target_size and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_size = 0
        current_batch.append(f)
        current_size += fsize

    if current_batch:
        batches.append(current_batch)

    return batches, manifest


def create_archive(batch_files, output_dir, archive_idx):
    """Create a compressed tar archive from a batch of .npy files."""
    archive_name = f"{ARCHIVE_PREFIX}_{archive_idx:04d}.tar"
    archive_path = os.path.join(output_dir, archive_name)

    # Use deterministic member order and mtime for reproducibility
    print(f"  Packing {archive_name} ({len(batch_files)} files, "
          f"{_fmt_bytes(sum(os.path.getsize(os.path.join(output_dir, f)) for f in batch_files))}) ...",
          end="", flush=True)
    start = time.time()

    with tarfile.open(archive_path, "w") as tar:
        for f in batch_files:
            filepath = os.path.join(output_dir, f)
            info = tar.gettarinfo(filepath, arcname=f)
            # Fixed mtime for reproducibility
            info.mtime = 0
            with open(filepath, "rb") as fh:
                tar.addfile(info, fh)

    elapsed = time.time() - start
    compressed_size = os.path.getsize(archive_path)
    raw_size = sum(os.path.getsize(os.path.join(output_dir, f)) for f in batch_files)
    print(f" done in {elapsed:.0f}s  ({_fmt_bytes(compressed_size)} from {_fmt_bytes(raw_size)})")

    return archive_name


def main():
    global ARCHIVE_PREFIX
    parser = argparse.ArgumentParser(description="Pack .npy shards into ~10GB archives")
    parser.add_argument("--output-dir", type=str, default=None,
                        help=f"Directory containing .npy files (default: {OUTPUT_DIR})")
    parser.add_argument("--size", type=str, default="10g",
                        help="Target archive size (e.g. 10g, 5g, 20g)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without creating archives")
    parser.add_argument("--prefix", type=str, default=ARCHIVE_PREFIX,
                        help=f"Archive filename prefix (default: {ARCHIVE_PREFIX})")
    args = parser.parse_args()

    ARCHIVE_PREFIX = args.prefix

    output_dir = args.output_dir or OUTPUT_DIR
    if not os.path.isdir(output_dir):
        print(f"Error: {output_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    # Parse target size
    size_str = args.size.lower().strip()
    if size_str.endswith("g"):
        target_size = int(float(size_str[:-1]) * 1024 ** 3)
    elif size_str.endswith("m"):
        target_size = int(float(size_str[:-1]) * 1024 ** 2)
    else:
        target_size = int(float(size_str))

    print(f"Output:   {output_dir}")
    print(f"Target:   {_fmt_bytes(target_size)} per archive")
    print()

    # Find .npy files
    npy_files = find_npy_files(output_dir)
    if not npy_files:
        print("No .npy files found.")
        return

    total_size = sum(os.path.getsize(os.path.join(output_dir, f)) for f in npy_files)
    print(f"Found {len(npy_files)} .npy files ({_fmt_bytes(total_size)} total)")

    # Plan batches
    batches, manifest = plan_batches(npy_files, output_dir, target_size)

    if not batches:
        print("All files already packed. Nothing to do.")
        return

    unpacked_count = sum(len(b) for b in batches)
    unpacked_size = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for b in batches for f in b
    )
    print(f"Unpacked: {unpacked_count} files ({_fmt_bytes(unpacked_size)})")
    print(f"Plan:     {len(batches)} archives to create")
    print()

    if args.dry_run:
        for i, batch in enumerate(batches):
            batch_size = sum(os.path.getsize(os.path.join(output_dir, f)) for f in batch)
            print(f"  {ARCHIVE_PREFIX}_{i:04d}.tar  "
                  f"{len(batch)} files  {_fmt_bytes(batch_size)}")
            for f in batch[:5]:
                print(f"    {f}")
            if len(batch) > 5:
                print(f"    ... +{len(batch) - 5} more")
        return

    # Determine starting archive index
    existing_archives = fnmatch.filter(os.listdir(output_dir), f"{ARCHIVE_PREFIX}_*.tar")
    if existing_archives:
        existing_indices = []
        for a in existing_archives:
            try:
                idx = int(a.replace(ARCHIVE_PREFIX + "_", "").replace(".tar", ""))
                existing_indices.append(idx)
            except ValueError:
                pass
        next_idx = max(existing_indices) + 1 if existing_indices else 0
    else:
        next_idx = 0

    # Create archives
    for i, batch in enumerate(batches):
        archive_idx = next_idx + i
        archive_name = create_archive(batch, output_dir, archive_idx)

        # Update manifest
        for f in batch:
            stable_id = compute_file_stable_id(os.path.join(output_dir, f))
            manifest["packed_files"].append(stable_id)
        manifest["archives"][archive_name] = [
            compute_file_stable_id(os.path.join(output_dir, f)) for f in batch
        ]
        save_manifest(output_dir, manifest)

    # Final summary
    manifest = load_manifest(output_dir)
    packed = len(manifest["packed_files"])
    archives = len(manifest["archives"])
    print(f"\nDone! {archives} archives created, {packed} files packed.")


if __name__ == "__main__":
    main()
