#!/usr/bin/env python3
"""Upload tokenized .npy shards to a HuggingFace Dataset repo.

Usage:
    # Upload all .npy files from a tokenized directory
    python scripts/production/upload_tokenized_to_hf.py \
        --hf-repo NoBey/l-llm-3b \
        --hf-token hf_xxx \
        --tokenized-dir data/production/tokenized_zh_first_3b

    # Dry run (list files to upload without actually uploading)
    python scripts/production/upload_tokenized_to_hf.py \
        --hf-repo NoBey/l-llm-3b \
        --hf-token hf_xxx \
        --tokenized-dir data/production/tokenized_zh_first_3b \
        --dry-run

    # Upload to a subdirectory inside the repo
    python scripts/production/upload_tokenized_to_hf.py \
        --hf-repo NoBey/l-llm-3b \
        --hf-token hf_xxx \
        --tokenized-dir data/production/tokenized_zh_first_3b \
        --repo-path zh_first_3b
"""

import argparse
import os
import sys
import time

try:
    from huggingface_hub import HfApi
except ModuleNotFoundError as exc:
    raise SystemExit("Run `uv sync --extra production --frozen` first.") from exc

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def parse_args():
    parser = argparse.ArgumentParser(description="Upload tokenized shards to HF Dataset repo")
    parser.add_argument("--hf-repo", required=True, help="HF dataset repo (e.g. NoBey/l-llm-3b)")
    parser.add_argument("--hf-token", default=None, help="HF token (or set HF_TOKEN env var)")
    parser.add_argument("--tokenized-dir", default=None,
                        help="Local dir containing .npy shards (auto-detected from datasets config if omitted)")
    parser.add_argument("--datasets-config", default=None,
                        help="Datasets config YAML (used to auto-detect tokenized-dir)")
    parser.add_argument("--repo-path", default=None,
                        help="Target path inside the repo (default: repo root)")
    parser.add_argument("--commit-message", default=None)
    parser.add_argument("--dry-run", action="store_true", help="List files without uploading")
    return parser.parse_args()


def main():
    args = parse_args()

    tokenized_dir = args.tokenized_dir
    if not tokenized_dir:
        if args.datasets_config:
            sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "production"))
            import prepare_data as prep
            cfg = prep.load_datasets_config(args.datasets_config)
            tokenized_dir = cfg.get("output_dir") or os.path.join(
                PROJECT_ROOT, "data", "production", "tokenized"
            )
        else:
            raise SystemExit("Pass --tokenized-dir or --datasets-config")

    tokenized_dir = os.path.abspath(tokenized_dir)
    if not os.path.isdir(tokenized_dir):
        raise SystemExit(f"Directory not found: {tokenized_dir}")

    # Collect .npy files
    npy_files = sorted(
        os.path.join(tokenized_dir, f)
        for f in os.listdir(tokenized_dir)
        if f.endswith(".npy")
    )

    if not npy_files:
        raise SystemExit(f"No .npy files found in {tokenized_dir}")

    total_bytes = sum(os.path.getsize(f) for f in npy_files)
    print(f"  Repo:        {args.hf_repo}")
    print(f"  Source:      {tokenized_dir}")
    print(f"  Files:       {len(npy_files)}")
    print(f"  Total size:  {total_bytes / 1e9:.2f} GB")
    if args.repo_path:
        print(f"  Repo path:   {args.repo_path}")
    print()

    if args.dry_run:
        print("  Files to upload:")
        for f in npy_files:
            size_mb = os.path.getsize(f) / 1e6
            print(f"    {os.path.basename(f):50s} {size_mb:8.1f} MB")
        print(f"\n  Total: {len(npy_files)} files, {total_bytes / 1e9:.2f} GB")
        return

    api = HfApi(token=args.hf_token)
    # Ensure repo exists
    try:
        api.repo_info(repo_id=args.hf_repo, repo_type="dataset")
    except Exception:
        print(f"  Creating dataset repo: {args.hf_repo}")
        api.create_repo(repo_id=args.hf_repo, repo_type="dataset", private=True)

    commit_msg = args.commit_message or (
        f"Upload {len(npy_files)} tokenized shards "
        f"({total_bytes / 1e9:.2f} GB)"
    )

    print(f"  Uploading (with resume support)...")
    t0 = time.time()
    api.upload_large_folder(
        folder_path=tokenized_dir,
        repo_id=args.hf_repo,
        repo_type="dataset",
        path_in_repo=args.repo_path or "",
    )
    elapsed = time.time() - t0
    rate = total_bytes / elapsed / 1e6

    print(f"\n  Done! {len(npy_files)} files uploaded in {elapsed:.0f}s ({rate:.1f} MB/s)")


if __name__ == "__main__":
    main()
