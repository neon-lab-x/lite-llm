#!/usr/bin/env python3
"""Download tokenized .npy shards from a HuggingFace Dataset repo.

Usage:
    # Download all shards to default dir
    python scripts/production/download_tokenized_from_hf.py \
        --hf-repo NoBey/l-llm-3b \
        --hf-token hf_xxx

    # Specify output directory
    python scripts/production/download_tokenized_from_hf.py \
        --hf-repo NoBey/l-llm-3b \
        --hf-token hf_xxx \
        --output-dir data/production/tokenized_zh_first_3b

    # Download from a subdirectory in the repo
    python scripts/production/download_tokenized_from_hf.py \
        --hf-repo NoBey/l-llm-3b \
        --hf-token hf_xxx \
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


HF_MIRROR = "https://hf-mirror.com"


def parse_args():
    parser = argparse.ArgumentParser(description="Download tokenized shards from HF Dataset repo")
    parser.add_argument("--hf-repo", required=True, help="HF dataset repo (e.g. NoBey/l-llm-3b)")
    parser.add_argument("--hf-token", default=None, help="HF token (or set HF_TOKEN env var)")
    parser.add_argument("--output-dir", default=None,
                        help="Local output dir (default: auto from datasets config)")
    parser.add_argument("--datasets-config", default=None,
                        help="Datasets config YAML (used to auto-detect output-dir)")
    parser.add_argument("--repo-path", default=None,
                        help="Only download files under this path in the repo")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel download threads (default: 4)")
    parser.add_argument("--no-mirror", action="store_true",
                        help="Disable HF mirror (use direct hf.co endpoint)")
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = args.output_dir
    if not output_dir:
        if args.datasets_config:
            sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "production"))
            import prepare_data as prep
            cfg = prep.load_datasets_config(args.datasets_config)
            output_dir = cfg.get("output_dir") or os.path.join(
                PROJECT_ROOT, "data", "production", "tokenized"
            )
        else:
            output_dir = os.path.join(PROJECT_ROOT, "data", "production", "tokenized")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    endpoint = None if args.no_mirror else HF_MIRROR
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
    api = HfApi(token=args.hf_token, endpoint=endpoint or "https://huggingface.co")

    # List remote .npy files
    print(f"  Scanning {args.hf_repo}...")
    all_files = list(api.list_repo_files(args.hf_repo, repo_type="dataset"))
    npy_files = [f for f in all_files if f.endswith(".npy")]
    if args.repo_path:
        prefix = args.repo_path.strip("/") + "/"
        npy_files = [f for f in npy_files if f.startswith(prefix)]

    if not npy_files:
        raise SystemExit("No .npy files found in repo" +
                         (f" under {args.repo_path}" if args.repo_path else ""))

    # Filter out already downloaded
    pending = []
    for f in npy_files:
        local_path = os.path.join(output_dir, f)
        if os.path.exists(local_path):
            continue
        pending.append(f)

    print(f"  Repo:        {args.hf_repo}")
    print(f"  Output:      {output_dir}")
    print(f"  Remote:      {len(npy_files)} files")
    print(f"  Pending:     {len(pending)} files")
    if args.repo_path:
        print(f"  Repo path:   {args.repo_path}")
    print(f"  Workers:     {args.workers}")
    print(f"  Mirror:      {endpoint or 'disabled'}")
    print()

    if not pending:
        print("  All files already downloaded.")
        return

    # Download
    print(f"  Downloading...")
    t0 = time.time()
    downloaded = 0
    total_bytes = 0

    def _download_one(repo_path):
        local = os.path.join(output_dir, repo_path)
        os.makedirs(os.path.dirname(local), exist_ok=True)
        return api.hf_hub_download(
            repo_id=args.hf_repo,
            filename=repo_path,
            repo_type="dataset",
            local_dir=output_dir,
        )

    if args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_download_one, f): f for f in pending}
            for future in as_completed(futures):
                repo_path = futures[future]
                try:
                    result = future.result()
                    downloaded += 1
                    size = os.path.getsize(os.path.join(output_dir, repo_path))
                    total_bytes += size
                    elapsed = time.time() - t0
                    rate = total_bytes / elapsed / 1e6
                    print(f"  [{downloaded}/{len(pending)}] {os.path.basename(repo_path)} "
                          f"({size/1e6:.1f} MB) {rate:.1f} MB/s")
                except Exception as e:
                    print(f"  FAILED: {repo_path}: {e}")
    else:
        for i, f in enumerate(pending, 1):
            _download_one(f)
            size = os.path.getsize(os.path.join(output_dir, f))
            total_bytes += size
            elapsed = time.time() - t0
            rate = total_bytes / elapsed / 1e6
            print(f"  [{i}/{len(pending)}] {os.path.basename(f)} "
                  f"({size/1e6:.1f} MB) {rate:.1f} MB/s")

    elapsed = time.time() - t0
    print(f"\n  Done! {len(pending)} files ({total_bytes/1e9:.2f} GB) in {elapsed:.0f}s "
          f"({total_bytes/elapsed/1e6:.1f} MB/s)")


if __name__ == "__main__":
    main()
