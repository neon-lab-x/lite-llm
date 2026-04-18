#!/usr/bin/env python3
"""Download pre-tokenized production data from HuggingFace dataset NoBey/lite-llm."""

import argparse
import os
import sys

from huggingface_hub import HfApi, snapshot_download

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from lite_llm.token_storage import print_token_summary

HF_REPO = "NoBey/lite-llm"
HF_SUBFOLDER = "tokenized"
DEFAULT_OUTPUT_DIR = "./data/production/tokenized"


def parse_args():
    parser = argparse.ArgumentParser(description="Download pre-tokenized production data from HuggingFace")
    parser.add_argument("--dry-run", action="store_true", help="Show remote file count and size without downloading")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.output_dir.startswith("./data/production/"):
        raise ValueError("Production data must stay under ./data/production/")

    api = HfApi()

    if args.dry_run:
        print(f"Listing files in {HF_REPO}/{HF_SUBFOLDER} ...\n")
        files = list(api.list_repo_files(repo_id=HF_REPO, repo_type="dataset", path_in_repo=HF_SUBFOLDER))
        npy_files = [f for f in files if f.endswith(".npy")]
        print(f"  {len(npy_files)} .npy shards on remote")
        print(f"  Use without --dry-run to download.\n")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Downloading from {HF_REPO}/{HF_SUBFOLDER} -> {args.output_dir}")
    print("Resume is enabled: already-present files will be skipped.\n")

    snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        local_dir=args.output_dir,
        allow_patterns=f"{HF_SUBFOLDER}/*.npy",
        resume_download=True,
    )

    # snapshot_download creates a subfolder; move files up if needed
    subfolder = os.path.join(args.output_dir, HF_SUBFOLDER)
    if os.path.isdir(subfolder):
        for fname in os.listdir(subfolder):
            src = os.path.join(subfolder, fname)
            dst = os.path.join(args.output_dir, fname)
            if os.path.exists(dst):
                # already at destination (maybe from a previous flat download)
                os.remove(src)
            else:
                os.rename(src, dst)
        os.rmdir(subfolder)

    print("\nDownload complete.")
    print_token_summary(args.output_dir)


if __name__ == "__main__":
    main()
