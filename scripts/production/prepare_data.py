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
DATA_DIR = "./data/production/tokenized"


def parse_args():
    parser = argparse.ArgumentParser(description="Download pre-tokenized production data from HuggingFace")
    parser.add_argument("--dry-run", action="store_true", help="Show remote file count and size without downloading")
    return parser.parse_args()


def main():
    args = parse_args()

    api = HfApi()

    if args.dry_run:
        print(f"Listing files in {HF_REPO}/{HF_SUBFOLDER} ...\n")
        files = list(api.list_repo_files(repo_id=HF_REPO, repo_type="dataset", path_in_repo=HF_SUBFOLDER))
        npy_files = [f for f in files if f.endswith(".npy")]
        print(f"  {len(npy_files)} .npy shards on remote")
        print(f"  Run without --dry-run to download.\n")
        return

    # local_dir points to ./data/production/ so that repo's tokenized/ maps to ./data/production/tokenized/
    local_dir = "./data/production/"
    os.makedirs(local_dir, exist_ok=True)

    print(f"Downloading from {HF_REPO}/{HF_SUBFOLDER} -> {DATA_DIR}")
    print("Resume is enabled: already-present files will be skipped.\n")

    snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=f"{HF_SUBFOLDER}/*.npy",
        resume_download=True,
    )

    print("\nDownload complete.")
    print_token_summary(DATA_DIR)


if __name__ == "__main__":
    main()
