#!/usr/bin/env python3
"""Upload pre-tokenized shards to HuggingFace dataset NoBey/lite-llm."""

import argparse
import os
import shutil
import sys
import tempfile

from huggingface_hub import HfApi

TOKEN_DIR = "./data/production/tokenized"
HF_REPO = "NoBey/lite-llm"


def main():
    parser = argparse.ArgumentParser(description="Upload tokenized shards to HuggingFace")
    parser.add_argument("--token", type=str, required=True, help="HuggingFace write token")
    parser.add_argument("--repo", type=str, default=HF_REPO)
    parser.add_argument("--data-dir", type=str, default=TOKEN_DIR)
    args = parser.parse_args()

    api = HfApi(token=args.token)

    # Create repo if it doesn't exist
    try:
        api.repo_info(repo_id=args.repo, repo_type="dataset")
        print(f"Dataset {args.repo} exists.")
    except Exception:
        print(f"Creating dataset {args.repo} ...")
        api.create_repo(repo_id=args.repo, repo_type="dataset", private=False)
        print("Created.")

    data_dir = os.path.abspath(args.data_dir)
    npy_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".npy"))
    print(f"Found {len(npy_files)} .npy files to upload.")

    total_bytes = sum(os.path.getsize(os.path.join(data_dir, f)) for f in npy_files)
    print(f"Total size: {total_bytes / 1e9:.2f} GB")

    # Create a temp staging dir with tokenized/ subfolder containing symlinks
    staging = tempfile.mkdtemp(prefix="hf_upload_")
    sub = os.path.join(staging, "tokenized")
    os.makedirs(sub)
    for f in npy_files:
        os.symlink(os.path.join(data_dir, f), os.path.join(sub, f))

    try:
        print(f"\nUploading to {args.repo}/tokenized ...")
        api.upload_large_folder(
            folder_path=staging,
            repo_id=args.repo,
            repo_type="dataset",
        )
        print("Done.")
    finally:
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()
