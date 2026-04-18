#!/usr/bin/env python3
"""Upload pre-tokenized shards to HuggingFace dataset NoBey/lite-llm."""

import argparse
import os
import sys

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

    npy_files = sorted(f for f in os.listdir(args.data_dir) if f.endswith(".npy"))
    print(f"Found {len(npy_files)} .npy files to upload.")

    total_bytes = sum(os.path.getsize(os.path.join(args.data_dir, f)) for f in npy_files)
    print(f"Total size: {total_bytes / 1e9:.2f} GB")

    # Upload the entire folder (handles parallelism internally)
    print(f"\nUploading to {args.repo} ...")
    api.upload_folder(
        folder_path=args.data_dir,
        repo_id=args.repo,
        repo_type="dataset",
        path_in_repo="tokenized",
    )
    print("Done.")


if __name__ == "__main__":
    main()
