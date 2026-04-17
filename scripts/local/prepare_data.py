#!/usr/bin/env python3
"""Create the isolated local smoke-test dataset."""

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from lite_llm.local_smoke import DEFAULT_OUTPUT_PATH, write_smoke_tokens


def parse_args():
    parser = argparse.ArgumentParser(description="Create local smoke-test token data")
    parser.add_argument(
        "--output-path",
        default=DEFAULT_OUTPUT_PATH,
        help="Where to write the local smoke-test token shard.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.output_path.startswith("./data/local_smoke/"):
        raise ValueError("Local smoke data must stay under ./data/local_smoke/")
    tokens = write_smoke_tokens(args.output_path)
    print(f"Saved {len(tokens):,} smoke-test tokens to {args.output_path}")
    print(f"Token range: min={int(tokens.min())}, max={int(tokens.max())}")


if __name__ == "__main__":
    main()
