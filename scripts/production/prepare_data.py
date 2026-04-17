#!/usr/bin/env python3
"""Download, filter, tokenize, and save production pre-training data."""

import argparse
import array as _array
import os
import sys
import time

try:
    from datasets import load_dataset
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Production data preparation requires the production dependency set. "
        "Run `uv sync --extra production --frozen` first."
    ) from exc

from transformers import AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from lite_llm.token_storage import (
    count_tokens_in_file,
    existing_token_files,
    flush_token_shard,
    next_shard_index,
    print_token_summary,
)


TOKENIZER_NAME = "Qwen/Qwen3.5-0.8B"
OUTPUT_DIR = "./data/production/tokenized"
FLUSH_EVERY = 50_000_000  # ~200MB per shard (int32); keeps file count manageable.

CATEGORY_TARGETS = {
    "english": 2_800_000_000,
    "chinese": 2_800_000_000,
    "code": 800_000_000,
    "math": 1_600_000_000,
}


def filter_fineweb_edu(row):
    score = row.get("educational_score", 0)
    return float(score) >= 3 if score else False


def filter_chinese_edu(row):
    text = row.get("text", "")
    if len(text) < 200:
        return False
    cn_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    return cn_chars / max(len(text), 1) >= 0.3


TOP_CODE_LANGS = frozenset(
    [
        "Python",
        "JavaScript",
        "Java",
        "C++",
        "Rust",
        "Go",
        "TypeScript",
        "C",
        "C#",
        "Kotlin",
    ]
)


def filter_code(row):
    return row.get("language", "") in TOP_CODE_LANGS and len(row.get("content", "")) >= 200


DATASET_SPECS = {
    "english": [
        {
            "name": "fineweb_edu",
            "hf_path": "HuggingFaceFW/fineweb-edu",
            "split": "train",
            "text_column": "text",
            "filter_fn": filter_fineweb_edu,
        },
    ],
    "chinese": [
        {
            "name": "fineweb_edu_chinese",
            "hf_path": "opencsg/Fineweb-Edu-Chinese-V2.1",
            "split": "train",
            "text_column": "text",
            "filter_fn": filter_chinese_edu,
        },
        {
            "name": "skypile_150b",
            "hf_path": "Skywork/SkyPile-150B",
            "split": "train",
            "text_column": "text",
            "filter_fn": filter_chinese_edu,
        },
    ],
    "code": [
        {
            "name": "github_code",
            "hf_path": "codeparrot/github-code",
            "split": "train",
            "text_column": "code",
            "filter_fn": filter_code,
        },
    ],
    "math": [
        {
            "name": "finemath_4plus",
            "hf_path": "HuggingFaceTB/finemath",
            "config": "finemath-4plus",
            "split": "train",
            "text_column": "text",
            "filter_fn": None,
        },
        {
            "name": "openweb_math",
            "hf_path": "open-web-math/open-web-math",
            "split": "train",
            "text_column": "text",
            "filter_fn": None,
        },
    ],
}


def process_dataset(spec, tokenizer, target_tokens, output_dir):
    name = spec["name"]
    hf_path = spec["hf_path"]
    config_name = spec.get("config")
    split = spec["split"]
    text_col = spec.get("text_column")
    filter_fn = spec.get("filter_fn")
    format_fn = spec.get("format_fn")
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError(
            f"Tokenizer {tokenizer.__class__.__name__} has no eos_token_id; "
            "pretraining packing requires an EOS to separate documents."
        )

    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"  HF: {hf_path}  split: {split}")
    print(f"  Target: {target_tokens:,} tokens")
    print(f"{'=' * 60}")

    try:
        kwargs = {"path": hf_path, "split": split, "streaming": True}
        if config_name:
            kwargs["name"] = config_name
        ds = load_dataset(**kwargs)
    except Exception as e:
        print(f"  ERROR loading dataset: {e}")
        return 0

    files = existing_token_files(name, output_dir)
    collected = sum(count_tokens_in_file(path) for path in files)
    shard_idx = next_shard_index(name, output_dir)

    if collected:
        pct = collected / max(target_tokens, 1) * 100
        print(
            f"  Found {len(files)} existing file(s): {collected:,} tokens already on disk "
            f"({pct:.1f}%)"
        )
        if collected >= target_tokens:
            print("  Target already satisfied on disk, skipping.")
            return collected

    all_tokens = _array.array("i")
    docs = 0
    filtered_out = 0
    start_time = time.time()

    for row in ds:
        if filter_fn and not filter_fn(row):
            filtered_out += 1
            continue

        if format_fn:
            text = format_fn(row)
        elif text_col:
            text = row.get(text_col, "")
        else:
            text = str(row)

        if not text or len(text.strip()) < 50:
            filtered_out += 1
            continue

        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)
        all_tokens.append(eos_id)  # document boundary marker for packing
        docs += 1

        if len(all_tokens) >= FLUSH_EVERY:
            shard_idx = flush_token_shard(all_tokens, name, output_dir, shard_idx)
            collected += len(all_tokens)
            all_tokens = _array.array("i")

        if docs % 5000 == 0 and docs > 0:
            elapsed = time.time() - start_time
            rate = (collected + len(all_tokens)) / elapsed if elapsed > 0 else 0
            pct = (collected + len(all_tokens)) / max(target_tokens, 1) * 100
            print(
                f"  {docs:,} docs | {collected + len(all_tokens):,} tokens "
                f"({pct:.1f}%) | {rate / 1e3:.0f} tok/s"
            )

        if collected + len(all_tokens) >= target_tokens:
            break

    if all_tokens:
        shard_idx = flush_token_shard(all_tokens, name, output_dir, shard_idx)
        collected += len(all_tokens)

    elapsed = time.time() - start_time
    print(
        f"  Done: {collected:,} tokens from {docs:,} docs "
        f"({elapsed:.0f}s, {filtered_out:,} filtered)"
    )
    return collected


def dry_run():
    print("DRY RUN — estimates only\n")
    for category, target in CATEGORY_TARGETS.items():
        gb = target * 4 / 1e9
        print(f"  {category:10s}  target {target / 1e9:.1f}B tokens  ~{gb:.0f} GB on disk")
    total_tokens = sum(CATEGORY_TARGETS.values())
    total_gb = total_tokens * 4 / 1e9
    print(f"  {'TOTAL':10s}  {total_tokens / 1e9:.1f}B tokens  ~{total_gb:.0f} GB on disk")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare production pre-training data")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--categories", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.dry_run:
        dry_run()
        return

    if not args.output_dir.startswith("./data/production/"):
        raise ValueError("Production data must stay under ./data/production/")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    print(f"  Vocab size: {len(tokenizer):,}")

    categories = args.categories.split(",") if args.categories else list(CATEGORY_TARGETS.keys())

    grand_total = 0
    for category in categories:
        target = int(CATEGORY_TARGETS[category] * args.scale)
        if category not in DATASET_SPECS:
            print(f"WARNING: no datasets defined for category '{category}'")
            continue

        cat_tokens = 0
        for spec in DATASET_SPECS[category]:
            remaining = target - cat_tokens
            if remaining <= 0:
                break
            cat_tokens += process_dataset(spec, tokenizer, remaining, args.output_dir)

        print(
            f"\n  Category '{category}': collected {cat_tokens:,} tokens "
            f"(target was {target:,})"
        )
        grand_total += cat_tokens

    print(f"\n  Grand total: {grand_total:,} tokens ({grand_total / 1e9:.2f}B)")
    print_token_summary(args.output_dir)


if __name__ == "__main__":
    main()
