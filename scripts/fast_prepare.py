#!/usr/bin/env python3
"""Fast multi-process data preparation for Lite-LLM pre-training.

Uses multiprocessing.Pool to parallelize tokenization across CPU cores.
Supports resume — skips already-completed shards.

Usage:
    # Full run (uses all CPU cores)
    python scripts/fast_prepare.py

    # Specific categories
    python scripts/fast_prepare.py --categories english,code

    # Scale down for testing
    python scripts/fast_prepare.py --scale 0.1
"""

import argparse
import array as _array
import glob
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════

TOKENIZER_NAME = "Qwen/Qwen3.5-0.8B"
OUTPUT_DIR = "./data/production/tokenized"
SHARD_SIZE = 500_000       # tokens per .npy shard
BATCH_SIZE = 200           # rows to accumulate before tokenizing

# Targets (can exceed original 8B — we have disk space for ~15B total)
CATEGORY_TARGETS = {
    "english": 4_000_000_000,  # 4.0B
    "chinese": 3_000_000_000,  # 3.0B (already have 2.8B)
    "code":    1_200_000_000,  # 1.2B
    "math":    2_000_000_000,  # 2.0B (already have 1.6B)
}

# ═══════════════════════════════════════════════════════════════════════
# Worker: tokenize a batch of texts in a subprocess
# ═══════════════════════════════════════════════════════════════════════

_tokenizer = None
_eos_id = None

def _init_worker():
    global _tokenizer, _eos_id
    _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    _eos_id = _tokenizer.eos_token_id or 0


def _tokenize_batch(texts):
    """Tokenize a list of texts, return flat token array."""
    tok = _tokenizer
    all_tokens = _array.array("i")
    for text in texts:
        if not text or len(text.strip()) < 50:
            continue
        ids = tok.encode(text, add_special_tokens=False)
        all_tokens.extend(ids)
        all_tokens.append(_eos_id)
    return all_tokens


# ═══════════════════════════════════════════════════════════════════════
# Filters
# ═══════════════════════════════════════════════════════════════════════

def _filter_fineweb_edu(row):
    return row.get("int_score", 0) >= 3

def _filter_chinese(row):
    text = row.get("text", "")
    if len(text) < 200:
        return False
    cn = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    return cn / max(len(text), 1) >= 0.3

TOP_CODE_LANGS = frozenset([
    "Python", "JavaScript", "Java", "C++", "Rust",
    "Go", "TypeScript", "C", "C#", "Kotlin",
])

def _filter_code(row):
    lang = row.get("language", "")
    content = row.get("content", row.get("code", ""))
    return lang in TOP_CODE_LANGS and len(content) >= 200

# ═══════════════════════════════════════════════════════════════════════
# Dataset specs
# ═══════════════════════════════════════════════════════════════════════

DATASET_SPECS = {
    "english": [
        {
            "name": "fineweb_edu",
            "hf_path": "HuggingFaceFW/fineweb-edu",
            "split": "train",
            "text_column": "text",
            "filter_fn": _filter_fineweb_edu,
        },
    ],
    "chinese": [
        {
            "name": "fineweb_edu_chinese",
            "hf_path": "opencsg/Fineweb-Edu-Chinese-V2.1",
            "split": "train",
            "text_column": "text",
            "filter_fn": _filter_chinese,
        },
        {
            "name": "skypile_150b",
            "hf_path": "Skywork/SkyPile-150B",
            "split": "train",
            "text_column": "text",
            "filter_fn": _filter_chinese,
        },
    ],
    "code": [
        {"name": "codesearchnet_python", "hf_path": "Nan-Do/code-search-net-python", "split": "train", "text_column": "code", "filter_fn": None},
        {"name": "codefeedback", "hf_path": "m-a-p/CodeFeedback-Filtered-Instruction", "split": "train", "text_column": "answer", "filter_fn": None},
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

# ═══════════════════════════════════════════════════════════════════════
# Core: stream → batch → parallel tokenize → save
# ═══════════════════════════════════════════════════════════════════════

def _existing_tokens(name, output_dir):
    """Count tokens already on disk for this dataset."""
    total = 0
    for pattern in [f"{name}.npy", f"{name}-*.npy"]:
        for path in glob.glob(os.path.join(output_dir, pattern)):
            total += len(np.load(path, mmap_mode="r", allow_pickle=False))
    return total


def _next_shard(name, output_dir):
    paths = glob.glob(os.path.join(output_dir, f"{name}-*.npy"))
    if not paths:
        return 0
    ids = []
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        try:
            ids.append(int(stem.rsplit("-", 1)[-1]))
        except ValueError:
            pass
    return (max(ids) + 1) if ids else 0


def _save_shard(tokens, name, output_dir, shard_idx):
    path = os.path.join(output_dir, f"{name}-{shard_idx:05d}.npy")
    np.save(path, np.array(tokens, dtype=np.int32))
    return shard_idx + 1


def process_dataset(spec, target_tokens, output_dir, num_workers):
    name = spec["name"]
    hf_path = spec["hf_path"]
    config_name = spec.get("config")
    split = spec["split"]
    text_col = spec.get("text_column")
    filter_fn = spec.get("filter_fn")

    print(f"\n{'='*60}", flush=True)
    print(f"  {name}", flush=True)
    print(f"  HF: {hf_path}  split: {split}", flush=True)
    print(f"  Target: {target_tokens:,} tokens", flush=True)
    print(f"  Workers: {num_workers}", flush=True)
    print(f"{'='*60}", flush=True)

    # --- check existing ---
    collected = _existing_tokens(name, output_dir)
    shard_idx = _next_shard(name, output_dir)
    if collected > 0:
        pct = collected / max(target_tokens, 1) * 100
        print(f"  Resume: {collected:,} tokens on disk ({pct:.1f}%)", flush=True)
        if collected >= target_tokens:
            print("  Target met, skipping.", flush=True)
            return collected

    # --- load dataset ---
    try:
        kwargs = {"path": hf_path, "split": split, "streaming": True}
        if config_name:
            kwargs["name"] = config_name
        ds = load_dataset(**kwargs)
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
        return collected

    # --- parallel tokenization ---
    pool = mp.Pool(num_workers, initializer=_init_worker)
    pending_futures = []
    batch_texts = []
    accumulated = _array.array("i")   # tokens waiting to be flushed
    docs = 0
    filtered = 0
    t0 = time.time()

    for row in ds:
        if filter_fn and not filter_fn(row):
            filtered += 1
            continue
        text = row.get(text_col, "") if text_col else ""
        if not text or len(text.strip()) < 50:
            filtered += 1
            continue

        batch_texts.append(text)
        docs += 1

        if len(batch_texts) >= BATCH_SIZE:
            future = pool.apply_async(_tokenize_batch, (batch_texts,))
            pending_futures.append(future)
            batch_texts = []

        # collect completed futures
        while pending_futures:
            f = pending_futures[0]
            if f.ready():
                tokens = f.get()
                accumulated.extend(tokens)
                collected += len(tokens)
                pending_futures.pop(0)

                # flush accumulated tokens to disk
                while len(accumulated) >= SHARD_SIZE:
                    shard = _array.array("i", accumulated[:SHARD_SIZE])
                    shard_idx = _save_shard(shard, name, output_dir, shard_idx)
                    accumulated = _array.array("i", accumulated[SHARD_SIZE:])

                # Progress
                if docs % (BATCH_SIZE * 5) == 0:
                    elapsed = time.time() - t0
                    rate = collected / elapsed if elapsed > 0 else 0
                    pct = collected / max(target_tokens, 1) * 100
                    print(f"  {docs:,} docs | {collected:,} tokens "
                          f"({pct:.1f}%) | {rate/1e3:.0f} tok/s", flush=True)

                if collected >= target_tokens:
                    if len(accumulated) > 0:
                        shard_idx = _save_shard(accumulated, name, output_dir, shard_idx)
                    pool.close()
                    pool.join()
                    print(f"  Target reached: {collected:,} tokens", flush=True)
                    return collected
            else:
                break

    # flush remaining
    if batch_texts:
        pending_futures.append(pool.apply_async(_tokenize_batch, (batch_texts,)))

    for f in pending_futures:
        tokens = f.get()
        accumulated.extend(tokens)
        collected += len(tokens)

    # save leftover
    if len(accumulated) > 0:
        shard_idx = _save_shard(accumulated, name, output_dir, shard_idx)

    pool.close()
    pool.join()

    elapsed = time.time() - t0
    print(f"  Done: {collected:,} tokens from {docs:,} docs "
          f"({elapsed:.0f}s, {filtered:,} filtered)", flush=True)
    return collected


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Fast multi-process data preparation")
    parser.add_argument("--categories", type=str, default=None,
                        help="Comma-separated: english,chinese,code,math")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Fraction of target tokens")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: CPU count)")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    num_workers = args.workers or max(1, mp.cpu_count() - 1)
    os.makedirs(args.output_dir, exist_ok=True)

    categories = args.categories.split(",") if args.categories else list(CATEGORY_TARGETS.keys())

    print(f"Lite-LLM Fast Data Preparation", flush=True)
    print(f"  Workers: {num_workers}", flush=True)
    print(f"  Output: {args.output_dir}", flush=True)
    print(f"  Categories: {', '.join(categories)}", flush=True)

    grand_total = 0
    for cat in categories:
        target = int(CATEGORY_TARGETS.get(cat, 0) * args.scale)
        if cat not in DATASET_SPECS:
            print(f"WARNING: no spec for '{cat}'")
            continue

        cat_tokens = 0
        for spec in DATASET_SPECS[cat]:
            remaining = target - cat_tokens
            if remaining <= 0:
                break
            collected = process_dataset(spec, remaining, args.output_dir, num_workers)
            cat_tokens += collected

        print(f"\n  Category '{cat}': {cat_tokens:,} tokens (target {target:,})", flush=True)
        grand_total += cat_tokens

    print(f"\n{'='*60}", flush=True)
    print(f"  GRAND TOTAL: {grand_total:,} tokens ({grand_total/1e9:.2f}B)", flush=True)
    print(f"{'='*60}", flush=True)

    # Summary
    total = 0
    for f in sorted(os.listdir(args.output_dir)):
        if not f.endswith(".npy"):
            continue
        path = os.path.join(args.output_dir, f)
        n = len(np.load(path, mmap_mode="r", allow_pickle=False))
        total += n
    print(f"  Files on disk: {total:,} tokens ({total/1e9:.2f}B)", flush=True)


if __name__ == "__main__":
    main()
