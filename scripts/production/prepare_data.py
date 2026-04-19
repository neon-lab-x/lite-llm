#!/usr/bin/env python3
"""Stream source HF datasets → tokenize → batch upload to NoBey/lite-llm → delete local.

Reads dataset specs from configs/production/datasets.yaml.
Pipeline: source HF → filter → tokenize → .npy shard → batch commit to NoBey/lite-llm → delete local.
"""

import argparse
import array as _array
import os
import re
import sys
import time

import yaml

try:
    from datasets import load_dataset
except ModuleNotFoundError as exc:
    raise SystemExit("Run `uv sync --extra production --frozen` first.") from exc

from huggingface_hub import CommitOperationAdd, HfApi
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from lite_llm.token_storage import count_tokens_in_file, existing_token_files

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TOKENIZER_NAME = "Qwen/Qwen3.5-0.8B"
HF_TARGET_REPO = "NoBey/lite-llm"
HF_TARGET_PATH = "tokenized"
OUTPUT_DIR = "./data/production/tokenized"
FLUSH_EVERY = 500_000          # tokens per shard (~2 MB)
COMMIT_BATCH_SIZE = 250        # shards per commit

# ---------------------------------------------------------------------------
# Filter implementations (mapped by name from datasets.yaml)
# ---------------------------------------------------------------------------

FILTERS = {}


def register(name):
    def decorator(fn):
        FILTERS[name] = fn
        return fn
    return decorator


@register("fineweb_edu")
def _filter_fineweb_edu(row):
    return row.get("int_score", 0) >= 3


@register("chinese")
def _filter_chinese(row):
    text = row.get("text", "")
    if len(text) < 200:
        return False
    cn = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    return cn / max(len(text), 1) >= 0.3


# ---------------------------------------------------------------------------
# Text extraction (mapped by name from datasets.yaml text_fn field)
# ---------------------------------------------------------------------------

TEXT_FNS = {
    "query_and_answer": lambda row: row.get("query", "") + "\n" + row.get("answer", ""),
}


# ---------------------------------------------------------------------------
# Load datasets.yaml
# ---------------------------------------------------------------------------

def load_datasets_config(path=None):
    if path is None:
        path = os.path.join(PROJECT_ROOT, "configs", "production", "datasets.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_specs(cfg):
    """Convert YAML config into processable spec dicts."""
    specs = []
    for ds in cfg["datasets"]:
        spec = dict(ds)
        # Resolve filter
        fname = ds.get("filter")
        spec["filter_fn"] = FILTERS[fname] if fname and fname in FILTERS else None
        # Resolve text_fn
        tfn = ds.get("text_fn")
        spec["text_fn"] = TEXT_FNS[tfn] if tfn else None
        specs.append(spec)
    return specs


# ---------------------------------------------------------------------------
# Batched uploader
# ---------------------------------------------------------------------------

class BatchUploader:
    def __init__(self, api: HfApi, repo_id: str, batch_size: int = COMMIT_BATCH_SIZE):
        self.api = api
        self.repo_id = repo_id
        self.batch_size = batch_size
        self.pending: list[str] = []
        self.total_uploaded = 0

    def add(self, local_path: str) -> None:
        self.pending.append(local_path)
        if len(self.pending) >= self.batch_size:
            self._flush()

    def _flush(self) -> None:
        ops = []
        paths = self.pending[:]
        for p in paths:
            ops.append(CommitOperationAdd(
                path_in_repo=f"{HF_TARGET_PATH}/{os.path.basename(p)}",
                path_or_fileobj=p,
            ))
        max_retries = 20
        for attempt in range(1, max_retries + 1):
            try:
                self.api.create_commit(
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    operations=ops,
                    commit_message=f"Upload {len(ops)} shards",
                )
                for p in paths:
                    os.remove(p)
                self.total_uploaded += len(paths)
                print(f"  [commit] {len(ops)} shards ({self.total_uploaded} total)")
                self.pending = []
                return
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    m = re.search(r"[Rr]etry after (\d+)", str(e))
                    base = int(m.group(1)) if m else 300
                    wait = min(base + 60 * attempt, 1800)
                    print(f"  [rate limit] waiting {wait}s (attempt {attempt}/{max_retries}) ...")
                    time.sleep(wait)
                else:
                    print(f"  [commit FAILED] {e}")
                    raise
        raise RuntimeError(f"Failed to commit {len(ops)} shards after {max_retries} retries")

    def finish(self) -> None:
        if self.pending:
            self._flush()


# ---------------------------------------------------------------------------
# Shard I/O
# ---------------------------------------------------------------------------

def _flush_shard(tokens, name, shard_idx):
    import numpy as np
    path = os.path.join(OUTPUT_DIR, f"{name}-{shard_idx:05d}.npy")
    np.save(path, np.array(tokens, dtype=np.int32))
    return path


# ---------------------------------------------------------------------------
# Per-dataset processing
# ---------------------------------------------------------------------------

def process_spec(spec, tokenizer, uploader, scale=1.0, no_resume=False):
    name = spec["name"]
    target = int(spec["target_tokens"] * scale)
    done_tokens = 0
    shard_idx = 0

    if not no_resume:
        try:
            remote_files = list(uploader.api.list_repo_files(
                repo_id=HF_TARGET_REPO, repo_type="dataset",
                path_in_repo=f"{HF_TARGET_PATH}/{name}",
            ))
            remote_shards = [f for f in remote_files if f.endswith(".npy")]
            done_tokens = len(remote_shards) * FLUSH_EVERY
            shard_idx = len(remote_shards)
            for lf in existing_token_files(name, OUTPUT_DIR):
                if f"{HF_TARGET_PATH}/{os.path.basename(lf)}" in remote_shards:
                    os.remove(lf)
        except Exception:
            pass

    if done_tokens >= target:
        print(f"  {name}: already met ({done_tokens:,} >= {target:,}), skipping.")
        return done_tokens

    print(f"\n{'=' * 60}")
    print(f"  {name}  ({spec['hf_path']})")
    print(f"  Target: {target:,} | Resume from shard: {shard_idx}")
    print(f"{'=' * 60}")

    kwargs = {"path": spec["hf_path"], "split": spec["split"], "streaming": True}
    if spec.get("config"):
        kwargs["name"] = spec["config"]
    ds = load_dataset(**kwargs)

    eos_id = tokenizer.eos_token_id
    filter_fn = spec.get("filter_fn")
    text_fn = spec.get("text_fn")
    text_col = spec.get("text_column")

    buf = _array.array("i")
    collected = done_tokens
    docs = 0
    start = time.time()

    for row in ds:
        if collected + len(buf) >= target:
            break
        if filter_fn and not filter_fn(row):
            continue
        text = text_fn(row) if text_fn else (row.get(text_col, "") if text_col else "")
        if not text or len(text.strip()) < 50:
            continue
        buf.extend(tokenizer.encode(text, add_special_tokens=False))
        buf.append(eos_id)
        docs += 1

        if len(buf) >= FLUSH_EVERY:
            uploader.add(_flush_shard(buf, name, shard_idx))
            collected += len(buf)
            buf = _array.array("i")
            shard_idx += 1

        if docs % 5000 == 0 and docs > 0:
            elapsed = time.time() - start
            rate = (collected + len(buf)) / elapsed if elapsed > 0 else 0
            pct = (collected + len(buf)) / max(target, 1) * 100
            print(f"  {name}: {docs:,} docs | {collected+len(buf):,} tok ({pct:.1f}%) | {rate/1e3:.0f} tok/s")

    if buf:
        uploader.add(_flush_shard(buf, name, shard_idx))
        collected += len(buf)

    print(f"  {name}: {collected:,} tokens from {docs:,} docs ({time.time()-start:.0f}s)")
    return collected


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Stream → tokenize → upload to NoBey/lite-llm")
    p.add_argument("--hf-token", type=str, required=True)
    p.add_argument("--datasets", type=str, default=None, help="Comma-separated names")
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=COMMIT_BATCH_SIZE)
    p.add_argument("--verify", action="store_true", help="2 shards per dataset, no resume")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    api = HfApi(token=args.hf_token)
    try:
        api.repo_info(repo_id=HF_TARGET_REPO, repo_type="dataset")
    except Exception:
        api.create_repo(repo_id=HF_TARGET_REPO, repo_type="dataset", private=False)

    print(f"Tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    print(f"  Vocab: {len(tokenizer):,}")

    cfg = load_datasets_config()
    specs = build_specs(cfg)
    if args.datasets:
        names = set(args.datasets.split(","))
        specs = [s for s in specs if s["name"] in names]

    no_resume = False
    if args.verify:
        specs = [dict(s, target_tokens=FLUSH_EVERY * 2) for s in specs]
        no_resume = True
        print("\nVERIFY MODE: 2 shards per dataset.\n")

    uploader = BatchUploader(api, HF_TARGET_REPO, args.batch_size)
    grand = 0
    for spec in specs:
        grand += process_spec(spec, tokenizer, uploader, args.scale, no_resume)
    uploader.finish()
    print(f"\nDone. {grand:,} tokens ({grand/1e9:.2f}B)")


if __name__ == "__main__":
    main()
