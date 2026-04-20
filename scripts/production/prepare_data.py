#!/usr/bin/env python3
"""Stream source HF datasets -> tokenize -> save locally and/or upload to HF.

Reads dataset specs from configs/production/datasets.yaml.

Pipeline: source HF -> filter -> tokenize -> .npy shard -> (optional) upload to HF.

Modes:
  (default)       Save .npy shards locally and upload to HuggingFace.
  --local-only     Save .npy shards locally only, skip HF upload.
  --dry-run        Count tokens only, no disk writes or uploads.
  --verify         2 shards per dataset, no resume (quick smoke test).
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

from transformers import AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from lite_llm.token_storage import (
    count_tokens_in_file,
    existing_token_files,
    next_shard_index,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TOKENIZER_NAME = "Qwen/Qwen3.5-0.8B"
HF_TARGET_REPO = "NoBey/lite-llm"
HF_TARGET_PATH = "tokenized"
OUTPUT_DIR = "/root/autodl-tmp/tokenized"
FLUSH_EVERY = 5_000_000          # tokens per shard (~20 MB)
COMMIT_BATCH_SIZE = 50           # shards per commit (~1 GB)
MIRROR_ENDPOINT = "https://hf-mirror.com"
DEFAULT_MIN_DOC_LENGTH = 50


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_tokens(n):
    """Human-readable token count: 1.23B, 456.7M, 12.3K."""
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


def _fmt_duration(seconds):
    """Human-readable duration: 1h23m, 45m12s, 30s."""
    if seconds < 0:
        return "--"
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m{s:02d}s"
    h, remainder = divmod(int(seconds), 3600)
    m = remainder // 60
    return f"{h}h{m:02d}m"


def _fmt_speed(tok_per_sec):
    """Human-readable speed: 15.2K tok/s, 1.2M tok/s."""
    if tok_per_sec >= 1e6:
        return f"{tok_per_sec / 1e6:.1f}M tok/s"
    if tok_per_sec >= 1e3:
        return f"{tok_per_sec / 1e3:.1f}K tok/s"
    return f"{tok_per_sec:.0f} tok/s"

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
    return row.get("int_score", 0) >= 2


@register("chinese")
def _filter_chinese(row):
    text = row.get("text", "")
    if len(text) < 100:
        return False
    cn = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    return cn / max(len(text), 1) >= 0.15


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
    def __init__(self, api, repo_id, batch_size=COMMIT_BATCH_SIZE):
        self.api = api
        self.repo_id = repo_id
        self.batch_size = batch_size
        self.pending: list[str] = []
        self.total_uploaded = 0

    def add(self, local_path):
        self.pending.append(local_path)
        if len(self.pending) >= self.batch_size:
            self._flush()

    def _flush(self):
        from huggingface_hub import CommitOperationAdd

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

    def finish(self):
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
# Resume logic
# ---------------------------------------------------------------------------

def _resume_progress(name, *, uploader=None):
    """Return (done_tokens, shard_idx) for resuming a dataset.

    - Local-only mode (uploader=None): count local shard tokens exactly.
    - Upload mode: estimate remote + count remaining local exactly.
      The last remote shard is assumed partial (conservative) to avoid
      skipping data.
    """
    local_files = existing_token_files(name, OUTPUT_DIR)
    local_tokens = sum(count_tokens_in_file(f) for f in local_files)
    local_idx = next_shard_index(name, OUTPUT_DIR)

    if uploader is None:
        return local_tokens, local_idx

    # Check remote progress
    try:
        remote_files = list(uploader.api.list_repo_files(
            repo_id=HF_TARGET_REPO, repo_type="dataset",
            path_in_repo=f"{HF_TARGET_PATH}/{name}",
        ))
        remote_shards = [f for f in remote_files if f.endswith(".npy")]
        remote_count = len(remote_shards)
        # Conservative: last remote shard may be partial
        remote_tokens = max(remote_count - 1, 0) * FLUSH_EVERY

        # Remove local files already on remote
        remote_set = set(remote_shards)
        for lf in local_files:
            if f"{HF_TARGET_PATH}/{os.path.basename(lf)}" in remote_set:
                os.remove(lf)

        # Recount remaining local (un-uploaded) shards
        remaining_files = existing_token_files(name, OUTPUT_DIR)
        remaining_tokens = sum(count_tokens_in_file(f) for f in remaining_files)
        remaining_idx = next_shard_index(name, OUTPUT_DIR)

        return remote_tokens + remaining_tokens, max(remote_count, remaining_idx)
    except Exception:
        return local_tokens, local_idx


# ---------------------------------------------------------------------------
# Per-dataset processing
# ---------------------------------------------------------------------------

def process_spec(spec, tokenizer, *, uploader=None, scale=1.0,
                 no_resume=False, dry_run=False):
    name = spec["name"]
    target = int(spec["target_tokens"] * scale)
    min_doc_length = spec.get("min_doc_length", DEFAULT_MIN_DOC_LENGTH)
    done_tokens = 0
    shard_idx = 0

    # Resume from existing progress
    if not no_resume and not dry_run:
        done_tokens, shard_idx = _resume_progress(name, uploader=uploader)

    if done_tokens >= target:
        print(f"  {name}: already met ({done_tokens:,} >= {target:,}), skipping.")
        return done_tokens

    print(f"\n{'=' * 60}")
    print(f"  {name}  ({spec['hf_path']})")
    print(f"  Target: {target:,} | Done: {done_tokens:,} | Shard: {shard_idx}")
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
    skipped = 0
    start = time.time()
    last_print_time = start

    for row in ds:
        if collected + len(buf) >= target:
            break
        if filter_fn and not filter_fn(row):
            continue
        text = text_fn(row) if text_fn else (row.get(text_col, "") if text_col else "")
        if not text or len(text.strip()) < min_doc_length:
            skipped += 1
            continue
        buf.extend(tokenizer.encode(text, add_special_tokens=False))
        buf.append(eos_id)
        docs += 1

        if len(buf) >= FLUSH_EVERY:
            collected += len(buf)
            if dry_run:
                buf = _array.array("i")
            else:
                path = _flush_shard(buf, name, shard_idx)
                if uploader is not None:
                    uploader.add(path)
                buf = _array.array("i")
                shard_idx += 1

        # Progress: every 2000 docs or 30 seconds
        now = time.time()
        if (docs % 2000 == 0 and docs > 0) or (now - last_print_time >= 30):
            current = collected + len(buf)
            elapsed = now - start
            rate = (current - done_tokens) / elapsed if elapsed > 0 else 0
            pct = current / max(target, 1) * 100
            remaining = (target - current) / rate if rate > 0 else -1
            disk_mb = collected * 4 / 1e6
            print(f"  [{name}] {pct:5.1f}% | "
                  f"{_fmt_tokens(current)}/{_fmt_tokens(target)} tok | "
                  f"{_fmt_speed(rate)} | "
                  f"ETA {_fmt_duration(remaining)} | "
                  f"{docs:,} docs | "
                  f"{_fmt_duration(elapsed)} elapsed"
                  f"{f' | {disk_mb:.0f} MB' if not dry_run else ''}")
            last_print_time = now

    # Flush remaining buffer
    if buf:
        collected += len(buf)
        if not dry_run:
            path = _flush_shard(buf, name, shard_idx)
            if uploader is not None:
                uploader.add(path)

    print(f"  [{name}] done: {collected:,} tokens from {docs:,} docs "
          f"({skipped:,} filtered) in {_fmt_duration(time.time() - start)}")
    return collected


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Stream -> tokenize -> save/upload")
    p.add_argument("--hf-token", type=str, default=None,
                   help="HuggingFace API token (required unless --dry-run or --local-only)")
    p.add_argument("--datasets", type=str, default=None,
                   help="Comma-separated dataset names to process")
    p.add_argument("--scale", type=float, default=1.0,
                   help="Scale factor for target tokens (e.g. 0.01 for smoke test)")
    p.add_argument("--batch-size", type=int, default=COMMIT_BATCH_SIZE,
                   help=f"Shards per HF commit (default: {COMMIT_BATCH_SIZE})")
    p.add_argument("--no-mirror", action="store_true",
                   help="Disable auto-mirror, download directly from huggingface.co")
    p.add_argument("--local-only", action="store_true",
                   help="Save .npy shards locally, skip HF upload")
    p.add_argument("--dry-run", action="store_true",
                   help="Count tokens only, no disk writes or uploads")
    p.add_argument("--verify", action="store_true",
                   help="Process 2 shards per dataset, no resume (quick smoke test)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    need_upload = not (args.dry_run or args.local_only)

    if need_upload and not args.hf_token:
        print("Error: --hf-token is required when uploading to HuggingFace.",
              file=sys.stderr)
        print("Use --local-only to save locally, or --dry-run to count only.",
              file=sys.stderr)
        sys.exit(1)

    # Mirror: set HF_ENDPOINT so load_dataset / AutoTokenizer go through the
    # mirror. The uploader HfApi gets an explicit endpoint so uploads always
    # go to the real HuggingFace regardless of this flag.
    if not args.no_mirror:
        os.environ["HF_ENDPOINT"] = MIRROR_ENDPOINT
        print(f"Mirror: {MIRROR_ENDPOINT}")
    else:
        print("Mirror: disabled (--no-mirror)")

    # Set up uploader (only for upload mode)
    uploader = None
    if need_upload:
        from huggingface_hub import HfApi

        api = HfApi(token=args.hf_token, endpoint="https://huggingface.co")
        try:
            api.repo_info(repo_id=HF_TARGET_REPO, repo_type="dataset")
        except Exception:
            api.create_repo(repo_id=HF_TARGET_REPO, repo_type="dataset", private=False)
        uploader = BatchUploader(api, HF_TARGET_REPO, args.batch_size)

    print(f"Tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    print(f"  Vocab: {len(tokenizer):,}")
    mode_str = "DRY RUN" if args.dry_run else ("LOCAL ONLY" if args.local_only else "UPLOAD")
    print(f"  Mode: {mode_str}")

    cfg = load_datasets_config()
    specs = build_specs(cfg)
    if args.datasets:
        names = set(args.datasets.split(","))
        specs = [s for s in specs if s["name"] in names]

    no_resume = bool(args.verify)
    if args.verify:
        specs = [dict(s, target_tokens=FLUSH_EVERY * 2) for s in specs]
        print("\nVERIFY MODE: 2 shards per dataset.\n")

    grand = 0
    for spec in specs:
        grand += process_spec(spec, tokenizer, uploader=uploader,
                              scale=args.scale, no_resume=no_resume,
                              dry_run=args.dry_run)

    if uploader is not None:
        uploader.finish()

    print(f"\nDone. {grand:,} tokens ({grand/1e9:.2f}B)")
    if args.local_only:
        print(f"Shards saved to {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
