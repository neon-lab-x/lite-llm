#!/usr/bin/env python3
"""Download parquet files from HF -> tokenize -> save locally and/or upload to HF.

Reads dataset specs from configs/production/datasets.yaml.

Pipeline: list parquet files -> download one at a time -> read with pyarrow ->
          filter -> tokenize -> .npy shard -> delete parquet -> next file.

This avoids HF streaming which is unreliable on some networks. Each parquet file
is downloaded independently with resume support, and deleted after processing.

Modes:
  (default)       Save .npy shards locally and upload to HuggingFace.
  --local-only     Save .npy shards locally only, skip HF upload.
  --dry-run        Count tokens only, no disk writes or uploads.
  --verify         2 shards per dataset, no resume (quick smoke test).
"""

import argparse
import array as _array
import gc
import json
import os
import re
import sys
import time

import yaml

try:
    from huggingface_hub import HfApi, hf_hub_download
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
OUTPUT_DIR = "/root/autodl-fs/tokenized"
FLUSH_EVERY = 5_000_000          # tokens per shard (~20 MB)
COMMIT_BATCH_SIZE = 50           # shards per commit (~1 GB)
MIRROR_ENDPOINT = "https://hf-mirror.com"
DEFAULT_MIN_DOC_LENGTH = 50
# Conservative estimate: 1 byte of parquet -> ~0.15 tokens (accounts for
# compression, metadata, and mixed-language content).
BYTES_PER_TOKEN_ESTIMATE = 1 / 0.15


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


def _fmt_bytes(n):
    """Human-readable bytes: 1.2 GB, 456 MB."""
    if n >= 1e9:
        return f"{n / 1e9:.1f} GB"
    if n >= 1e6:
        return f"{n / 1e6:.0f} MB"
    return f"{n / 1e3:.0f} KB"


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
# Progress file for parquet-level resume
# ---------------------------------------------------------------------------

def _progress_path(name, cache_dir):
    return os.path.join(cache_dir, f"{name}_progress.json")


def _load_progress(name, cache_dir):
    """Load set of already-processed parquet file paths."""
    path = _progress_path(name, cache_dir)
    if os.path.exists(path):
        with open(path) as f:
            return set(json.load(f))
    return set()


def _save_progress(name, cache_dir, processed_files):
    """Save set of processed parquet file paths."""
    path = _progress_path(name, cache_dir)
    with open(path, "w") as f:
        json.dump(sorted(processed_files), f)


# ---------------------------------------------------------------------------
# Data file listing
# ---------------------------------------------------------------------------

# Extensions we can read locally (no HF streaming needed)
_DATA_EXTS = {".parquet", ".csv", ".tsv", ".jsonl", ".json", ".arrow"}


def _detect_format(path):
    """Return format tag from file extension."""
    ext = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""
    return ext.lstrip(".") if ext in _DATA_EXTS else None


def _list_data_files(spec, download_api):
    """List downloadable data files for a dataset, sorted by name.

    Handles the various directory layouts HF datasets use:
      {config}/train-*.parquet           (e.g. finemath)
      data/{subdir}/train-*.parquet      (e.g. fineweb-edu)
      {subdir}/000000.parquet            (e.g. fineweb-edu-chinese)
      train.csv                          (e.g. codesearchnet)
      data.jsonl                         (e.g. codefeedback)

    Returns list of path_in_repo strings.
    """
    hf_path = spec["hf_path"]
    config = spec.get("config")
    split = spec.get("split", "train")

    all_files = list(download_api.list_repo_files(
        repo_id=hf_path, repo_type="dataset",
    ))

    data_files = [f for f in all_files if _detect_format(f)]

    # Filter by config if specified
    if config:
        filtered = [f for f in data_files if f.startswith(f"{config}/")]
        if filtered:
            data_files = filtered
        else:
            data_files = [f for f in data_files if config in f]

    # For non-parquet files (csv, jsonl, json), try to filter by split name.
    # E.g. train.csv matches split "train", test.csv does not.
    non_parquet = [f for f in data_files if not f.endswith(".parquet")]
    if non_parquet:
        split_matched = [f for f in non_parquet
                         if os.path.basename(f).startswith(f"{split}.")
                         or os.path.basename(f).startswith(f"{split}-")]
        if split_matched:
            # Replace non-parquet entries with split-filtered ones
            data_files = [f for f in data_files if f.endswith(".parquet")] + split_matched

    data_files.sort()
    return data_files


# ---------------------------------------------------------------------------
# Row iteration over various file formats
# ---------------------------------------------------------------------------

def _iter_rows(local_path, fmt):
    """Yield dicts from a data file (parquet, csv, jsonl, json)."""
    if fmt == "parquet":
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(local_path)
        for rg in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(rg)
            columns = table.column_names
            for i in range(table.num_rows):
                yield {col: table.column(col)[i].as_py() for col in columns}
            del table

    elif fmt in ("csv", "tsv"):
        import csv
        with open(local_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t" if fmt == "tsv" else ",")
            for row in reader:
                yield dict(row)

    elif fmt == "jsonl":
        import json
        with open(local_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    elif fmt == "json":
        import json
        with open(local_path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            yield from data
        else:
            yield data

    elif fmt == "arrow":
        import pyarrow.ipc as ipc
        with ipc.open_file(local_path) as reader:
            table = reader.read_all()
            columns = table.column_names
            for i in range(table.num_rows):
                yield {col: table.column(col)[i].as_py() for col in columns}

    else:
        raise ValueError(f"Unsupported format: {fmt}")


def _flush_buf(buf, name, shard_idx, dry_run, uploader):
    """Flush token buffer to shard file."""
    if not buf or dry_run:
        return
    path = _flush_shard(buf, name, shard_idx)
    if uploader is not None:
        uploader.add(path)


def _safe_delete(local_path, cache_dir):
    """Delete downloaded file and clean up empty parent dirs."""
    if not local_path or not os.path.exists(local_path):
        return
    os.remove(local_path)
    try:
        parent = os.path.dirname(local_path)
        while parent != cache_dir:
            if not os.listdir(parent):
                os.rmdir(parent)
                parent = os.path.dirname(parent)
            else:
                break
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Per-dataset processing
# ---------------------------------------------------------------------------

MAX_DOWNLOAD_RETRIES = 5


def process_spec(spec, tokenizer, *, download_api=None, uploader=None,
                 scale=1.0, no_resume=False, dry_run=False, cache_dir=None):
    name = spec["name"]
    target = int(spec["target_tokens"] * scale)
    min_doc_length = spec.get("min_doc_length", DEFAULT_MIN_DOC_LENGTH)
    done_tokens = 0
    shard_idx = 0

    if cache_dir is None:
        cache_dir = os.path.join(OUTPUT_DIR, "_cache")

    # Resume from existing shards
    if not no_resume and not dry_run:
        done_tokens, shard_idx = _resume_progress(name, uploader=uploader)

    if done_tokens >= target:
        print(f"  {name}: already met ({done_tokens:,} >= {target:,}), skipping.")
        return done_tokens

    print(f"\n{'=' * 60}")
    print(f"  {name}  ({spec['hf_path']})")
    print(f"  Target: {target:,} | Done: {done_tokens:,} | Shard: {shard_idx}")
    print(f"{'=' * 60}")

    # List data files
    if download_api is None:
        print(f"  [{name}] No download API, cannot list files.")
        return done_tokens

    print(f"  [{name}] Listing data files from {spec['hf_path']} ...", flush=True)
    data_files = _list_data_files(spec, download_api)
    print(f"  [{name}] Found {len(data_files)} data files")

    if not data_files:
        print(f"  [{name}] WARNING: no data files found, skipping.")
        return done_tokens

    # Load progress (which files already processed)
    processed = _load_progress(name, cache_dir) if not no_resume else set()

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

    os.makedirs(cache_dir, exist_ok=True)

    for file_idx, repo_path in enumerate(data_files):
        if collected + len(buf) >= target:
            break

        # Skip already-processed files
        if repo_path in processed:
            continue

        fmt = _detect_format(repo_path)

        # Download file
        local_path = None
        for retry in range(1, MAX_DOWNLOAD_RETRIES + 1):
            try:
                print(f"  [{name}] Downloading ({file_idx + 1}/{len(data_files)}) "
                      f"{os.path.basename(repo_path)} ...", end="", flush=True)
                local_path = hf_hub_download(
                    repo_id=spec["hf_path"],
                    filename=repo_path,
                    repo_type="dataset",
                    cache_dir=cache_dir,
                )
                fsize = os.path.getsize(local_path) if local_path else 0
                print(f" done ({_fmt_bytes(fsize)})")
                break
            except KeyboardInterrupt:
                raise
            except Exception as e:
                if retry >= MAX_DOWNLOAD_RETRIES:
                    print(f"\n  [{name}] FAILED to download {repo_path} "
                          f"after {MAX_DOWNLOAD_RETRIES} retries: {e}")
                    _flush_buf(buf, name, shard_idx, dry_run, uploader)
                    raise
                wait = min(10 * retry, 120)
                print(f"\n  [{name}] Download error (attempt {retry}/{MAX_DOWNLOAD_RETRIES}): {e}")
                print(f"  [{name}] Retrying in {wait}s ...")
                time.sleep(wait)

        if local_path is None:
            continue

        # Process file based on format
        try:
            rows_iter = _iter_rows(local_path, fmt)

            for row in rows_iter:
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

                    # Progress report
                    now = time.time()
                    if now - last_print_time >= 30 or collected >= target:
                        elapsed = now - start
                        rate = (collected - done_tokens) / elapsed if elapsed > 0 else 0
                        pct = collected / max(target, 1) * 100
                        remaining = (target - collected) / rate if rate > 0 else -1
                        disk_mb = collected * 4 / 1e6
                        print(f"  [{name}] {pct:5.1f}% | "
                              f"{_fmt_tokens(collected)}/{_fmt_tokens(target)} tok | "
                              f"{_fmt_speed(rate)} | "
                              f"ETA {_fmt_duration(remaining)} | "
                              f"{docs:,} docs | "
                              f"{_fmt_duration(elapsed)} elapsed"
                              f"{f' | {disk_mb:.0f} MB disk' if not dry_run else ''}")
                        last_print_time = now

                    if collected >= target:
                        break

            # Mark file as processed and delete it
            processed.add(repo_path)
            if not dry_run:
                _save_progress(name, cache_dir, processed)
                _safe_delete(local_path, cache_dir)

            gc.collect()

        except Exception as e:
            print(f"  [{name}] Error processing {repo_path}: {e}")
            if buf and not dry_run:
                collected += len(buf)
                path = _flush_shard(buf, name, shard_idx)
                if uploader is not None:
                    uploader.add(path)
                shard_idx += 1
                buf = _array.array("i")
            gc.collect()

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
    p = argparse.ArgumentParser(description="Download parquet -> tokenize -> save/upload")
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
    p.add_argument("--output-dir", type=str, default=None,
                   help=f"Output directory for .npy shards (default: {OUTPUT_DIR})")
    p.add_argument("--cache-dir", type=str, default=None,
                   help="Directory for temporary parquet downloads (default: {OUTPUT_DIR}/_cache)")
    p.add_argument("--download-only", action="store_true",
                   help="Download parquet files only, skip tokenization")
    p.add_argument("--skip-download", action="store_true",
                   help="Skip download, process already-cached parquet files")
    return p.parse_args()


def main():
    global OUTPUT_DIR
    args = parse_args()

    if args.output_dir:
        OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cache_dir = args.cache_dir or os.path.join(OUTPUT_DIR, "_cache")

    need_upload = not (args.dry_run or args.local_only)

    if need_upload and not args.hf_token:
        print("Error: --hf-token is required when uploading to HuggingFace.",
              file=sys.stderr)
        print("Use --local-only to save locally, or --dry-run to count only.",
              file=sys.stderr)
        sys.exit(1)

    # Mirror: set HF_ENDPOINT so hf_hub_download / AutoTokenizer go through
    # the mirror. The uploader HfApi gets an explicit endpoint so uploads
    # always go to the real HuggingFace.
    mirror = None if args.no_mirror else MIRROR_ENDPOINT
    if mirror:
        os.environ["HF_ENDPOINT"] = mirror
        print(f"Mirror: {mirror}")
    else:
        print("Mirror: disabled (--no-mirror)")

    # Set up download API (for listing and downloading parquet files)
    download_api = HfApi(endpoint=mirror or "https://huggingface.co")

    # Set up uploader (only for upload mode)
    uploader = None
    if need_upload:
        api = HfApi(token=args.hf_token, endpoint="https://huggingface.co")
        try:
            api.repo_info(repo_id=HF_TARGET_REPO, repo_type="dataset")
        except Exception:
            api.create_repo(repo_id=HF_TARGET_REPO, repo_type="dataset", private=False)
        uploader = BatchUploader(api, HF_TARGET_REPO, args.batch_size)

    mode_str = "DRY RUN" if args.dry_run else ("LOCAL ONLY" if args.local_only else "UPLOAD")

    cfg = load_datasets_config()
    specs = build_specs(cfg)
    if args.datasets:
        names = set(args.datasets.split(","))
        specs = [s for s in specs if s["name"] in names]

    # Print startup summary
    scaled_total = sum(int(s["target_tokens"] * args.scale) for s in specs)
    print(f"\n{'=' * 60}")
    print(f"  Mode:       {mode_str}")
    print(f"  Output:     {os.path.abspath(OUTPUT_DIR)}")
    print(f"  Cache:      {os.path.abspath(cache_dir)}")
    print(f"  Datasets:   {len(specs)}")
    print(f"  Scale:      {args.scale}")
    print(f"  Target:     {_fmt_tokens(scaled_total)} tokens")
    print(f"{'=' * 60}")

    # Download tokenizer
    print(f"\n[1/2] Downloading tokenizer: {TOKENIZER_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    print(f"      Tokenizer ready (vocab={len(tokenizer):,})")

    no_resume = bool(args.verify)
    if args.verify:
        specs = [dict(s, target_tokens=FLUSH_EVERY * 2) for s in specs]
        print("\nVERIFY MODE: 2 shards per dataset.\n")

    # Print dataset plan
    print(f"\n[2/2] Dataset plan ({len(specs)} datasets):")
    for i, s in enumerate(specs, 1):
        t = _fmt_tokens(int(s["target_tokens"] * args.scale))
        print(f"      {i}. {s['name']}: {t} tokens from {s['hf_path']}")
    print()

    grand = 0
    for idx, spec in enumerate(specs, 1):
        print(f"--- Dataset {idx}/{len(specs)} ---")
        grand += process_spec(spec, tokenizer, download_api=download_api,
                              uploader=uploader, scale=args.scale,
                              no_resume=no_resume, dry_run=args.dry_run,
                              cache_dir=cache_dir)

    if uploader is not None:
        uploader.finish()

    print(f"\n{'=' * 60}")
    print(f"  Done! {_fmt_tokens(grand)} tokens processed")
    print(f"{'=' * 60}")
    if args.local_only:
        print(f"  Shards saved to {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
