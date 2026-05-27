#!/usr/bin/env python3
"""Download HF dataset files -> tokenize -> save locally and/or upload to HF.

Reads dataset specs from configs/production/datasets.yaml or --datasets-config.

Pipeline: list parquet files -> download one at a time -> read with pyarrow ->
          filter -> tokenize -> .npy shard -> delete parquet -> next file.

This avoids HF streaming which is unreliable on some networks. Each dataset's
file order is deterministically shuffled before processing so partial runs do
not only consume lexicographically early shards. Each file is downloaded into an
isolated temporary cache, processed, and the cache is deleted before continuing.

Modes:
  (default)       Save .npy shards locally and upload to HuggingFace.
  --plan-only      Print remote file counts/sizes and local disk estimates.
  --local-only     Save .npy shards locally only, skip HF upload.
  --dry-run        Count tokens only, no disk writes or uploads.
  --verify         2 shards per dataset, no resume (quick smoke test).
"""

import argparse
import array as _array
import gc
import hashlib
import json
import os
import random
import re
import shutil
import sys
import time

import yaml

try:
    from huggingface_hub import HfApi, hf_hub_download
except ModuleNotFoundError as exc:
    raise SystemExit("Run `uv sync --extra production --frozen` first.") from exc

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
OUTPUT_DIR = "data/production/tokenized"
FLUSH_EVERY = 5_000_000          # tokens per shard (~20 MB)
COMMIT_BATCH_SIZE = 50           # shards per commit (~1 GB)
MIRROR_ENDPOINT = "https://hf-mirror.com"
DEFAULT_MIN_DOC_LENGTH = 50
# Conservative estimate: 1 byte of parquet -> ~0.15 tokens (accounts for
# compression, metadata, and mixed-language content).
BYTES_PER_TOKEN_ESTIMATE = 1 / 0.15
DEFAULT_SAMPLING_SEED = 42
DEFAULT_FILE_ORDER = "shuffled"
DEFAULT_ROW_GROUP_ORDER = "shuffled"
DEFAULT_ROW_ORDER = "shuffled"
DEFAULT_JSON_SHUFFLE_BUFFER = 4096
DEFAULT_MIN_FREE_GB = 80
DEFAULT_MAX_CACHE_GB = 60


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


def _parse_gb(value, default):
    if value is None:
        return int(default * 1e9)
    return int(float(value) * 1e9)


def _stable_seed(*parts):
    """Build a deterministic 32-bit seed from config strings."""
    text = "::".join(str(p) for p in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _cache_size_bytes(path):
    if not os.path.exists(path):
        return 0
    total = 0
    for root, _, files in os.walk(path):
        for filename in files:
            try:
                total += os.path.getsize(os.path.join(root, filename))
            except OSError:
                pass
    return total


def _ensure_disk_budget(path, *, min_free_bytes, max_cache_bytes=None,
                        cache_path=None, incoming_bytes=None):
    """Abort before the next large operation if disk budget is unsafe."""
    os.makedirs(path, exist_ok=True)
    free = shutil.disk_usage(path).free
    if free < min_free_bytes:
        raise RuntimeError(
            f"Free disk under safety floor: {_fmt_bytes(free)} available, "
            f"need at least {_fmt_bytes(min_free_bytes)} free."
        )

    if incoming_bytes and free - incoming_bytes < min_free_bytes:
        raise RuntimeError(
            f"Next file is {_fmt_bytes(incoming_bytes)}, which would leave "
            f"{_fmt_bytes(free - incoming_bytes)} free under the safety floor "
            f"({_fmt_bytes(min_free_bytes)})."
        )

    if incoming_bytes and max_cache_bytes is not None and incoming_bytes > max_cache_bytes:
        raise RuntimeError(
            f"Next file is {_fmt_bytes(incoming_bytes)}, above the configured "
            f"temporary cache limit {_fmt_bytes(max_cache_bytes)}. Increase "
            "--max-cache-gb if this source file is expected."
        )

    if cache_path and max_cache_bytes is not None:
        cache_size = _cache_size_bytes(cache_path)
        if cache_size > max_cache_bytes:
            raise RuntimeError(
                f"Download cache is {_fmt_bytes(cache_size)}, above the "
                f"configured limit {_fmt_bytes(max_cache_bytes)}. Remove "
                f"{cache_path} or increase --max-cache-gb."
            )


def _clean_download_cache(download_cache_dir):
    """Remove HF hub blobs for the last downloaded file."""
    if os.path.exists(download_cache_dir):
        shutil.rmtree(download_cache_dir)
    os.makedirs(download_cache_dir, exist_ok=True)


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


def _coerce_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_existing(row, columns):
    for col in columns:
        value = row.get(col)
        if value not in (None, ""):
            return value
    return None


def _extract_text(row, spec):
    """Extract text from a row using a configured function or column list."""
    text_fn = spec.get("text_fn")
    if text_fn:
        return text_fn(row)

    columns = spec.get("text_columns") or spec.get("text_column")
    if isinstance(columns, str):
        columns = [columns]
    if not columns:
        return ""

    if spec.get("join_text_columns"):
        parts = [str(row.get(col, "") or "").strip() for col in columns]
        return "\n".join(p for p in parts if p)
    value = _first_existing(row, columns)
    return str(value or "")


def _compare(value, op, expected):
    if op == ">=":
        return value >= expected
    if op == ">":
        return value > expected
    if op == "<=":
        return value <= expected
    if op == "<":
        return value < expected
    if op in ("=", "=="):
        return value == expected
    if op == "!=":
        return value != expected
    raise ValueError(f"Unsupported filter op: {op}")


def _chinese_ratio(text):
    if not text:
        return 0.0
    cn = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    return cn / max(len(text), 1)


def _passes_filter_config(row, text, filter_cfg):
    """Generic YAML-driven filters used by production datasets."""
    if not filter_cfg:
        return True

    min_length = filter_cfg.get("min_length")
    if min_length is not None and len(text) < int(min_length):
        return False

    max_length = filter_cfg.get("max_length")
    if max_length is not None and len(text) > int(max_length):
        return False

    min_chinese_ratio = filter_cfg.get("min_chinese_ratio")
    if min_chinese_ratio is None and filter_cfg.get("type") == "chinese":
        min_chinese_ratio = filter_cfg.get("min_ratio")
    if min_chinese_ratio is not None and _chinese_ratio(text) < float(min_chinese_ratio):
        return False

    allowed_languages = filter_cfg.get("allowed_languages")
    if allowed_languages:
        lang = row.get(filter_cfg.get("language_column", "language"))
        if lang not in set(allowed_languages):
            return False

    min_quality_score = filter_cfg.get("min_quality_score")
    if min_quality_score is not None:
        columns = (
            filter_cfg.get("quality_columns")
            or filter_cfg.get("quality_column")
            or ["score", "int_score", "quality_score", "edu_score"]
        )
        if isinstance(columns, str):
            columns = [columns]
        score = _coerce_float(_first_existing(row, columns))
        if score is None or score < float(min_quality_score):
            return False

    # Backward-compatible named filters from the old config format:
    # {field: int_score, op: >=, value: 2}
    if "field" in filter_cfg:
        field_value = row.get(filter_cfg["field"])
        expected = filter_cfg.get("value")
        left = _coerce_float(field_value)
        right = _coerce_float(expected)
        if left is not None and right is not None:
            field_value, expected = left, right
        if field_value is None or not _compare(field_value, filter_cfg.get("op", "=="), expected):
            return False

    return True


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
    default_sampling = cfg.get("sampling", {})
    named_filters = cfg.get("filters", {})
    specs = []
    for ds in cfg["datasets"]:
        if ds.get("enabled", True) is False:
            continue

        spec = dict(ds)
        if "source" in spec and "hf_path" not in spec:
            spec["hf_path"] = spec["source"]
        if "subset" in spec and "config" not in spec:
            spec["config"] = spec["subset"]
        if "text_field" in spec and "text_column" not in spec:
            spec["text_column"] = spec["text_field"]

        # Resolve sampling defaults. Per-dataset values override top-level.
        sampling = dict(default_sampling)
        sampling.update(ds.get("sampling", {}) or {})
        spec["sampling"] = sampling

        # Resolve legacy named filters and newer inline filter blocks.
        fname = ds.get("filter")
        filter_cfg = {}
        if fname and fname in named_filters:
            filter_cfg.update(named_filters[fname] or {})
        filter_cfg.update(ds.get("filters", {}) or {})
        if "min_chars" in filter_cfg and "min_length" not in filter_cfg:
            filter_cfg["min_length"] = filter_cfg["min_chars"]
        if "max_chars" in filter_cfg and "max_length" not in filter_cfg:
            filter_cfg["max_length"] = filter_cfg["max_chars"]
        if "add_eos" in filter_cfg and "add_eos" not in spec:
            spec["add_eos"] = filter_cfg["add_eos"]
        spec["filter_config"] = filter_cfg
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
    def __init__(
        self,
        api,
        repo_id,
        target_path,
        batch_size=COMMIT_BATCH_SIZE,
        keep_uploaded=False,
    ):
        self.api = api
        self.repo_id = repo_id
        self.target_path = target_path.strip("/")
        self.batch_size = batch_size
        self.keep_uploaded = keep_uploaded
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
                path_in_repo=f"{self.target_path}/{os.path.basename(p)}",
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
                if not self.keep_uploaded:
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
            repo_id=uploader.repo_id,
            repo_type="dataset",
            path_in_repo=f"{uploader.target_path}/{name}",
        ))
        remote_shards = [f for f in remote_files if f.endswith(".npy")]
        remote_count = len(remote_shards)
        # Conservative: last remote shard may be partial
        remote_tokens = max(remote_count - 1, 0) * FLUSH_EVERY

        # Remove local files already on remote
        remote_set = set(remote_shards)
        for lf in local_files:
            if f"{uploader.target_path}/{os.path.basename(lf)}" in remote_set:
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


def _list_repo_file_infos(download_api, hf_path):
    """Return path/size dicts for files in an HF dataset repo.

    Uses list_repo_files for speed; size info is omitted as it's not critical
    for our workflow and list_repo_tree is extremely slow for large repos.
    """
    return [
        {"path": path, "size": None}
        for path in download_api.list_repo_files(repo_id=hf_path, repo_type="dataset")
    ]


def _order_data_files(data_files, spec):
    sampling = spec.get("sampling", {})
    mode = sampling.get("file_order", DEFAULT_FILE_ORDER)
    seed = sampling.get("seed", DEFAULT_SAMPLING_SEED)
    ordered = sorted(data_files, key=lambda f: f["path"])
    if mode == "sorted":
        return ordered
    if mode != "shuffled":
        raise ValueError(f"Unsupported file_order: {mode}")
    rng = random.Random(_stable_seed(seed, spec["name"], spec["hf_path"], "files"))
    rng.shuffle(ordered)
    return ordered


def _list_data_files(spec, download_api):
    """List downloadable data files for a dataset.

    Handles the various directory layouts HF datasets use:
      {config}/train-*.parquet           (e.g. finemath)
      data/{subdir}/train-*.parquet      (e.g. fineweb-edu)
      {subdir}/000000.parquet            (e.g. fineweb-edu-chinese)
      train.csv                          (e.g. codesearchnet)
      data.jsonl                         (e.g. codefeedback)

    Returns list of {"path": path_in_repo, "size": bytes_or_none} dicts. The
    order is deterministic and shuffled by default to avoid prefix bias.
    """
    hf_path = spec["hf_path"]
    config = spec.get("config")
    split = spec.get("split", "train")

    all_files = _list_repo_file_infos(download_api, hf_path)

    data_files = [f for f in all_files if _detect_format(f["path"])]

    # Filter by config if specified
    if config:
        filtered = [f for f in data_files if f["path"].startswith(f"{config}/")]
        if filtered:
            data_files = filtered
        else:
            data_files = [f for f in data_files if config in f["path"]]

    # For non-parquet files (csv, jsonl, json), try to filter by split name.
    # E.g. train.csv matches split "train", test.csv does not.
    non_parquet = [f for f in data_files if not f["path"].endswith(".parquet")]
    if non_parquet:
        split_matched = [f for f in non_parquet
                         if os.path.basename(f["path"]).startswith(f"{split}.")
                         or os.path.basename(f["path"]).startswith(f"{split}-")]
        if split_matched:
            # Replace non-parquet entries with split-filtered ones
            data_files = [f for f in data_files if f["path"].endswith(".parquet")] + split_matched

    return _order_data_files(data_files, spec)


# ---------------------------------------------------------------------------
# Row iteration over various file formats
# ---------------------------------------------------------------------------

def _row_order(spec, repo_path, salt):
    sampling = spec.get("sampling", {})
    seed = sampling.get("seed", DEFAULT_SAMPLING_SEED)
    return random.Random(_stable_seed(seed, spec["name"], repo_path, salt))


def _maybe_shuffle(items, spec, repo_path, salt, mode_key, default_mode):
    sampling = spec.get("sampling", {})
    mode = sampling.get(mode_key, default_mode)
    items = list(items)
    if mode == "sorted":
        return items
    if mode != "shuffled":
        raise ValueError(f"Unsupported {mode_key}: {mode}")
    _row_order(spec, repo_path, salt).shuffle(items)
    return items


def _flush_row_buffer(buf, spec, repo_path, salt):
    if not buf:
        return []
    sampling = spec.get("sampling", {})
    mode = sampling.get("row_order", DEFAULT_ROW_ORDER)
    rows = list(buf)
    if mode == "shuffled":
        _row_order(spec, repo_path, salt).shuffle(rows)
    elif mode != "sorted":
        raise ValueError(f"Unsupported row_order: {mode}")
    return rows


def _iter_rows(local_path, fmt, *, spec, repo_path):
    """Yield dicts from a data file (parquet, csv, jsonl, json)."""
    if fmt == "parquet":
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(local_path)
        row_groups = _maybe_shuffle(
            range(pf.metadata.num_row_groups),
            spec,
            repo_path,
            "row-groups",
            "row_group_order",
            DEFAULT_ROW_GROUP_ORDER,
        )
        for rg in row_groups:
            table = pf.read_row_group(rg)
            columns = table.column_names
            row_ids = _maybe_shuffle(
                range(table.num_rows),
                spec,
                repo_path,
                f"rows:{rg}",
                "row_order",
                DEFAULT_ROW_ORDER,
            )
            for i in row_ids:
                yield {col: table.column(col)[i].as_py() for col in columns}
            del table

    elif fmt in ("csv", "tsv"):
        import csv
        buffer_size = int(spec.get("sampling", {}).get(
            "json_shuffle_buffer", DEFAULT_JSON_SHUFFLE_BUFFER,
        ))
        with open(local_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t" if fmt == "tsv" else ",")
            buf = []
            batch_idx = 0
            for row in reader:
                buf.append(dict(row))
                if len(buf) >= buffer_size:
                    yield from _flush_row_buffer(buf, spec, repo_path, f"csv:{batch_idx}")
                    buf = []
                    batch_idx += 1
            yield from _flush_row_buffer(buf, spec, repo_path, f"csv:{batch_idx}")

    elif fmt == "jsonl":
        import json
        buffer_size = int(spec.get("sampling", {}).get(
            "json_shuffle_buffer", DEFAULT_JSON_SHUFFLE_BUFFER,
        ))
        with open(local_path, encoding="utf-8") as f:
            buf = []
            batch_idx = 0
            for line in f:
                line = line.strip()
                if line:
                    buf.append(json.loads(line))
                if len(buf) >= buffer_size:
                    yield from _flush_row_buffer(buf, spec, repo_path, f"jsonl:{batch_idx}")
                    buf = []
                    batch_idx += 1
            yield from _flush_row_buffer(buf, spec, repo_path, f"jsonl:{batch_idx}")

    elif fmt == "json":
        import json
        with open(local_path, encoding="utf-8") as f:
            first = f.read(1)
            f.seek(0)
            if first == "[":
                data = json.load(f)
                rows = _flush_row_buffer(data, spec, repo_path, "json")
                yield from rows
            else:
                buffer_size = int(spec.get("sampling", {}).get(
                    "json_shuffle_buffer", DEFAULT_JSON_SHUFFLE_BUFFER,
                ))
                buf = []
                batch_idx = 0
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    buf.append(json.loads(line))
                    if len(buf) >= buffer_size:
                        yield from _flush_row_buffer(buf, spec, repo_path, f"json:{batch_idx}")
                        buf = []
                        batch_idx += 1
                yield from _flush_row_buffer(buf, spec, repo_path, f"json:{batch_idx}")

    elif fmt == "arrow":
        import pyarrow.ipc as ipc
        with ipc.open_file(local_path) as reader:
            table = reader.read_all()
            columns = table.column_names
            row_ids = _maybe_shuffle(
                range(table.num_rows),
                spec,
                repo_path,
                "arrow-rows",
                "row_order",
                DEFAULT_ROW_ORDER,
            )
            for i in row_ids:
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


# ---------------------------------------------------------------------------
# Per-dataset processing
# ---------------------------------------------------------------------------

MAX_DOWNLOAD_RETRIES = 5


def process_spec(spec, tokenizer, *, download_api=None, uploader=None,
                 scale=1.0, no_resume=False, dry_run=False, cache_dir=None,
                 min_free_bytes=None, max_cache_bytes=None, hf_token=None):
    name = spec["name"]
    target = int(spec["target_tokens"] * scale)
    min_doc_length = spec.get("min_doc_length", DEFAULT_MIN_DOC_LENGTH)
    done_tokens = 0
    shard_idx = 0

    if cache_dir is None:
        cache_dir = os.path.join(OUTPUT_DIR, "_cache")
    state_dir = os.path.join(cache_dir, "state")
    download_cache_dir = os.path.join(cache_dir, "downloads")
    min_free_bytes = min_free_bytes if min_free_bytes is not None else _parse_gb(None, DEFAULT_MIN_FREE_GB)
    max_cache_bytes = max_cache_bytes if max_cache_bytes is not None else _parse_gb(None, DEFAULT_MAX_CACHE_GB)

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
    processed = _load_progress(name, state_dir) if not no_resume else set()

    eos_id = tokenizer.eos_token_id
    filter_fn = spec.get("filter_fn")
    filter_cfg = spec.get("filter_config", {})
    add_eos = bool(spec.get("add_eos", True))

    buf = _array.array("i")
    collected = done_tokens
    docs = 0
    skipped = 0
    start = time.time()
    last_print_time = start

    os.makedirs(state_dir, exist_ok=True)
    os.makedirs(download_cache_dir, exist_ok=True)
    sampling = spec.get("sampling", {})
    print(f"  [{name}] Sampling: file_order={sampling.get('file_order', DEFAULT_FILE_ORDER)}, "
          f"row_group_order={sampling.get('row_group_order', DEFAULT_ROW_GROUP_ORDER)}, "
          f"row_order={sampling.get('row_order', DEFAULT_ROW_ORDER)}, "
          f"seed={sampling.get('seed', DEFAULT_SAMPLING_SEED)}")

    for file_idx, file_info in enumerate(data_files):
        if collected + len(buf) >= target:
            break

        repo_path = file_info["path"]

        # Skip already-processed files
        if repo_path in processed:
            continue

        fmt = _detect_format(repo_path)
        file_size = file_info.get("size")
        _ensure_disk_budget(
            OUTPUT_DIR,
            min_free_bytes=min_free_bytes,
            max_cache_bytes=max_cache_bytes,
            cache_path=download_cache_dir,
            incoming_bytes=file_size,
        )

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
                    cache_dir=download_cache_dir,
                    token=hf_token,
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
                    _clean_download_cache(download_cache_dir)
                    raise
                wait = min(10 * retry, 120)
                print(f"\n  [{name}] Download error (attempt {retry}/{MAX_DOWNLOAD_RETRIES}): {e}")
                print(f"  [{name}] Retrying in {wait}s ...")
                time.sleep(wait)

        if local_path is None:
            continue

        # Process file based on format
        try:
            rows_iter = _iter_rows(local_path, fmt, spec=spec, repo_path=repo_path)
            file_completed = True

            for row in rows_iter:
                if collected + len(buf) >= target:
                    file_completed = False
                    break

                text = _extract_text(row, spec)
                if filter_fn and not filter_fn(row):
                    continue
                if not _passes_filter_config(row, text, filter_cfg):
                    skipped += 1
                    continue
                if not text or len(text.strip()) < min_doc_length:
                    skipped += 1
                    continue
                buf.extend(tokenizer.encode(text, add_special_tokens=False))
                if add_eos and eos_id is not None:
                    buf.append(eos_id)
                docs += 1

                if len(buf) >= FLUSH_EVERY:
                    collected += len(buf)
                    if dry_run:
                        buf = _array.array("i")
                    else:
                        _ensure_disk_budget(OUTPUT_DIR, min_free_bytes=min_free_bytes)
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
                        file_completed = False
                        break

            # Mark even a partially consumed final file as done. That avoids
            # duplicate samples when a later run increases --scale; the loss is
            # only the unused tail of one deterministically shuffled file.
            processed.add(repo_path)
            if not dry_run and not file_completed:
                print(f"  [{name}] Target reached inside {repo_path}; "
                      "marking file consumed to avoid duplicate samples.")
            if not dry_run:
                _save_progress(name, state_dir, processed)
            _clean_download_cache(download_cache_dir)

            gc.collect()

        except Exception as e:
            print(f"  [{name}] Error processing {repo_path}: {e}")
            if buf and not dry_run:
                collected += len(buf)
                _ensure_disk_budget(OUTPUT_DIR, min_free_bytes=min_free_bytes)
                path = _flush_shard(buf, name, shard_idx)
                if uploader is not None:
                    uploader.add(path)
                shard_idx += 1
                buf = _array.array("i")
            _clean_download_cache(download_cache_dir)
            gc.collect()

    # Flush remaining buffer
    if buf:
        collected += len(buf)
        if not dry_run:
            _ensure_disk_budget(OUTPUT_DIR, min_free_bytes=min_free_bytes)
            path = _flush_shard(buf, name, shard_idx)
            if uploader is not None:
                uploader.add(path)

    print(f"  [{name}] done: {collected:,} tokens from {docs:,} docs "
          f"({skipped:,} filtered) in {_fmt_duration(time.time() - start)}")
    return collected


def print_plan_only(specs, download_api, *, scale):
    """Print remote file counts and local shard estimates without downloads."""
    print(f"\n{'=' * 60}")
    print("  PLAN ONLY")
    print(f"{'=' * 60}")
    total_target = 0
    total_disk = 0
    for spec in specs:
        target = int(spec["target_tokens"] * scale)
        total_target += target
        token_disk = target * 4
        total_disk += token_disk
        try:
            files = _list_data_files(spec, download_api)
            total_remote = sum(f.get("size") or 0 for f in files)
            largest = max((f.get("size") or 0 for f in files), default=0)
            remote_text = _fmt_bytes(total_remote) if total_remote else "unknown"
            largest_text = _fmt_bytes(largest) if largest else "unknown"
            file_text = f"{len(files)} files, remote={remote_text}, largest={largest_text}"
        except Exception as exc:
            file_text = f"file listing failed: {exc}"
        print(f"  {spec['name']:24s} target={_fmt_tokens(target):>8s} "
              f"shards~{target // FLUSH_EVERY:>5,d} disk~{_fmt_bytes(token_disk):>8s} | {file_text}")
    print(f"  {'TOTAL':24s} target={_fmt_tokens(total_target):>8s} "
          f"disk~{_fmt_bytes(total_disk):>8s}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Download HF data files -> tokenize -> save/upload")
    p.add_argument("--datasets-config", type=str, default=None,
                   help="Dataset recipe YAML (default: configs/production/datasets.yaml)")
    p.add_argument("--hf-token", type=str, default=None,
                   help="HuggingFace API token (required unless --plan-only, --dry-run, or --local-only)")
    p.add_argument("--hf-repo", type=str, default=HF_TARGET_REPO,
                   help=f"HF Dataset repo for upload mode (default: {HF_TARGET_REPO})")
    p.add_argument("--hf-path", type=str, default=HF_TARGET_PATH,
                   help=f"Path inside HF Dataset repo (default: {HF_TARGET_PATH})")
    p.add_argument("--hf-private", action="store_true",
                   help="Create the HF Dataset repo as private if it does not exist")
    p.add_argument("--keep-uploaded", action="store_true",
                   help="Keep local shard files after successful HF uploads")
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
    p.add_argument("--min-free-gb", type=float, default=None,
                   help=f"Abort before downloads/writes if free disk is below this (default: {DEFAULT_MIN_FREE_GB})")
    p.add_argument("--max-cache-gb", type=float, default=None,
                   help=f"Abort if temporary HF download cache grows above this (default: {DEFAULT_MAX_CACHE_GB})")
    p.add_argument("--plan-only", action="store_true",
                   help="List shuffled file counts/sizes and token targets; no tokenizer download or data files")
    return p.parse_args()


def main():
    global OUTPUT_DIR
    args = parse_args()

    cfg = load_datasets_config(args.datasets_config)

    if args.output_dir:
        OUTPUT_DIR = args.output_dir
    elif cfg.get("output_dir"):
        OUTPUT_DIR = cfg["output_dir"]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cache_dir = args.cache_dir or cfg.get("cache_dir") or os.path.join(OUTPUT_DIR, "_cache")

    need_upload = not (args.dry_run or args.local_only or args.plan_only)

    if need_upload and not args.hf_token:
        print("Error: --hf-token is required when uploading to HuggingFace.",
              file=sys.stderr)
        print("Use --local-only to save locally, --dry-run to count only, or --plan-only to inspect.",
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

    # Set up download API (for listing and downloading source files). Passing
    # the token here lets local-only runs access gated datasets after the user
    # accepts their terms on HuggingFace.
    download_api = HfApi(token=args.hf_token, endpoint=mirror or "https://huggingface.co")

    # Set up uploader (only for upload mode)
    uploader = None
    if need_upload:
        api = HfApi(token=args.hf_token, endpoint="https://huggingface.co")
        try:
            api.repo_info(repo_id=args.hf_repo, repo_type="dataset")
        except Exception:
            api.create_repo(repo_id=args.hf_repo, repo_type="dataset", private=args.hf_private)
        uploader = BatchUploader(
            api,
            args.hf_repo,
            args.hf_path,
            args.batch_size,
            keep_uploaded=args.keep_uploaded,
        )

    mode_str = (
        "PLAN ONLY" if args.plan_only
        else "DRY RUN" if args.dry_run
        else "LOCAL ONLY" if args.local_only
        else "UPLOAD"
    )

    disk_cfg = cfg.get("disk", {})
    min_free_gb = args.min_free_gb if args.min_free_gb is not None else disk_cfg.get("min_free_gb")
    max_cache_gb = args.max_cache_gb if args.max_cache_gb is not None else disk_cfg.get("max_cache_gb")
    min_free_bytes = _parse_gb(min_free_gb, DEFAULT_MIN_FREE_GB)
    max_cache_bytes = _parse_gb(max_cache_gb, DEFAULT_MAX_CACHE_GB)

    specs = build_specs(cfg)
    if args.datasets:
        names = set(args.datasets.split(","))
        specs = [s for s in specs if s["name"] in names]

    # Print startup summary
    scaled_total = sum(int(s["target_tokens"] * args.scale) for s in specs)
    print(f"\n{'=' * 60}")
    print(f"  Mode:       {mode_str}")
    print(f"  Recipe:     {os.path.abspath(args.datasets_config) if args.datasets_config else 'configs/production/datasets.yaml'}")
    print(f"  Output:     {os.path.abspath(OUTPUT_DIR)}")
    print(f"  Cache:      {os.path.abspath(cache_dir)}")
    if need_upload:
        print(f"  HF upload:  {args.hf_repo}/{args.hf_path.strip('/')}")
    print(f"  Disk floor: {_fmt_bytes(min_free_bytes)} free")
    print(f"  Cache cap:  {_fmt_bytes(max_cache_bytes)}")
    print(f"  Datasets:   {len(specs)}")
    print(f"  Scale:      {args.scale}")
    print(f"  Target:     {_fmt_tokens(scaled_total)} tokens")
    print(f"{'=' * 60}")

    if args.plan_only:
        print_plan_only(specs, download_api, scale=args.scale)
        return

    _ensure_disk_budget(
        OUTPUT_DIR,
        min_free_bytes=min_free_bytes,
        max_cache_bytes=max_cache_bytes,
        cache_path=os.path.join(cache_dir, "downloads"),
    )

    # Download tokenizer
    from transformers import AutoTokenizer

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
                              cache_dir=cache_dir,
                              min_free_bytes=min_free_bytes,
                              max_cache_bytes=max_cache_bytes,
                              hf_token=args.hf_token)

    if uploader is not None:
        uploader.finish()

    print(f"\n{'=' * 60}")
    print(f"  Done! {_fmt_tokens(grand)} tokens processed")
    print(f"{'=' * 60}")
    if args.local_only:
        print(f"  Shards saved to {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
