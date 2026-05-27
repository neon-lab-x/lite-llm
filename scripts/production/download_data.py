#!/usr/bin/env python3
"""Select and download raw pretraining documents into local parquet shards.

Stage 1 of the production data pipeline:

    HF source shards -> filtered local raw parquet documents

This script intentionally does not tokenize. It downloads one upstream data file
at a time, applies the dataset recipe's lightweight quality/language/length
filters, writes accepted rows to local parquet shards, then deletes the upstream
download cache. The next stage can tokenize these local parquet files entirely
from disk/CPU without touching HuggingFace again.
"""

import argparse
import gc
import json
import os
import sys
import time

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
except ModuleNotFoundError as exc:
    raise SystemExit("Run `uv sync --extra production --frozen` first.") from exc

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, SCRIPT_DIR)

import prepare_data as prep

DEFAULT_RAW_SHARD_EST_TOKENS = 10_000_000
MAX_DOWNLOAD_RETRIES = 5


def _default_download_dir(cfg):
    if cfg.get("raw_output_dir"):
        return cfg["raw_output_dir"]
    recipe = cfg.get("recipe_name", "recipe")
    return os.path.join(PROJECT_ROOT, "data", "production", f"raw_{recipe}")


def _state_dir(download_dir):
    return os.path.join(download_dir, "_state")


def _download_cache_dir(download_dir):
    return os.path.join(download_dir, "_cache", "downloads")


def _manifest_path(download_dir, name):
    return os.path.join(_state_dir(download_dir), f"{name}_download_manifest.json")


def _load_manifest(download_dir, name):
    path = _manifest_path(download_dir, name)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {
        "dataset": name,
        "processed_files": [],
        "estimated_tokens": 0,
        "docs": 0,
        "shards": [],
        "next_shard": 0,
    }


def _save_manifest(download_dir, name, manifest):
    os.makedirs(_state_dir(download_dir), exist_ok=True)
    path = _manifest_path(download_dir, name)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _estimate_tokens(text):
    """Cheap token estimate used only to decide when the raw subset is enough."""
    if not text:
        return 0
    cn = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    ratio = cn / max(len(text), 1)
    # Qwen tokenizer is compact on Chinese; code/English tends to use longer
    # character spans per token. Oversampling is acceptable because tokenization
    # is the authoritative second stage.
    chars_per_token = 1.5 if ratio >= 0.30 else 3.2
    return max(1, int(len(text) / chars_per_token))


class RawShardWriter:
    def __init__(self, download_dir, dataset_name, manifest, flush_est_tokens):
        self.download_dir = download_dir
        self.dataset_name = dataset_name
        self.manifest = manifest
        self.flush_est_tokens = int(flush_est_tokens)
        self.rows = []
        self.buffer_est_tokens = 0

    def add(self, *, text, est_tokens, source_repo, source_file):
        self.rows.append({
            "text": text,
            "dataset": self.dataset_name,
            "source_repo": source_repo,
            "source_file": source_file,
            "estimated_tokens": int(est_tokens),
        })
        self.buffer_est_tokens += int(est_tokens)
        return self.buffer_est_tokens >= self.flush_est_tokens

    def flush(self):
        if not self.rows:
            return None

        shard_idx = int(self.manifest.get("next_shard", 0))
        dataset_dir = os.path.join(self.download_dir, self.dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        out_path = os.path.join(dataset_dir, f"{self.dataset_name}-{shard_idx:05d}.parquet")

        table = pa.Table.from_pylist(self.rows)
        pq.write_table(table, out_path, compression="zstd")

        docs = len(self.rows)
        est_tokens = self.buffer_est_tokens
        self.manifest["shards"].append({
            "path": os.path.relpath(out_path, self.download_dir),
            "docs": docs,
            "estimated_tokens": est_tokens,
        })
        self.manifest["docs"] = int(self.manifest.get("docs", 0)) + docs
        self.manifest["estimated_tokens"] = (
            int(self.manifest.get("estimated_tokens", 0)) + est_tokens
        )
        self.manifest["next_shard"] = shard_idx + 1

        self.rows = []
        self.buffer_est_tokens = 0
        return out_path


def _passes_row(spec, row, text):
    filter_fn = spec.get("filter_fn")
    if filter_fn and not filter_fn(row):
        return False
    return prep._passes_filter_config(row, text, spec.get("filter_config", {}))


def process_spec(
    spec,
    *,
    download_api,
    download_dir,
    hf_token,
    min_free_bytes,
    max_cache_bytes,
    raw_shard_est_tokens,
    no_resume,
):
    name = spec["name"]
    target = int(spec["target_tokens"])
    manifest = (
        {
            "dataset": name,
            "processed_files": [],
            "estimated_tokens": 0,
            "docs": 0,
            "shards": [],
            "next_shard": 0,
        }
        if no_resume else _load_manifest(download_dir, name)
    )
    processed = set(manifest.get("processed_files", []))

    if int(manifest.get("estimated_tokens", 0)) >= target:
        print(
            f"  {name}: raw subset already met "
            f"({manifest['estimated_tokens']:,} >= {target:,}), skipping."
        )
        return manifest

    print(f"\n{'=' * 60}")
    print(f"  {name}  ({spec['hf_path']})")
    print(f"  Target: {target:,} estimated tokens")
    print(f"  Done:   {int(manifest.get('estimated_tokens', 0)):,} estimated tokens")
    print(f"{'=' * 60}")

    data_files = prep._list_data_files(spec, download_api)
    print(f"  [{name}] Found {len(data_files)} source files")
    if not data_files:
        print(f"  [{name}] WARNING: no source files found.")
        return manifest

    cache_dir = _download_cache_dir(download_dir)
    os.makedirs(cache_dir, exist_ok=True)
    writer = RawShardWriter(download_dir, name, manifest, raw_shard_est_tokens)

    start = time.time()
    last_report = start
    skipped = 0

    for file_idx, file_info in enumerate(data_files, 1):
        if int(manifest.get("estimated_tokens", 0)) + writer.buffer_est_tokens >= target:
            break

        repo_path = file_info["path"]
        if repo_path in processed:
            continue

        fmt = prep._detect_format(repo_path)
        prep._ensure_disk_budget(
            download_dir,
            min_free_bytes=min_free_bytes,
            max_cache_bytes=max_cache_bytes,
            cache_path=cache_dir,
            incoming_bytes=file_info.get("size"),
        )

        local_path = None
        for retry in range(1, MAX_DOWNLOAD_RETRIES + 1):
            try:
                print(
                    f"  [{name}] Downloading ({file_idx}/{len(data_files)}) "
                    f"{os.path.basename(repo_path)} ...",
                    end="",
                    flush=True,
                )
                local_path = hf_hub_download(
                    repo_id=spec["hf_path"],
                    filename=repo_path,
                    repo_type="dataset",
                    cache_dir=cache_dir,
                    token=hf_token,
                )
                print(f" done ({prep._fmt_bytes(os.path.getsize(local_path))})")
                break
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                if retry >= MAX_DOWNLOAD_RETRIES:
                    prep._clean_download_cache(cache_dir)
                    raise RuntimeError(f"Failed to download {repo_path}: {exc}") from exc
                wait = min(10 * retry, 120)
                print(f"\n  [{name}] download error: {exc}; retrying in {wait}s")
                time.sleep(wait)

        if local_path is None:
            continue

        file_completed = True
        try:
            for row in prep._iter_rows(local_path, fmt, spec=spec, repo_path=repo_path):
                text = prep._extract_text(row, spec)
                if not text or not _passes_row(spec, row, text):
                    skipped += 1
                    continue

                est_tokens = _estimate_tokens(text)
                if writer.add(
                    text=text,
                    est_tokens=est_tokens,
                    source_repo=spec["hf_path"],
                    source_file=repo_path,
                ):
                    path = writer.flush()
                    _save_manifest(download_dir, name, manifest)
                    print(f"  [{name}] wrote {path}")

                now = time.time()
                current = int(manifest.get("estimated_tokens", 0)) + writer.buffer_est_tokens
                if now - last_report >= 30 or current >= target:
                    pct = current / max(target, 1) * 100
                    print(
                        f"  [{name}] {pct:5.1f}% | "
                        f"{prep._fmt_tokens(current)}/{prep._fmt_tokens(target)} est tok | "
                        f"{int(manifest.get('docs', 0)) + len(writer.rows):,} docs | "
                        f"{skipped:,} skipped"
                    )
                    last_report = now

                if current >= target:
                    file_completed = False
                    break

            if writer.rows and (
                int(manifest.get("estimated_tokens", 0)) + writer.buffer_est_tokens >= target
            ):
                path = writer.flush()
                print(f"  [{name}] wrote {path}")

            processed.add(repo_path)
            manifest["processed_files"] = sorted(processed)
            if not file_completed:
                print(f"  [{name}] Target reached inside {repo_path}; marking file consumed.")
            _save_manifest(download_dir, name, manifest)
            prep._clean_download_cache(cache_dir)
            gc.collect()
        except Exception:
            if writer.rows:
                path = writer.flush()
                _save_manifest(download_dir, name, manifest)
                print(f"  [{name}] wrote partial {path}")
            prep._clean_download_cache(cache_dir)
            gc.collect()
            raise

    if writer.rows:
        path = writer.flush()
        _save_manifest(download_dir, name, manifest)
        print(f"  [{name}] wrote final {path}")

    elapsed = prep._fmt_duration(time.time() - start)
    print(
        f"  [{name}] done: {manifest['estimated_tokens']:,} estimated tokens, "
        f"{manifest['docs']:,} docs, {skipped:,} skipped in {elapsed}"
    )
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description="Download selected raw documents for a recipe")
    parser.add_argument("--datasets-config", default=None)
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset names")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--download-dir", default=None)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--min-free-gb", type=float, default=None)
    parser.add_argument("--max-cache-gb", type=float, default=None)
    parser.add_argument("--raw-shard-est-tokens", type=int, default=DEFAULT_RAW_SHARD_EST_TOKENS)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = prep.load_datasets_config(args.datasets_config)
    specs = prep.build_specs(cfg)
    if args.datasets:
        names = set(args.datasets.split(","))
        specs = [s for s in specs if s["name"] in names]
    if args.scale != 1.0:
        specs = [dict(s, target_tokens=int(s["target_tokens"] * args.scale)) for s in specs]

    download_dir = args.download_dir or _default_download_dir(cfg)
    os.makedirs(download_dir, exist_ok=True)

    disk_cfg = cfg.get("disk", {})
    min_free_gb = args.min_free_gb if args.min_free_gb is not None else disk_cfg.get("min_free_gb")
    max_cache_gb = args.max_cache_gb if args.max_cache_gb is not None else disk_cfg.get("max_cache_gb")
    min_free_bytes = prep._parse_gb(min_free_gb, prep.DEFAULT_MIN_FREE_GB)
    max_cache_bytes = prep._parse_gb(max_cache_gb, prep.DEFAULT_MAX_CACHE_GB)

    mirror = None if args.no_mirror else prep.MIRROR_ENDPOINT
    if mirror:
        os.environ["HF_ENDPOINT"] = mirror
        print(f"Mirror: {mirror}")
    else:
        print("Mirror: disabled (--no-mirror)")

    download_api = HfApi(token=args.hf_token, endpoint=mirror or "https://huggingface.co")

    print(f"\n{'=' * 60}")
    print(f"  RAW DOWNLOAD")
    print(f"  Recipe:     {os.path.abspath(args.datasets_config) if args.datasets_config else 'configs/production/datasets.yaml'}")
    print(f"  Output:     {os.path.abspath(download_dir)}")
    print(f"  Datasets:   {len(specs)}")
    print(f"  Scale:      {args.scale}")
    print(f"  Disk floor: {prep._fmt_bytes(min_free_bytes)} free")
    print(f"  Cache cap:  {prep._fmt_bytes(max_cache_bytes)}")
    print(f"{'=' * 60}")

    for spec in specs:
        if not spec.get("hf_path"):
            print(f"  {spec['name']}: disabled/no source, skipping.")
            continue
        process_spec(
            spec,
            download_api=download_api,
            download_dir=download_dir,
            hf_token=args.hf_token,
            min_free_bytes=min_free_bytes,
            max_cache_bytes=max_cache_bytes,
            raw_shard_est_tokens=args.raw_shard_est_tokens,
            no_resume=args.no_resume,
        )


if __name__ == "__main__":
    main()
