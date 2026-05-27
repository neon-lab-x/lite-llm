#!/usr/bin/env python3
"""Tokenize locally downloaded raw parquet documents into training shards.

Stage 2 of the production data pipeline:

    local raw parquet documents -> int32 token .npy shards

Run `scripts/production/download_data.py` first. This script does not download
upstream corpora; it only reads local parquet files and uses CPU tokenization.
"""

import argparse
import array as _array
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import prepare_data as prep
from lite_llm.token_storage import existing_token_files, next_shard_index


# --- worker globals (set by _init_tokenizer in each child process) ---
_w_tokenizer = None
_w_eos_id = None


def _init_tokenizer(tokenizer_name, hf_token=None):
    global _w_tokenizer, _w_eos_id
    _w_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, trust_remote_code=True, token=hf_token,
    )
    _w_eos_id = _w_tokenizer.eos_token_id


def _tokenize_one_file(path):
    """Tokenize a single parquet file. Returns (path, np.int32 array, doc_count)."""
    buf = _array.array("i")
    docs = 0
    for text in _iter_texts(path):
        ids = _w_tokenizer.encode(text, add_special_tokens=False)
        buf.extend(ids)
        if _w_eos_id is not None:
            buf.append(_w_eos_id)
        docs += 1
    return path, np.array(buf, dtype=np.int32), docs


# --- config helpers ---

def _default_raw_dir(cfg):
    if cfg.get("raw_output_dir"):
        return cfg["raw_output_dir"]
    recipe = cfg.get("recipe_name", "recipe")
    return os.path.join(PROJECT_ROOT, "data", "production", f"raw_{recipe}")


def _default_output_dir(cfg):
    return cfg.get("output_dir") or os.path.join(PROJECT_ROOT, "data", "production", "tokenized")


def _state_dir(output_dir):
    return os.path.join(output_dir, "_cache", "state")


def _manifest_path(output_dir, name):
    return os.path.join(_state_dir(output_dir), f"{name}_tokenize_manifest.json")


def _load_manifest(output_dir, name):
    path = _manifest_path(output_dir, name)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"dataset": name, "processed_raw_files": [], "tokens": 0, "docs": 0}


def _save_manifest(output_dir, name, manifest):
    os.makedirs(_state_dir(output_dir), exist_ok=True)
    path = _manifest_path(output_dir, name)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _raw_files(raw_dir, name):
    dataset_dir = os.path.join(raw_dir, name)
    if not os.path.isdir(dataset_dir):
        return []
    return sorted(
        os.path.join(dataset_dir, filename)
        for filename in os.listdir(dataset_dir)
        if filename.endswith(".parquet")
    )


def _flush_tokens(tokens, output_dir, name, shard_idx, uploader=None):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}-{shard_idx:05d}.npy")
    np.save(path, np.array(tokens, dtype=np.int32))
    if uploader is not None:
        uploader.add(path)
    return path


def _iter_texts(path):
    pf = pq.ParquetFile(path)
    for rg in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(rg, columns=["text"])
        col = table.column("text")
        for i in range(table.num_rows):
            text = col[i].as_py()
            if text:
                yield text


# --- dataset processors ---

def _process_sequential(name, files, raw_dir, output_dir, manifest,
                        processed, shard_idx, tokenizer, uploader):
    eos_id = tokenizer.eos_token_id
    buf = _array.array("i")

    for file_idx, path in enumerate(files, 1):
        rel = os.path.relpath(path, raw_dir)
        if rel in processed:
            continue

        docs = 0
        for text in _iter_texts(path):
            ids = tokenizer.encode(text, add_special_tokens=False)
            buf.extend(ids)
            if eos_id is not None:
                buf.append(eos_id)
            docs += 1

            if len(buf) >= prep.FLUSH_EVERY:
                out_path = _flush_tokens(buf, output_dir, name, shard_idx, uploader=uploader)
                print(f"  [{name}] wrote {out_path}")
                manifest["tokens"] = int(manifest.get("tokens", 0)) + len(buf)
                manifest["docs"] = int(manifest.get("docs", 0)) + docs
                docs = 0
                buf = _array.array("i")
                shard_idx += 1

        processed.add(rel)
        manifest["processed_raw_files"] = sorted(processed)
        manifest["docs"] = int(manifest.get("docs", 0)) + docs
        _save_manifest(output_dir, name, manifest)
        print(f"  [{name}] processed raw ({file_idx}/{len(files)}) {rel}")

    if buf:
        out_path = _flush_tokens(buf, output_dir, name, shard_idx, uploader=uploader)
        print(f"  [{name}] wrote final {out_path}")
        manifest["tokens"] = int(manifest.get("tokens", 0)) + len(buf)
        _save_manifest(output_dir, name, manifest)

    _print_final(name, output_dir)


def _process_parallel(name, files, raw_dir, output_dir, manifest,
                      processed, shard_idx, uploader, workers, tokenizer_name, hf_token=None):
    pending = []
    for path in files:
        rel = os.path.relpath(path, raw_dir)
        if rel not in processed:
            pending.append((path, rel))

    if not pending:
        print(f"  [{name}] all files already processed.")
        _print_final(name, output_dir)
        return

    print(f"  [{name}] tokenizing {len(pending)} files with {workers} workers...")

    total_tokens = int(manifest.get("tokens", 0))
    total_docs = int(manifest.get("docs", 0))
    done_count = len(files) - len(pending)
    t0 = time.time()

    paths = [p for p, _ in pending]
    rels = [r for _, r in pending]

    with mp.Pool(workers, initializer=_init_tokenizer,
                 initargs=(tokenizer_name, hf_token)) as pool:
        for i, (_path, tokens, docs) in enumerate(pool.imap(_tokenize_one_file, paths)):
            rel = rels[i]
            done_count += 1

            offset = 0
            while offset < len(tokens):
                chunk = tokens[offset:offset + prep.FLUSH_EVERY]
                out_path = _flush_tokens(chunk, output_dir, name, shard_idx, uploader=uploader)
                total_tokens += len(chunk)
                shard_idx += 1
                offset += len(chunk)

            total_docs += docs
            processed.add(rel)
            manifest["processed_raw_files"] = sorted(processed)
            manifest["tokens"] = total_tokens
            manifest["docs"] = total_docs
            _save_manifest(output_dir, name, manifest)

            elapsed = time.time() - t0
            rate = total_tokens / elapsed if elapsed > 0 else 0
            print(
                f"  [{name}] ({done_count}/{len(files)}) {os.path.basename(rel)} "
                f"| {docs:,} docs | {total_tokens:,} tok | {rate:,.0f} tok/s"
            )

    _print_final(name, output_dir)


def _print_final(name, output_dir):
    local_tokens = sum(
        len(np.load(p, mmap_mode="r", allow_pickle=False))
        for p in existing_token_files(name, output_dir)
    )
    print(f"  [{name}] done: {local_tokens:,} local tokens")


def process_dataset(spec, *, raw_dir, output_dir, tokenizer, uploader, no_resume,
                    workers=1, tokenizer_name=None, hf_token=None):
    name = spec["name"]
    files = _raw_files(raw_dir, name)
    if not files:
        print(f"  [{name}] no raw parquet files under {raw_dir}, skipping.")
        return

    manifest = (
        {"dataset": name, "processed_raw_files": [], "tokens": 0, "docs": 0}
        if no_resume else _load_manifest(output_dir, name)
    )
    processed = set(manifest.get("processed_raw_files", []))
    shard_idx = 0 if no_resume else next_shard_index(name, output_dir)

    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"  Raw files: {len(files)} | shard: {shard_idx} | workers: {workers}")
    print(f"{'=' * 60}")

    if workers > 1:
        _process_parallel(
            name, files, raw_dir, output_dir, manifest, processed,
            shard_idx, uploader, workers, tokenizer_name, hf_token,
        )
    else:
        _process_sequential(
            name, files, raw_dir, output_dir, manifest, processed,
            shard_idx, tokenizer, uploader,
        )


# --- CLI ---

def parse_args():
    parser = argparse.ArgumentParser(description="Tokenize local raw parquet docs")
    parser.add_argument("--datasets-config", default=None)
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--tokenizer-name", default=prep.TOKENIZER_NAME)
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel tokenization workers (default: 1)")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--hf-repo", default=prep.HF_TARGET_REPO)
    parser.add_argument("--hf-path", default=prep.HF_TARGET_PATH)
    parser.add_argument("--hf-private", action="store_true")
    parser.add_argument("--keep-uploaded", action="store_true")
    parser.add_argument("--batch-size", type=int, default=prep.COMMIT_BATCH_SIZE)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = prep.load_datasets_config(args.datasets_config)
    specs = prep.build_specs(cfg)
    if args.datasets:
        names = set(args.datasets.split(","))
        specs = [s for s in specs if s["name"] in names]

    raw_dir = args.raw_dir or _default_raw_dir(cfg)
    output_dir = args.output_dir or _default_output_dir(cfg)
    os.makedirs(output_dir, exist_ok=True)

    need_upload = not args.local_only
    if need_upload and not args.hf_token:
        raise SystemExit("Pass --hf-token for upload mode, or use --local-only.")

    uploader = None
    if need_upload:
        api = HfApi(token=args.hf_token, endpoint="https://huggingface.co")
        try:
            api.repo_info(repo_id=args.hf_repo, repo_type="dataset")
        except Exception:
            api.create_repo(repo_id=args.hf_repo, repo_type="dataset", private=args.hf_private)
        uploader = prep.BatchUploader(
            api,
            args.hf_repo,
            args.hf_path,
            args.batch_size,
            keep_uploaded=args.keep_uploaded,
        )

    print(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)

    print(f"\n{'=' * 60}")
    print("  TOKENIZE RAW")
    print(f"  Raw:      {os.path.abspath(raw_dir)}")
    print(f"  Output:   {os.path.abspath(output_dir)}")
    print(f"  Datasets: {len(specs)}")
    print(f"  Workers:  {args.workers}")
    if uploader is not None:
        print(f"  HF:       {args.hf_repo}/{args.hf_path.strip('/')}")
    print(f"{'=' * 60}")

    for spec in specs:
        process_dataset(
            spec,
            raw_dir=raw_dir,
            output_dir=output_dir,
            tokenizer=tokenizer,
            uploader=uploader,
            no_resume=args.no_resume,
            workers=args.workers,
            tokenizer_name=args.tokenizer_name,
            hf_token=args.hf_token,
        )

    if uploader is not None:
        uploader.finish()


if __name__ == "__main__":
    main()
