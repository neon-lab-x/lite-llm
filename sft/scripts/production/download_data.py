#!/usr/bin/env python3
"""Download and normalize SFT datasets into ChatML messages JSONL.

This is the production data-prep entrypoint for the isolated SFT flow. It reads
recipes from ``sft/configs/production/datasets_*.yaml`` and writes one JSONL file
per source dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import yaml

try:
    from datasets import load_dataset
except ModuleNotFoundError as exc:
    raise SystemExit("Run `uv sync --extra production --frozen` first.") from exc

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from sft.data_utils import ALLOWED_ROLES


def load_recipe(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _default_output_dir(cfg: dict) -> str:
    if cfg.get("output_dir"):
        return cfg["output_dir"]
    recipe = cfg.get("recipe_name", "sft")
    return os.path.join("data", "production", "sft", recipe)


def _normalize_messages(messages) -> list[dict[str, str]] | None:
    if not isinstance(messages, list):
        return None
    normalized = []
    for msg in messages:
        if not isinstance(msg, dict):
            return None
        role = msg.get("role")
        content = msg.get("content")
        if role not in ALLOWED_ROLES or not isinstance(content, str) or not content.strip():
            return None
        normalized.append({"role": role, "content": content.strip()})
    if not normalized or not any(msg["role"] == "assistant" for msg in normalized):
        return None
    return normalized


def _first_existing(example: dict, fields: list[str]):
    for field in fields:
        value = example.get(field)
        if value not in (None, ""):
            return value
    return None


def _format_user_text(instruction: str, extra_input: str | None) -> str:
    instruction = instruction.strip()
    extra_input = (extra_input or "").strip()
    if extra_input:
        return f"{instruction}\n\n{extra_input}"
    return instruction


def convert_example(example: dict, spec: dict) -> list[dict[str, str]] | None:
    fmt = spec.get("format", "messages")
    if fmt == "messages":
        return _normalize_messages(example.get(spec.get("messages_field", "messages")))

    if fmt == "alpaca":
        instruction = _first_existing(example, [spec.get("instruction_field", "instruction")])
        extra_input = _first_existing(example, [spec.get("input_field", "input")])
        output = _first_existing(
            example,
            [spec.get("output_field", "output"), "response", "target", "answer"],
        )
        if not instruction or not output:
            return None
        return [
            {"role": "user", "content": _format_user_text(str(instruction), str(extra_input or ""))},
            {"role": "assistant", "content": str(output).strip()},
        ]

    if fmt == "prompt_response":
        prompt = _first_existing(
            example,
            [spec.get("prompt_field", "prompt"), "instruction", "input", "problem", "question"],
        )
        response = _first_existing(
            example,
            [spec.get("response_field", "response"), "output", "target", "solution", "answer"],
        )
        if not prompt or not response:
            return None
        return [
            {"role": "user", "content": str(prompt).strip()},
            {"role": "assistant", "content": str(response).strip()},
        ]

    if fmt == "firefly":
        prompt = example.get(spec.get("input_field", "input"))
        response = example.get(spec.get("target_field", "target"))
        if not prompt or not response:
            return None
        return [
            {"role": "user", "content": str(prompt).strip()},
            {"role": "assistant", "content": str(response).strip()},
        ]

    if fmt == "chosen_messages":
        return _normalize_messages(example.get(spec.get("chosen_field", "chosen")))

    raise ValueError(f"Unsupported SFT dataset format: {fmt}")


def _passes_filters(messages: list[dict[str, str]], spec: dict) -> bool:
    filters = spec.get("filters", {})
    text = "\n".join(msg["content"] for msg in messages)
    min_chars = filters.get("min_chars")
    if min_chars is not None and len(text) < int(min_chars):
        return False
    max_chars = filters.get("max_chars")
    if max_chars is not None and len(text) > int(max_chars):
        return False
    min_assistant_chars = filters.get("min_assistant_chars")
    if min_assistant_chars is not None:
        assistant_chars = sum(len(msg["content"]) for msg in messages if msg["role"] == "assistant")
        if assistant_chars < int(min_assistant_chars):
            return False
    return True


def _existing_count(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _load_hf_dataset(spec: dict, token: str | None, streaming: bool):
    kwargs = {
        "path": spec["source"],
        "split": spec.get("split", "train"),
        "streaming": streaming,
        "token": token,
    }
    config = spec.get("config") or spec.get("subset")
    if config:
        kwargs["name"] = config
    data_files = spec.get("data_files")
    if data_files:
        kwargs["data_files"] = data_files
    return load_dataset(**kwargs)


def process_spec(spec: dict, *, output_dir: str, hf_token: str | None, streaming: bool, no_resume: bool):
    if spec.get("enabled", True) is False:
        print(f"  {spec['name']}: disabled, skipping.")
        return

    name = spec["name"]
    target = int(spec.get("target_samples", 0) or 0)
    output_path = os.path.join(output_dir, f"{name}.jsonl")
    os.makedirs(output_dir, exist_ok=True)

    existing = 0 if no_resume else _existing_count(output_path)
    if target and existing >= target:
        print(f"  {name}: already has {existing:,}/{target:,} samples, skipping.")
        return

    mode = "w" if no_resume or existing == 0 else "a"
    seed = int(spec.get("seed", 42))
    shuffle_buffer = int(spec.get("shuffle_buffer", 10_000))

    print(f"\n{'=' * 60}")
    print(f"  {name}  ({spec['source']})")
    print(f"  Target: {target or 'all'} | Existing: {existing:,} | Streaming: {streaming}")
    if spec.get("license") in {"review", "unknown"}:
        print("  WARNING: recipe marks this dataset license as needing review.")
    print(f"{'=' * 60}")

    ds = _load_hf_dataset(spec, token=hf_token, streaming=streaming)
    if spec.get("shuffle", True):
        if streaming:
            ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
        else:
            ds = ds.shuffle(seed=seed)

    written = existing
    seen_valid = 0
    skipped = 0
    with open(output_path, mode, encoding="utf-8") as f:
        for example in ds:
            if target and written >= target:
                break
            messages = convert_example(example, spec)
            if messages is None or not _passes_filters(messages, spec):
                skipped += 1
                continue
            if existing and seen_valid < existing:
                seen_valid += 1
                continue
            row = {"messages": messages, "source": name}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            if written % 10_000 == 0:
                print(f"  [{name}] {written:,} samples written...")

    print(f"  {name}: wrote {written:,} total samples to {output_path}")
    if skipped:
        print(f"  {name}: skipped {skipped:,} rows during conversion/filtering")


def parse_args():
    parser = argparse.ArgumentParser(description="Download SFT datasets to normalized JSONL")
    parser.add_argument(
        "--datasets-config",
        default="sft/configs/production/datasets_zh_first_3b.yaml",
    )
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset names to process")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--no-streaming", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_recipe(args.datasets_config)
    output_dir = args.output_dir or _default_output_dir(cfg)
    specs = cfg.get("datasets", [])
    if args.datasets:
        wanted = set(args.datasets.split(","))
        specs = [spec for spec in specs if spec["name"] in wanted]

    print(f"SFT recipe: {cfg.get('recipe_name', os.path.basename(args.datasets_config))}")
    print(f"Output: {os.path.abspath(output_dir)}")
    for spec in specs:
        process_spec(
            spec,
            output_dir=output_dir,
            hf_token=args.hf_token,
            streaming=not args.no_streaming,
            no_resume=args.no_resume,
        )


if __name__ == "__main__":
    main()
