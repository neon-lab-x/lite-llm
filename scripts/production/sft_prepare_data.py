#!/usr/bin/env python3
"""Download HF SFT datasets → convert to ChatML messages format → save as JSONL.

Reads dataset specs from configs/production/sft_datasets.yaml.
Pipeline: HF dataset → format conversion → JSONL to ./data/production/sft/.
"""

import json
import os
import sys

import yaml

try:
    from datasets import load_dataset
except ModuleNotFoundError as exc:
    raise SystemExit("Run `uv sync --extra production --frozen` first.") from exc

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

OUTPUT_DIR = "./data/production/sft"


# ---------------------------------------------------------------------------
# Format converters: normalize various HF dataset formats to messages format
# ---------------------------------------------------------------------------

def _convert_oasst(example):
    """Convert OpenAssistant Guanaco format to messages.

    Guanaco has 'text' field with Human:/Assistant: prefixes.
    """
    text = example.get("text", "")
    messages = []
    parts = text.split("### ")
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.startswith("Human:"):
            messages.append({"role": "user", "content": part[len("Human:"):].strip()})
        elif part.startswith("Assistant:"):
            messages.append({"role": "assistant", "content": part[len("Assistant:"):].strip()})
    return {"messages": messages} if messages else None


_SHAREGPT_ROLE_MAP = {"human": "user", "gpt": "assistant", "system": "system"}


def _convert_sharegpt(example):
    """Convert ShareGPT format to ChatML messages.

    ShareGPT uses a ``conversations`` list where each turn has
    ``from`` (human/gpt/system) and ``value`` keys.
    """
    turns = example.get("conversations") or []
    messages = []
    for turn in turns:
        src = turn.get("from", "")
        role = _SHAREGPT_ROLE_MAP.get(src)
        content = turn.get("value", "").strip()
        if role is None or not content:
            continue
        messages.append({"role": role, "content": content})
    return {"messages": messages} if messages else None


def _convert_firefly(example):
    """Convert Firefly (YeungNLP) Chinese instruction format to messages.

    Firefly uses flat ``input`` / ``target`` fields (single-turn).
    """
    user_content = (example.get("input") or "").strip()
    assistant_content = (example.get("target") or "").strip()
    if not user_content or not assistant_content:
        return None
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


FORMAT_CONVERTERS = {
    "messages": None,    # already in ChatML format, no conversion needed
    "oasst": _convert_oasst,
    "sharegpt": _convert_sharegpt,
    "firefly": _convert_firefly,
}


def load_datasets_config(path=None):
    if path is None:
        path = os.path.join(PROJECT_ROOT, "configs", "production", "sft_datasets.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def process_spec(spec, output_dir):
    name = spec["name"]
    hf_path = spec["hf_path"]
    split = spec["split"]
    fmt = spec.get("format", "messages")
    target = spec.get("target_samples")

    output_path = os.path.join(output_dir, f"{name}.jsonl")
    if os.path.exists(output_path):
        with open(output_path) as f:
            existing = sum(1 for _ in f)
        if target and existing >= target:
            print(f"  {name}: already exists ({existing:,} samples), skipping.")
            return

    print(f"\n  {name} ({hf_path})")

    kwargs = {"path": hf_path, "split": split}
    if spec.get("config"):
        kwargs["name"] = spec["config"]
    ds = load_dataset(**kwargs)

    converter = FORMAT_CONVERTERS.get(fmt)
    os.makedirs(output_dir, exist_ok=True)

    count = 0
    skipped = 0
    with open(output_path, "w") as f:
        for example in ds:
            if converter:
                result = converter(example)
                if result is None:
                    skipped += 1
                    continue
                messages = result["messages"]
            else:
                messages = example.get("messages", [])

            if not messages:
                skipped += 1
                continue

            # Validate structure
            valid = all("role" in m and "content" in m for m in messages)
            if not valid:
                skipped += 1
                continue

            f.write(json.dumps({"messages": messages}) + "\n")
            count += 1

            if target and count >= target:
                break

    if skipped > 0:
        print(f"  WARNING: {name}: skipped {skipped:,} invalid/empty examples during conversion.")
    if count == 0:
        print(f"  ERROR: {name}: 0 samples written — check that 'format: {fmt}' matches the actual dataset schema!")
    else:
        print(f"  {name}: saved {count:,} conversations to {output_path}")


def main():
    cfg = load_datasets_config()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for spec in cfg["datasets"]:
        process_spec(spec, OUTPUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
