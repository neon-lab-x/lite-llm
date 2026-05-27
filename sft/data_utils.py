"""SFT JSONL data loading and validation utilities.

The SFT flow keeps data in a normalized messages format:

    {"messages": [{"role": "user", "content": "..."}, ...]}

This module deliberately avoids depending on HuggingFace Datasets so the local
SFT smoke path can run with the base project dependencies.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from torch.utils.data import Dataset


ALLOWED_ROLES = {"system", "user", "assistant"}


@dataclass(frozen=True)
class SftRecord:
    messages: list[dict[str, str]]
    source: str = "unknown"


class SftJsonlDataset(Dataset):
    def __init__(self, records: Sequence[SftRecord]):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        return {"messages": record.messages, "source": record.source}


def iter_jsonl_paths(path: str) -> list[str]:
    if os.path.isdir(path):
        files = [
            os.path.join(path, filename)
            for filename in sorted(os.listdir(path))
            if filename.endswith((".jsonl", ".json"))
        ]
        if not files:
            raise FileNotFoundError(f"No .jsonl/.json SFT files found in {path}")
        return files

    if not os.path.exists(path):
        raise FileNotFoundError(f"SFT data path not found: {path}")
    return [path]


def _coerce_messages(value) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError("messages must be a list")

    messages: list[dict[str, str]] = []
    for msg in value:
        if not isinstance(msg, dict):
            raise ValueError("each message must be a dict")
        role = msg.get("role")
        content = msg.get("content")
        if role not in ALLOWED_ROLES:
            raise ValueError(f"invalid role {role!r}; expected one of {sorted(ALLOWED_ROLES)}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("message content must be a non-empty string")
        messages.append({"role": role, "content": content.strip()})
    if not messages:
        raise ValueError("messages must not be empty")
    if not any(msg["role"] == "assistant" for msg in messages):
        raise ValueError("conversation must contain at least one assistant message")
    return messages


def load_sft_records(data_path: str) -> list[SftRecord]:
    records: list[SftRecord] = []
    for path in iter_jsonl_paths(data_path):
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    messages = _coerce_messages(obj.get("messages"))
                except Exception as exc:
                    raise ValueError(f"{path}:{line_no}: invalid SFT row: {exc}") from exc
                records.append(SftRecord(messages=messages, source=obj.get("source", os.path.basename(path))))
    if not records:
        raise ValueError(f"No SFT records loaded from {data_path}")
    return records


def validate_messages_format(records: Iterable[SftRecord | dict]) -> None:
    for idx, record in enumerate(records):
        messages = record.messages if isinstance(record, SftRecord) else record.get("messages")
        try:
            _coerce_messages(messages)
        except Exception as exc:
            raise ValueError(f"Row {idx}: {exc}") from exc


def split_sft_records(
    records: Sequence[SftRecord],
    val_fraction: float = 0.05,
    seed: int = 42,
) -> tuple[list[SftRecord], Optional[list[SftRecord]]]:
    if val_fraction <= 0:
        return list(records), None
    if val_fraction >= 1.0:
        raise ValueError("val_fraction must be in [0, 1)")

    indices = list(range(len(records)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_count = max(1, int(len(indices) * val_fraction))
    val_indices = set(indices[:val_count])
    train = [record for i, record in enumerate(records) if i not in val_indices]
    val = [record for i, record in enumerate(records) if i in val_indices]
    if not train:
        raise ValueError("SFT split produced an empty training set")
    return train, val


def write_jsonl(records: Sequence[dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

