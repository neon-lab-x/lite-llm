"""SFT data pipeline.

Loads OpenAI ChatML-format JSONL (one conversation per line, ``messages`` field)
into HuggingFace Datasets suitable for TRL SFTTrainer.

Expected JSONL format::

    {"messages": [{"role": "system", "content": "..."}, ...]}
"""

import os
from typing import Optional, Tuple

from datasets import Dataset, load_dataset


def load_sft_dataset(data_path: str, split: str = "train") -> Dataset:
    """Load a JSONL file with messages-format conversations.

    Accepts a single file path or a directory (globs all ``.jsonl`` / ``.json``).
    """
    if os.path.isdir(data_path):
        files = sorted(
            f
            for f in os.listdir(data_path)
            if f.endswith(".jsonl") or f.endswith(".json")
        )
        if not files:
            raise FileNotFoundError(f"No .jsonl files found in {data_path}")
        data_files = [os.path.join(data_path, f) for f in files]
    else:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"SFT data file not found: {data_path}")
        data_files = data_path

    return load_dataset("json", data_files=data_files, split=split)


def split_sft_train_val(
    dataset: Dataset,
    val_fraction: float = 0.05,
    seed: int = 42,
) -> Tuple[Dataset, Optional[Dataset]]:
    """Split a dataset into train/val by conversation (random split)."""
    if val_fraction <= 0:
        return dataset, None
    if val_fraction >= 1.0:
        raise ValueError(
            f"val_fraction must be in (0, 1), got {val_fraction}. "
            "This would produce an empty training set."
        )

    split = dataset.train_test_split(test_size=val_fraction, seed=seed)
    return split["train"], split["test"]


def validate_messages_format(dataset: Dataset, num_samples: Optional[int] = None) -> None:
    """Validate that dataset has the expected messages structure.

    By default checks the full dataset, so corrupt entries in the middle/tail
    are caught before training starts.
    """
    if "messages" not in dataset.column_names:
        raise ValueError(
            f"SFT dataset must have a 'messages' column. "
            f"Got columns: {dataset.column_names}"
        )
    if num_samples is None or num_samples <= 0:
        check_n = len(dataset)
    else:
        check_n = min(num_samples, len(dataset))
    allowed_roles = {"system", "user", "assistant"}
    for idx in range(check_n):
        sample = dataset[idx]
        messages = sample["messages"]
        if not isinstance(messages, list) or len(messages) == 0:
            raise ValueError(
                f"Row {idx}: messages field must be a non-empty list, got {type(messages)}"
            )
        for msg in messages:
            if not isinstance(msg, dict):
                raise ValueError(
                    f"Row {idx}: each message must be a dict with 'role'/'content', "
                    f"got {type(msg)}"
                )
            if "role" not in msg or "content" not in msg:
                raise ValueError(
                    f"Row {idx}: each message must have 'role' and 'content' keys. "
                    f"Got: {list(msg.keys())}"
                )
            role = msg["role"]
            content = msg["content"]
            if role not in allowed_roles:
                raise ValueError(
                    f"Row {idx}: invalid role '{role}'. "
                    f"Allowed roles: {sorted(allowed_roles)}"
                )
            if not isinstance(content, str) or not content.strip():
                raise ValueError(
                    f"Row {idx}: content must be a non-empty string, "
                    f"got {type(content)} with value={content!r}"
                )
