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

    split = dataset.train_test_split(test_size=val_fraction, seed=seed)
    return split["train"], split["test"]


def validate_messages_format(dataset: Dataset) -> None:
    """Validate that dataset has the expected messages structure."""
    if "messages" not in dataset.column_names:
        raise ValueError(
            f"SFT dataset must have a 'messages' column. "
            f"Got columns: {dataset.column_names}"
        )
    sample = dataset[0]
    messages = sample["messages"]
    if not isinstance(messages, list) or len(messages) == 0:
        raise ValueError("messages field must be a non-empty list")
    for msg in messages:
        if "role" not in msg or "content" not in msg:
            raise ValueError(
                f"Each message must have 'role' and 'content' keys. Got: {list(msg.keys())}"
            )
