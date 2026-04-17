"""Pretraining data pipeline.

- ``PretrainDataset`` packs a list of flat token arrays into fixed-length
  training sequences. Internally it concatenates logically (using a prefix-sum
  index), so document boundaries inside a sequence are *not* masked; insert an
  EOS token between documents at tokenization time (see ``scripts/prepare_data.py``).
- ``DataCollatorForPretraining`` stacks ``input_ids`` / ``labels`` from the
  dataset and returns a batch dict.
- ``load_tokenized_dataset`` loads every ``*.npy`` / ``*.bin`` file from a
  directory with ``mmap`` so we don't blow up RAM.
- ``split_train_val`` splits a flat token stream into two ``PretrainDataset``s
  so the Trainer can compute eval loss / perplexity.
"""

import bisect
import glob
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


# ---------------------------------------------------------------------------
# Pre-training Dataset (packed sequences)
# ---------------------------------------------------------------------------

class PretrainDataset(Dataset):
    """Packs fixed-length sequences over one logical token stream."""

    def __init__(
        self,
        token_arrays: List[np.ndarray],
        max_seq_length: int,
        token_offset: int = 0,
        max_tokens: Optional[int] = None,
    ):
        if max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")

        self.max_seq_length = max_seq_length
        self.token_arrays = [arr for arr in token_arrays if len(arr) > 0]
        self.array_lengths = [int(len(arr)) for arr in self.token_arrays]
        self.cumulative_lengths: List[int] = []

        total = 0
        for length in self.array_lengths:
            total += length
            self.cumulative_lengths.append(total)
        self.total_tokens = total

        self.token_offset = int(token_offset)
        self.token_limit = (
            self.total_tokens if max_tokens is None
            else min(self.total_tokens, self.token_offset + int(max_tokens))
        )
        if self.token_offset < 0 or self.token_offset > self.token_limit:
            raise ValueError("token_offset out of range")

        usable = max(self.token_limit - self.token_offset, 0)
        self.n_sequences = usable // self.max_seq_length

    def __len__(self) -> int:
        return self.n_sequences

    def _slice_tokens(self, start: int, length: int) -> np.ndarray:
        remaining = length
        cursor = start
        chunks = []

        while remaining > 0:
            array_idx = bisect.bisect_right(self.cumulative_lengths, cursor)
            array_start = 0 if array_idx == 0 else self.cumulative_lengths[array_idx - 1]
            local_start = cursor - array_start
            array = self.token_arrays[array_idx]
            take = min(remaining, len(array) - local_start)
            chunks.append(np.asarray(array[local_start: local_start + take]))
            cursor += take
            remaining -= take

        if len(chunks) == 1:
            return chunks[0]
        return np.concatenate(chunks, axis=0)

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.n_sequences:
            raise IndexError(idx)

        start = self.token_offset + idx * self.max_seq_length
        # Single int32->int64 copy via torch.tensor; avoids the prior
        # mmap->np.array(copy=True)->torch.from_numpy->.to(long) double copy.
        tokens = torch.tensor(
            self._slice_tokens(start, self.max_seq_length), dtype=torch.long
        )
        return {"input_ids": tokens}


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------

@dataclass
class DataCollatorForPretraining:
    """Stacks ``input_ids`` and clones them as labels for next-token prediction."""

    def __call__(self, features):
        input_ids = torch.stack([f["input_ids"] for f in features])
        labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _load_token_arrays(data_dir: str) -> List[np.ndarray]:
    file_paths = sorted(
        glob.glob(os.path.join(data_dir, "*.npy"))
        + glob.glob(os.path.join(data_dir, "*.bin"))
    )
    arrays: List[np.ndarray] = []
    for path in file_paths:
        if path.endswith(".npy"):
            arrays.append(np.load(path, mmap_mode="r", allow_pickle=False))
        else:
            arrays.append(np.memmap(path, dtype=np.int32, mode="r"))
    if not arrays:
        raise FileNotFoundError(f"No tokenized data found in {data_dir}")
    return arrays


def load_tokenized_dataset(
    data_dir: str, max_seq_length: int
) -> PretrainDataset:
    """Load every ``*.npy`` / ``*.bin`` file in ``data_dir`` into a dataset."""
    return PretrainDataset(_load_token_arrays(data_dir), max_seq_length)


def split_train_val(
    data_dir: str,
    max_seq_length: int,
    val_fraction: float = 0.005,
    max_val_tokens: Optional[int] = 50_000_000,
) -> Tuple[PretrainDataset, Optional[PretrainDataset]]:
    """Split tokens into a train + val dataset.

    Validation is taken from the *tail* of the token stream (deterministic,
    reproducible, no shuffling required). Returns ``(train_ds, val_ds)``; the
    val dataset is ``None`` when there are not enough tokens to fill a single
    eval sequence.
    """
    arrays = _load_token_arrays(data_dir)
    total = sum(int(len(a)) for a in arrays)
    if total <= 0:
        raise ValueError(f"No tokens loaded from {data_dir}")

    if not (0.0 <= val_fraction < 1.0):
        raise ValueError("val_fraction must be in [0, 1)")

    val_tokens = int(total * val_fraction)
    if max_val_tokens is not None:
        val_tokens = min(val_tokens, int(max_val_tokens))
    # Align val tokens down to a whole number of eval sequences.
    val_tokens = (val_tokens // max_seq_length) * max_seq_length

    train_tokens = total - val_tokens
    train_tokens = (train_tokens // max_seq_length) * max_seq_length

    train_ds = PretrainDataset(arrays, max_seq_length, token_offset=0, max_tokens=train_tokens)
    if val_tokens == 0:
        return train_ds, None

    val_ds = PretrainDataset(
        arrays,
        max_seq_length,
        token_offset=train_tokens,
        max_tokens=val_tokens,
    )
    return train_ds, val_ds


def tokenize_and_save(
    texts: List[str],
    tokenizer: PreTrainedTokenizerBase,
    output_path: str,
    add_eos: bool = True,
):
    """Tokenize a list of texts and save the flat int32 token array.

    When ``add_eos`` is True (default), a ``tokenizer.eos_token_id`` is appended
    after each document so downstream packing can learn document boundaries.
    """
    eos_id = tokenizer.eos_token_id if add_eos else None
    all_tokens: List[int] = []
    for text in texts:
        all_tokens.extend(tokenizer.encode(text, add_special_tokens=False))
        if eos_id is not None:
            all_tokens.append(eos_id)
    arr = np.array(all_tokens, dtype=np.int32)
    np.save(output_path, arr)
    print(f"Saved {len(arr):,} tokens to {output_path}")
    return arr
