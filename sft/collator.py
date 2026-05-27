"""ChatML SFT collator with assistant-only loss masking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


@dataclass
class DataCollatorForChatSFT:
    tokenizer: Any
    max_seq_length: int
    train_on_assistant_end: bool = True

    def __post_init__(self):
        if self.max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            eos_id = getattr(self.tokenizer, "eos_token_id", None)
            if eos_id is None:
                raise ValueError("Tokenizer must define pad_token_id or eos_token_id")
            self.tokenizer.pad_token_id = eos_id

    def _encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def _render(self, messages: list[dict[str, str]]) -> tuple[list[int], list[int]]:
        input_ids: list[int] = []
        labels: list[int] = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            header_ids = self._encode(f"{IM_START}{role}\n")
            content_ids = self._encode(content)
            end_ids = self._encode(IM_END)
            newline_ids = self._encode("\n")

            input_ids.extend(header_ids)
            input_ids.extend(content_ids)
            input_ids.extend(end_ids)
            input_ids.extend(newline_ids)

            labels.extend([-100] * len(header_ids))
            if role == "assistant":
                labels.extend(content_ids)
                labels.extend(end_ids if self.train_on_assistant_end else [-100] * len(end_ids))
            else:
                labels.extend([-100] * len(content_ids))
                labels.extend([-100] * len(end_ids))
            labels.extend([-100] * len(newline_ids))

        return input_ids[: self.max_seq_length], labels[: self.max_seq_length]

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        rendered = [self._render(feature["messages"]) for feature in features]
        max_len = max(len(input_ids) for input_ids, _ in rendered)
        max_len = min(max_len, self.max_seq_length)
        pad_id = int(self.tokenizer.pad_token_id)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for input_ids, labels in rendered:
            input_ids = input_ids[:max_len]
            labels = labels[:max_len]
            pad_len = max_len - len(input_ids)
            batch_input_ids.append(input_ids + [pad_id] * pad_len)
            batch_attention_mask.append([1] * len(input_ids) + [0] * pad_len)
            batch_labels.append(labels + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }

