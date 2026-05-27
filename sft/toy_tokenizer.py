"""Offline toy ChatML tokenizer for local SFT smoke tests."""

from __future__ import annotations

import json
import os

from sft.collator import IM_END, IM_START


class ToyChatTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    im_start_id = 2
    im_end_id = 3
    pad_token = "<pad>"
    eos_token = IM_END

    def __init__(self, vocab_size: int = 256):
        if vocab_size < 32:
            raise ValueError("ToyChatTokenizer needs vocab_size >= 32")
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.vocab_size

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        ids: list[int] = []
        i = 0
        while i < len(text):
            if text.startswith(IM_START, i):
                ids.append(self.im_start_id)
                i += len(IM_START)
                continue
            if text.startswith(IM_END, i):
                ids.append(self.im_end_id)
                i += len(IM_END)
                continue
            ids.append(4 + (ord(text[i]) % (self.vocab_size - 4)))
            i += 1
        return ids

    def save_pretrained(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "toy_tokenizer_config.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "toy_chatml", "vocab_size": self.vocab_size}, f, indent=2)

