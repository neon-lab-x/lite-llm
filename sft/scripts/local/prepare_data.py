#!/usr/bin/env python3
"""Create local SFT smoke data and a tiny pretrained checkpoint."""

from __future__ import annotations

import os
import sys

import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from lite_llm.configuration import LiteLlmConfig
from lite_llm.modeling import LiteLlmForCausalLM
from sft.data_utils import write_jsonl
from sft.toy_tokenizer import ToyChatTokenizer


DATA_PATH = "./data/local_smoke/sft/sft_data.jsonl"
MODEL_CONFIG = "./sft/configs/local/model.yaml"
PRETRAINED_DIR = "./artifacts/local/sft/pretrained"


SAMPLES = [
    {
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Say hello."},
            {"role": "assistant", "content": "Hello."},
        ],
        "source": "local_smoke",
    },
    {
        "messages": [
            {"role": "user", "content": "What is 2 plus 2?"},
            {"role": "assistant", "content": "2 plus 2 equals 4."},
        ],
        "source": "local_smoke",
    },
    {
        "messages": [
            {"role": "user", "content": "Name one primary color."},
            {"role": "assistant", "content": "Red is one primary color."},
        ],
        "source": "local_smoke",
    },
    {
        "messages": [
            {"role": "user", "content": "Return the word done."},
            {"role": "assistant", "content": "done"},
        ],
        "source": "local_smoke",
    },
]


def main():
    write_jsonl(SAMPLES, DATA_PATH)
    print(f"Saved {len(SAMPLES):,} SFT smoke conversations to {DATA_PATH}")

    with open(MODEL_CONFIG, encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)
    config = LiteLlmConfig(**model_cfg)
    model = LiteLlmForCausalLM(config)
    os.makedirs(PRETRAINED_DIR, exist_ok=True)
    model.save_pretrained(PRETRAINED_DIR)
    ToyChatTokenizer(vocab_size=config.vocab_size).save_pretrained(PRETRAINED_DIR)
    print(f"Saved tiny SFT base checkpoint to {PRETRAINED_DIR}")


if __name__ == "__main__":
    main()

