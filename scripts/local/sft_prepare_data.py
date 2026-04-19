#!/usr/bin/env python3
"""Create isolated local SFT smoke-test data and dummy pretrained checkpoint.

Outputs:
  ./data/local_smoke/sft/sft_data.jsonl   — tiny ChatML conversations
  ./artifacts/local/sft_pretrained/        — random-init tiny model checkpoint
"""

import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = "./data/local_smoke/sft"
CKPT_DIR = "./artifacts/local/sft_pretrained"

CONVERSATIONS = [
    {"messages": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there! How can I help you?"},
    ]},
    {"messages": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."},
    ]},
    {"messages": [
        {"role": "user", "content": "Tell me a joke."},
        {"role": "assistant", "content": "Why did the chicken cross the road? To get to the other side!"},
    ]},
    {"messages": [
        {"role": "user", "content": "Good morning"},
        {"role": "assistant", "content": "Good morning! Have a great day!"},
    ]},
    {"messages": [
        {"role": "user", "content": "What color is the sky?"},
        {"role": "assistant", "content": "The sky is blue on a clear day."},
    ]},
    {"messages": [
        {"role": "user", "content": "Thank you"},
        {"role": "assistant", "content": "You are welcome!"},
    ]},
    {"messages": [
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I am doing well, thank you for asking!"},
    ]},
    {"messages": [
        {"role": "user", "content": "What is your name?"},
        {"role": "assistant", "content": "I am a language model assistant."},
    ]},
    {"messages": [
        {"role": "user", "content": "Goodbye"},
        {"role": "assistant", "content": "Goodbye! See you next time!"},
    ]},
    {"messages": [
        {"role": "user", "content": "What is 3*3?"},
        {"role": "assistant", "content": "3 times 3 equals 9."},
    ]},
    {"messages": [
        {"role": "user", "content": "Tell me about cats."},
        {"role": "assistant", "content": "Cats are small furry animals that are popular pets."},
    ]},
    {"messages": [
        {"role": "user", "content": "What is water?"},
        {"role": "assistant", "content": "Water is a clear liquid essential for life."},
    ]},
    {"messages": [
        {"role": "user", "content": "How old is the sun?"},
        {"role": "assistant", "content": "The sun is about 4.6 billion years old."},
    ]},
    {"messages": [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a popular programming language."},
    ]},
    {"messages": [
        {"role": "user", "content": "Nice weather today."},
        {"role": "assistant", "content": "Yes, it is a beautiful day!"},
    ]},
    {"messages": [
        {"role": "user", "content": "Who wrote Shakespeare?"},
        {"role": "assistant", "content": "William Shakespeare wrote his own plays."},
    ]},
]


def main():
    # --- Write JSONL data ---
    os.makedirs(DATA_DIR, exist_ok=True)
    jsonl_path = os.path.join(DATA_DIR, "sft_data.jsonl")
    with open(jsonl_path, "w") as f:
        for conv in CONVERSATIONS:
            f.write(json.dumps(conv) + "\n")
    print(f"Saved {len(CONVERSATIONS)} conversations to {jsonl_path}")

    # --- Write dummy pretrained checkpoint ---
    os.makedirs(CKPT_DIR, exist_ok=True)

    from lite_llm.configuration import LiteLlmConfig
    from lite_llm.modeling import LiteLlmForCausalLM

    model_cfg = {
        "model_type": "lite_llm",
        "vocab_size": 248320,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "attention_type": "mla",
        "num_attention_heads": 4,
        "head_dim": 16,
        "q_lora_rank": 16,
        "kv_lora_rank": 16,
        "qk_norm": True,
        "max_position_embeddings": 256,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6,
        "initializer_range": 0.02,
        "tie_word_embeddings": True,
    }
    config = LiteLlmConfig(**model_cfg)
    model = LiteLlmForCausalLM(config)
    model.save_pretrained(CKPT_DIR)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Saved dummy checkpoint ({total_params:,} params) to {CKPT_DIR}")


if __name__ == "__main__":
    main()
