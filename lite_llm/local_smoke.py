import os

import numpy as np


VOCAB_SIZE = 256
TOKENS_PER_SHARD = 4096
DEFAULT_OUTPUT_DIR = "./data/local_smoke/tokenized"
DEFAULT_OUTPUT_PATH = os.path.join(DEFAULT_OUTPUT_DIR, "smoke_tokens.npy")


def build_tokens():
    base_pattern = []
    for offset in range(1, 33):
        base_pattern.extend(
            [
                offset,
                (offset * 3) % VOCAB_SIZE,
                (offset * 5) % VOCAB_SIZE,
                (offset * 7) % VOCAB_SIZE,
            ]
        )

    repeats = TOKENS_PER_SHARD // len(base_pattern)
    remainder = TOKENS_PER_SHARD % len(base_pattern)
    tokens = (base_pattern * repeats) + base_pattern[:remainder]
    return np.array(tokens, dtype=np.int32)


def write_smoke_tokens(output_path: str = DEFAULT_OUTPUT_PATH):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tokens = build_tokens()
    np.save(output_path, tokens)
    return tokens
