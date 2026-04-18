# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Lite-LLM is a from-scratch pre-training project for a causal language model (~1.5B parameters). The architecture is a Llama-style decoder-only transformer with Multi-head Latent Attention (MLA), SwiGLU FFN, RoPE, RMSNorm, and QK-Norm. It integrates with HuggingFace Transformers (`PreTrainedModel`/`PretrainedConfig`/`GenerationMixin`) for compatibility with the HF Trainer ecosystem, and supports KV cache for fast `generate()`.

## Flows

The repo has two isolated, non-crossing flows:

- **Local (CPU smoke test)**: `scripts/local/prepare_data.py`, `scripts/local/train.py`, `configs/local/{model,train}.yaml`, data under `./data/local_smoke/`, artifacts under `./artifacts/local/`.
- **Production (GPU + DeepSpeed)**: `scripts/production/prepare_data.py`, `scripts/production/train.py`, `configs/production/{model,train,deepspeed_zero2}.{yaml,json}`, data under `./data/production/`, artifacts under `./artifacts/production/`.

`lite_llm/flow_validation.py` enforces that each flow's config cannot accidentally reach into the other flow's paths, CPU/GPU setting, tokenizer, or DeepSpeed.

## Commands

```bash
# Local smoke test (CPU, ~0.2s)
uv run python scripts/local/prepare_data.py
uv run python scripts/local/train.py

# Production data (server, network required)
uv sync --extra production --frozen
uv run python scripts/production/prepare_data.py --dry-run
uv run python scripts/production/prepare_data.py --scale 0.01

# Production training (single node multi-GPU via DeepSpeed)
uv run deepspeed scripts/production/train.py
# or without DeepSpeed (remove the 'deepspeed' key in train.yaml):
uv run python scripts/production/train.py

# Periodic checkpoint upload to HF (run in a separate terminal alongside training)
export HF_TOKEN=hf_xxxxxxxxxxxx
python scripts/production/upload_checkpoint_to_hf.py \
    --repo username/lite-llm-checkpoints \
    --interval 3600   # default 1 hour; uploads latest complete checkpoint each cycle

# One-shot checkpoint upload (for manual or cron use)
python scripts/production/upload_checkpoint_to_hf.py \
    --repo username/lite-llm-checkpoints --once

# Upload tokenized data shards to HF Dataset (one-time, separate from checkpoints)
python scripts/production/upload_to_hf.py --token $HF_TOKEN --repo username/lite-llm
```

## Configuration

All config is YAML/JSON-driven — no hardcoded hyperparameters in training code.

- `configs/production/model.yaml` — model architecture (hidden, layers, MLA heads, head_dim, q/kv latent ranks, rope_theta, initializer_range, etc.)
- `configs/production/train.yaml` — training hyperparameters, data paths, `val_fraction` for eval split, DeepSpeed reference
- `configs/production/deepspeed_zero2.json` — DeepSpeed ZeRO-2 config (activation checkpointing is *not* here; HF gradient_checkpointing is used instead)

The tokenizer is `Qwen/Qwen3.5-0.8B` (`len(tokenizer) = 248077`; model `vocab_size = 248320`, padded up to a multiple of 128 for tensor-core friendly matmuls).

## Architecture

**Model core** (`lite_llm/`):

- `configuration.py` — `LiteLlmConfig(PretrainedConfig)` with MLA fields (`num_attention_heads`, `head_dim`, `q_lora_rank`, `kv_lora_rank`, `qk_norm`) and `initializer_range`.
- `modeling.py` — `RMSNorm` → `RotaryEmbedding` → `MLAAttention` (with latent Q/KV projections, optional QK-Norm, and KV cache via `transformers.Cache`) → `SwiGLUFFN` → `TransformerBlock` → `LiteLlmModel` → `LiteLlmForCausalLM`. Weight tying is declared via `_tied_weights_keys`. `_init_weights` uses `N(0, initializer_range)`, and residual projections (`o_proj`, `down_proj`) are further downscaled by `1/sqrt(2 * num_hidden_layers)` (GPT-2 style).
- `data_utils.py` — `PretrainDataset` (packed fixed-length sequences from `.npy` token arrays), `DataCollatorForPretraining`, `load_tokenized_dataset`, and `split_train_val` (holds out the tail of the stream for eval perplexity).
- `train_runner.py` — shared training entrypoint used by both flows. Contains `RichLoggingCallback` (prints tok/s, ETA, progress, elapsed, GPU memory on every log step) and wires it into `Trainer` automatically. `tokens_per_step` is derived from `TrainingArguments` so no extra config is needed.
- `flow_validation.py` — makes sure local and production configs stay isolated.
- `token_storage.py`, `local_smoke.py` — shard utilities and local tiny dataset generator.

**Attention**: MLA with low-rank latent projections. Q uses `q_down_proj -> q_up_proj`, while K/V share `kv_down_proj` before separate `k_up_proj` / `v_up_proj`. RoPE is applied after optional QK-Norm. `past_key_values` uses the `transformers.cache_utils.Cache` API.

## Production Scripts

Each production script has a single, independent responsibility. Do NOT merge their roles:

| Script | Purpose |
|---|---|
| `scripts/production/prepare_data.py` | Stream, filter, tokenize, and shard data to `.npy` files |
| `scripts/production/train.py` | Launch training via `train_runner.run_training` |
| `scripts/production/upload_to_hf.py` | **One-time** upload of tokenized `.npy` shards to a HF **Dataset** repo |
| `scripts/production/upload_checkpoint_to_hf.py` | **Periodic** upload of training checkpoints to a HF **Model** repo (run alongside training) |

## Dependencies

- Base: PyTorch ≥ 2.1, transformers ≥ 4.38 (<6), datasets, pyyaml, accelerate.
- Production extra (`uv sync --extra production`): deepspeed, wandb.

## Documentation Rule

**Always update `README.md` when any of the following change:**
- Training flow, scripts, or their entry points
- Configuration fields in `train.yaml` or `model.yaml`
- New scripts added to `scripts/production/` or `scripts/local/`
- Logging format or monitoring behavior
- Checkpoint / upload strategy

README sections most likely to need updating: §1 (layout), §6 (production training), §6.3–6.5 (launch, logging, HF upload).
