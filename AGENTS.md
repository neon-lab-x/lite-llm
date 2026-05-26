# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

Lite-LLM is a from-scratch pre-training project for a causal language model (~1.5B parameters). The architecture is a Llama-style decoder-only transformer with Multi-head Latent Attention (MLA), SwiGLU FFN, RoPE, RMSNorm, and QK-Norm. It integrates with HuggingFace Transformers (`PreTrainedModel`/`PretrainedConfig`/`GenerationMixin`) for compatibility with the HF Trainer ecosystem, and supports KV cache for fast `generate()`.

## Flows

The repo has two training modes, each with isolated local and production flows:

**Pretraining** (from random init):
- **Local**: `scripts/local/prepare_data.py`, `scripts/local/train.py`, `configs/local/{model,train}.yaml`, data under `./data/local_smoke/`, artifacts under `./artifacts/local/`.
- **Production**: `scripts/production/prepare_data.py`, `scripts/production/train.py`, `configs/production/{model,train,datasets,deepspeed_zero2}.{yaml,json}`, data under `./data/production/`, artifacts under `./artifacts/production/`.

**SFT** (from pretrained checkpoint):
- **Local**: `scripts/local/sft_prepare_data.py`, `scripts/local/sft_train.py`, `configs/local/{sft_model,sft_train}.yaml`, data under `./data/local_smoke/sft/`, artifacts under `./artifacts/local/sft_*`.
- **Production**: `scripts/production/sft_prepare_data.py`, `scripts/production/sft_train.py`, `configs/production/{sft_train,sft_datasets}.yaml`, data under `./data/production/sft/`, artifacts under `./artifacts/production/sft_*`.

`lite_llm/flow_validation.py` enforces that each flow's config cannot accidentally reach into the other flow's paths, CPU/GPU setting, tokenizer, or DeepSpeed.

**Legacy redirects**: `scripts/prepare_data.py` and `scripts/train.py` at the repo root are deprecated stubs that raise errors pointing to `scripts/local/` or `scripts/production/`.

## Commands

```bash
# Run tests
uv run python -m unittest tests.test_training_fixes -v
uv run python -m unittest tests.test_sft -v

# --- Pretraining ---

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

# Alternative: multi-process data prep (parallelizes tokenization across CPU cores)
uv run python scripts/fast_prepare.py --categories english chinese --scale 0.01 --workers 4

# --- SFT ---

# Install SFT dependency
uv sync --extra sft

# Local SFT smoke test (CPU, self-contained)
uv run python scripts/local/sft_prepare_data.py   # generates JSONL + dummy checkpoint
uv run python scripts/local/sft_train.py

# Production SFT (requires pretrained checkpoint + GPU)
uv sync --extra production --extra sft --frozen
uv run python scripts/production/sft_prepare_data.py
uv run deepspeed scripts/production/sft_train.py
```

## Configuration

All config is YAML/JSON-driven — no hardcoded hyperparameters in training code.

- `configs/production/model.yaml` — model architecture (hidden, layers, MLA heads, head_dim, q/kv latent ranks, rope_theta, initializer_range, etc.)
- `configs/production/train.yaml` — pretraining hyperparameters, data paths, `val_fraction` for eval split, DeepSpeed reference
- `configs/production/datasets.yaml` — dataset specs for `prepare_data.py`: HF dataset names, named filters (e.g. `fineweb_edu: int_score >= 3`), target token counts per dataset
- `configs/production/deepspeed_zero2.json` — DeepSpeed ZeRO-2 config (activation checkpointing is *not* here; HF `gradient_checkpointing` is used instead)
- `configs/production/sft_train.yaml` — SFT training hyperparameters (lower LR=2e-5, 3 epochs, `pretrained_model_path`)
- `configs/production/sft_datasets.yaml` — SFT dataset specs: HF dataset names, format converters, target sample counts

The tokenizer is `Qwen/Qwen3.5-0.8B` (`len(tokenizer) = 248077`; model `vocab_size = 248320`, padded up to a multiple of 128 for tensor-core friendly matmuls).

## Architecture

**Model core** (`lite_llm/`):

- `configuration.py` — `LiteLlmConfig(PretrainedConfig)` with MLA fields (`num_attention_heads`, `head_dim`, `q_lora_rank`, `kv_lora_rank`, `qk_norm`) and `initializer_range`.
- `modeling.py` — `RMSNorm` → `RotaryEmbedding` → `MLAAttention` (with latent Q/KV projections, optional QK-Norm, and KV cache via `transformers.Cache`) → `SwiGLUFFN` → `TransformerBlock` → `LiteLlmModel` → `LiteLlmForCausalLM`. Weight tying is declared via `_tied_weights_keys`. `_init_weights` uses `N(0, initializer_range)`, and residual projections (`o_proj`, `down_proj`) are further downscaled by `1/sqrt(2 * num_hidden_layers)` (GPT-2 style).
- `data_utils.py` — `PretrainDataset` (packed fixed-length sequences from `.npy` token arrays), `DataCollatorForPretraining`, `load_tokenized_dataset`, and `split_train_val` (holds out the tail of the stream for eval perplexity).
- `sft_data_utils.py` — `load_sft_dataset` (loads ChatML JSONL into HF Dataset), `split_sft_train_val` (random split by conversation), `validate_messages_format`.
- `train_runner.py` — shared pretraining entrypoint. Contains `RichLoggingCallback` (prints tok/s, ETA, progress, elapsed, GPU memory on every log step) and wires it into `Trainer`.
- `sft_runner.py` — SFT training entrypoint. Uses TRL `SFTTrainer` + `DataCollatorForCompletionOnlyLM` for loss masking on assistant tokens only. Imports shared utilities from `train_runner.py`.
- `flow_validation.py` — validates pretraining (local/production) and SFT (local/production) configs stay isolated.
- `token_storage.py`, `local_smoke.py` — shard utilities and local tiny dataset generator.

**Attention**: MLA with low-rank latent projections. Q uses `q_down_proj -> q_up_proj`, while K/V share `kv_down_proj` before separate `k_up_proj` / `v_up_proj`. RoPE is applied after optional QK-Norm. `past_key_values` uses the `transformers.cache_utils.Cache` API.

**SFT loss masking**: `DataCollatorForCompletionOnlyLM` searches for `<|im_start|>assistant\n` token pattern and masks all non-assistant tokens with `-100`. The model sees full context but gradients are only computed on assistant responses.

## Production Scripts

Each production script has a single, independent responsibility. Do NOT merge their roles:

| Script | Purpose |
|---|---|
| `scripts/production/prepare_data.py` | Stream, filter, tokenize, shard data to `.npy` files |
| `scripts/production/pack_shards.py` | Pack `.npy` token shards into ~10GB `.tar.zst` archives for upload |
| `scripts/production/train.py` | Launch pretraining via `train_runner.run_training` |
| `scripts/production/upload_checkpoint_to_hf.py` | **Periodic** upload of training checkpoints to a HF **Model** repo (run alongside training) |
| `scripts/production/sft_prepare_data.py` | Download HF SFT datasets, convert to ChatML JSONL |
| `scripts/production/sft_train.py` | Launch SFT via `sft_runner.run_sft_training` |
| `scripts/production/run_prepare_loop.sh` | Looping wrapper for `prepare_data.py` (re-runs on failure) |

## Tests

- 19 unit tests in `tests/test_training_fixes.py` covering: gradient checkpointing backward, weight tying, residual init scaling, loss decrease (with/without QK-Norm), parameter count, KV-cache generation, RoPE numerical correctness, checkpoint ordering, data collation, cross-file packing, train/val splitting, EOS insertion, flow isolation validation, and others.
- 37 unit tests in `tests/test_sft.py` covering: SFT data loading (single file, directory, empty dir), train/val splitting, messages format validation, local/production SFT flow validation (tokenizer, DeepSpeed, path, pretrained_model_path rules), loss masking, TRL runner compatibility, and processing class fallback checks.

## Dependencies

- Base: PyTorch ≥ 2.1, transformers ≥ 4.38 (<6), pyyaml, accelerate, numpy.
- SFT extra (`uv sync --extra sft`): trl ≥ 0.12.
- Production extra (`uv sync --extra production`): datasets, deepspeed, huggingface_hub, pyarrow, wandb.

## Documentation Rule

**Always update `README.md` when any of the following change:**
- Training flow, scripts, or their entry points
- Configuration fields in `train.yaml`, `model.yaml`, `datasets.yaml`, `sft_train.yaml`, or `sft_datasets.yaml`
- New scripts added to `scripts/production/` or `scripts/local/`
- Logging format or monitoring behavior
- Checkpoint / upload strategy

README sections most likely to need updating: §1 (layout), §5 (data prep), §6 (production training), §9 (data pipeline).
