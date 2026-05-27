# Lite-LLM SFT Flow

This directory is a parallel supervised fine-tuning flow. It intentionally keeps
SFT code, configs, scripts, and dataset recipes out of the pretraining flow.

The only shared dependency is the base model implementation in `lite_llm/`.

## Layout

```
sft/
├── collator.py                         # ChatML rendering + assistant-only loss mask
├── data_utils.py                       # JSONL loading / validation / train-val split
├── flow_validation.py                  # SFT-specific local/production isolation rules
├── runner.py                           # HF Trainer SFT entrypoint
├── toy_tokenizer.py                    # Offline local smoke tokenizer
├── configs/
│   ├── local/{model,train}.yaml
│   └── production/{datasets,train}_*.yaml
└── scripts/
    ├── local/{prepare_data,train}.py
    └── production/{download_data,train}.py
```

## Data Format

All SFT data is normalized to JSONL with one conversation per line:

```json
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}],"source":"dataset_name"}
```

Training renders conversations as ChatML:

```text
<|im_start|>user
...
<|im_end|>
<|im_start|>assistant
...
<|im_end|>
```

Only assistant response tokens and the assistant `<|im_end|>` are supervised;
system/user/header/padding tokens are masked with `-100`.

## Local Smoke

```bash
uv run python sft/scripts/local/prepare_data.py
uv run python sft/scripts/local/train.py
```

The local flow uses a tiny random checkpoint and an offline toy ChatML tokenizer.

## Production

```bash
# 3B-pretrain-checkpoint SFT data
uv run python sft/scripts/production/download_data.py \
  --datasets-config sft/configs/production/datasets_zh_first_3b.yaml

# 3B-pretrain-checkpoint SFT
uv run deepspeed sft/scripts/production/train.py \
  --train-config sft/configs/production/train_zh_first_3b.yaml \
  --model-config configs/production/model.yaml

# 20B-pretrain-checkpoint SFT
uv run deepspeed sft/scripts/production/train.py \
  --train-config sft/configs/production/train_zh_first_20b.yaml \
  --model-config configs/production/model.yaml
```

Production SFT expects the corresponding pretraining checkpoint under the
`pretrained_model_path` configured in the train YAML.

