# Lite-LLM

从零（from-scratch）预训练一个 **~1.5B 参数的因果语言模型** 的工程脚手架。模型架构是 Llama 风格的 decoder-only Transformer：**Multi-head Latent Attention (MLA) + SwiGLU + RoPE + RMSNorm + QK-Norm**，并接入 HuggingFace `Trainer` 生态，支持单机多卡 + DeepSpeed ZeRO-2 训练，内置基于 `transformers.Cache` 的 KV cache 用于推理加速。

> 这是**预训练**项目：模型权重完全从随机初始化训出来，不继承任何已有 checkpoint。复用的只是 Qwen 的 tokenizer（vocab.json + merges.txt），目的是省掉自己训 BPE 的力气，并直接拥有覆盖中英代码数学的多语言词表。

---

## 1. 项目布局

```
lite-llm/
├── lite_llm/                    # 模型与训练核心库
│   ├── configuration.py         # LiteLlmConfig（PretrainedConfig 子类）
│   ├── modeling.py              # LiteLlmForCausalLM（PreTrainedModel + GenerationMixin）
│   ├── data_utils.py            # PretrainDataset / DataCollator / split_train_val
│   ├── sft_data_utils.py        # SFT 数据加载 / ChatML 格式校验
│   ├── train_runner.py          # 预训练共享入口（YAML → Trainer）
│   ├── sft_runner.py            # SFT 入口（TRL SFTTrainer + loss masking）
│   ├── flow_validation.py       # 隔离 local / production / pretrain / SFT 的硬规则
│   ├── token_storage.py         # 生产数据 shard 命名 / 续传辅助
│   └── local_smoke.py           # 本地确定性 smoke 数据生成
├── scripts/
│   ├── local/                   # 本地 CPU smoke flow
│   │   ├── prepare_data.py
│   │   ├── train.py
│   │   ├── sft_prepare_data.py  # SFT smoke 数据 + dummy checkpoint
│   │   └── sft_train.py
│   └── production/              # 生产 GPU + DeepSpeed flow
│       ├── prepare_data.py
│       ├── train.py
│       ├── sft_prepare_data.py  # SFT 数据集下载 → JSONL
│       ├── sft_train.py
│       ├── upload_to_hf.py              # 一次性上传 tokenized 数据到 HF Dataset
│       └── upload_checkpoint_to_hf.py   # 训练中定期上传 checkpoint 到 HF Model
├── configs/
│   ├── local/{model,train,sft_model,sft_train}.yaml
│   └── production/{model,train,sft_train,sft_datasets}.yaml + deepspeed_zero2.json
├── tests/
│   ├── test_training_fixes.py   # 预训练测试（19 个）
│   └── test_sft.py              # SFT 测试（16 个）
├── pyproject.toml               # uv 管理的依赖
└── .python-version              # 固定 Python 3.11
```

---

## 2. 两条互不穿越的 flow

整个仓库严格分成两条 flow × 两种模式，**`lite_llm/flow_validation.py` 在每次启动训练时强制校验**，越界直接 raise：

### 预训练（Pretraining）

|                       | Local (smoke test)              | Production (真实训练)                  |
|-----------------------|----------------------------------|----------------------------------------|
| 入口                  | `scripts/local/prepare_data.py` + `train.py` | `scripts/production/prepare_data.py` + `train.py` |
| 配置                  | `configs/local/{model,train}.yaml` | `configs/production/{model,train}.yaml + json` |
| 数据目录              | `./data/local_smoke/`            | `./data/production/`                   |
| 产物目录              | `./artifacts/local/`             | `./artifacts/production/`              |
| 设备                  | 必须 `use_cpu=true`              | 必须 `use_cpu=false`                   |
| Tokenizer             | 不能加载远程 tokenizer           | 必须设置 `tokenizer_name`              |
| DeepSpeed             | 禁止                             | 必须配置                               |
| 模型 vocab            | `vocab_size ≤ 4096`              | `vocab_size ≥ 10000`                   |
| 启动时是否 resume     | 必须 `false`（每次干净重跑）     | 必须 `true`（自动续训）                |

### SFT（监督微调）

|                       | Local (smoke test)              | Production (真实训练)                  |
|-----------------------|----------------------------------|----------------------------------------|
| 入口                  | `scripts/local/sft_prepare_data.py` + `sft_train.py` | `scripts/production/sft_prepare_data.py` + `sft_train.py` |
| 配置                  | `configs/local/{sft_model,sft_train}.yaml` | `configs/production/{sft_train,sft_datasets}.yaml` |
| 数据目录              | `./data/local_smoke/sft/`        | `./data/production/sft/`               |
| 产物目录              | `./artifacts/local/sft_*`        | `./artifacts/production/sft_*`         |
| 设备                  | 必须 `use_cpu=true`              | 必须 `use_cpu=false`                   |
| Tokenizer             | 必须设置（SFT 需要 chat template） | 必须设置                               |
| DeepSpeed             | 禁止                             | 必须配置                               |
| 预训练 checkpoint     | 必须指定（dummy 或真实）         | 必须指向 `./artifacts/production/` 下  |
| 依赖                  | `uv sync --extra sft`            | `uv sync --extra production --extra sft` |

如果你试图把生产数据指向本地路径、把 DeepSpeed 加进本地配置、或者 SFT flow 缺少 `pretrained_model_path`，启动时会立刻报错。

---

## 3. 环境

项目用 [**uv**](https://github.com/astral-sh/uv) 管理依赖，Python 版本由 `.python-version` 固定为 `3.11`：

```bash
# 本地开发 / smoke test 用（基础依赖：pytorch, transformers, datasets, pyyaml, accelerate, numpy）
uv sync

# SFT 微调用（额外加：trl）
uv sync --extra sft

# 生产训练用（额外加：deepspeed, wandb）
uv sync --extra production --frozen

# 生产 SFT（两者都需要）
uv sync --extra production --extra sft --frozen
```

依赖版本范围（见 `pyproject.toml`）：

- Python ≥ 3.11
- PyTorch ≥ 2.1, < 3
- transformers ≥ 4.38, < 6
- accelerate ≥ 0.26, < 2
- numpy ≥ 1.26, < 3

---

## 4. 本地 Smoke Test（CPU，秒级跑通）

用一份 4096 token 的确定性数据 + 一个 0.084M 参数的 tiny MLA 模型跑 8 步训练，目的是**在不依赖任何远程资源、不需要 GPU 的情况下，把训练管线（dataset → collator → model.forward → backward → optimizer → checkpoint → resume）一次性跑通**。

```bash
uv run python scripts/local/prepare_data.py
uv run python scripts/local/train.py
```

预期输出：8 步内 loss 从 ~5.6 单调下降到 ~5.0，产物写到 `./artifacts/local/checkpoints/`（`checkpoint-4`、`checkpoint-8`、`final`）。整个流程在普通笔记本上 < 1 秒。

本地 tiny 模型配置（`configs/local/model.yaml`）：

| 字段 | 值 |
|---|---|
| `vocab_size` | 256 |
| `hidden_size` | 64 |
| `intermediate_size` | 128 |
| `num_hidden_layers` | 2 |
| `attention_type` | `mla` |
| `num_attention_heads` | 4 |
| `head_dim` | 16 |
| `q_lora_rank` / `kv_lora_rank` | 16 / 16 |
| `max_position_embeddings` | 256 |

---

## 5. 生产数据准备

`scripts/production/prepare_data.py` 按四个类别从公开数据集流式（streaming）拉取，每条样本经质量过滤 → tokenize → 写成 **int32 扁平 token 数组**，每达到 ~50M token 就刷一个 shard 到磁盘（约 200 MB / shard）。

**默认目标配额**（`CATEGORY_TARGETS`，可改）：

| 类别 | 目标 token | 数据源 |
|---|---|---|
| English | 2.8B | `HuggingFaceFW/fineweb-edu` (`sample-100BT`, `educational_score ≥ 4`) |
| Chinese | 2.8B | `opencsg/Fineweb-Edu-Chinese-V2.1` + `Skywork/SkyPile-150B`（中文字符占比 ≥ 30%） |
| Code | 0.8B | `bigcode/the-stack-v2-train`（限 Top-10 主流语言） |
| Math | 1.6B | `HuggingFaceTB/finemath` (`finemath-4plus`) + `open-web-math/open-web-math` |
| **总计** | **8B** | |

**Shard 命名 / 续传**：每个数据源写成 `{name}-00000.npy`、`{name}-00001.npy`…，下次重跑时自动从最大 shard 索引继续，已经达标的源直接跳过。这样可以放心断点续传或分多次跑。

**文档边界**：每篇文档末尾追加一个 `tokenizer.eos_token_id`，用作 packing 阶段的"文档结束"信号。

> **关于 EOS 的取舍**：当前生产配置用的是 chat 版的 `Qwen/Qwen3.5-0.8B`，它的 `eos_token = <|im_end|>`（聊天回合结束符）。chat 版和 base 版（`Qwen/Qwen3.5-0.8B-Base`，`eos_token = <|endoftext|>`）的词表、merges、token id **完全一样**，唯一区别就是 `eos_token` 字段指向哪个 special token。我们已知此处用 chat 版，未来若做 chat-style SFT，需要意识到 `<|im_end|>` 在预训练阶段就被当成"文档分隔符"训练过了。

**常用命令**：

```bash
# 估算磁盘 / token 规模，不实际下载
uv run python scripts/production/prepare_data.py --dry-run

# 试跑：抽 1% 配额，用于 sanity check
uv run python scripts/production/prepare_data.py --scale 0.01

# 只跑某些类别
uv run python scripts/production/prepare_data.py --categories english,math

# 改输出目录（必须仍在 ./data/production/ 下）
uv run python scripts/production/prepare_data.py --output-dir ./data/production/tokenized
```

需要调整数据源 / 过滤规则 / 配额，直接改 `scripts/production/prepare_data.py` 顶部的 `DATASET_SPECS`、`CATEGORY_TARGETS`、`filter_*` 函数即可。

---

## 6. 生产训练

### 6.1 模型配置（`configs/production/model.yaml`）

| 字段 | 值 | 说明 |
|---|---|---|
| `vocab_size` | 248320 | Qwen tokenizer 长度 248077 向上取到 128 倍数（多留一档余量） |
| `hidden_size` | 2048 | |
| `intermediate_size` | 5504 | |
| `num_hidden_layers` | 24 | |
| `attention_type` | `mla` | Multi-head Latent Attention |
| `num_attention_heads` | 16 | |
| `head_dim` | 128 | |
| `q_lora_rank` / `kv_lora_rank` | 512 / 192 | latent bottleneck rank |
| `max_position_embeddings` | 8192 | 上下文长度 |
| `rope_theta` | 10000 | 固定 RoPE base |
| `qk_norm` | true | Q/K 逐 head RMSNorm |
| `tie_word_embeddings` | true | `lm_head` 与 `embed_tokens` 共享权重 |

**实际参数量**（已验证）：

```
total          : 1,499,570,176  (1.500 B)
  embed/lm_head:   508,559,360  (509 M, tied)
  non-embed    :   991,010,816  (991 M)
```

### 6.2 训练配置（`configs/production/train.yaml`）

要点：

- **Optimizer**: AdamW，`lr=3e-4`、`weight_decay=0.1`、`betas=(0.9, 0.95)`、`max_grad_norm=1.0`
- **Scheduler**: cosine，`warmup_ratio=0.05`
- **Batch**: `per_device=2`，`grad_accum=16`；若按 `8 GPUs` 训练，则每次 optimizer step 约 `2.1M tokens`
- **Seq length**: 8192
- **精度**: bf16
- **显存优化**: HF gradient checkpointing 打开（不在 DeepSpeed 里配，由 `train_runner` 通过 TrainingArguments 启用）
- **Eval**: `val_fraction=0.005` 从 token 流尾部留出，每 2000 步算一次 eval loss / perplexity；上限 `max_val_tokens=50_000_000`
- **Checkpointing**: 每 2000 步保存，`save_total_limit=5`
- **断点续训**: `resume_from_last_checkpoint=true`，启动时 `train_runner` 自动找 `output_dir` 下最大 step 的 `checkpoint-*` 恢复
- **DeepSpeed**: ZeRO-2（`configs/production/deepspeed_zero2.json`），优化器和参数都不 offload，开启 `overlap_comm` / `reduce_scatter` / `contiguous_gradients`

### 6.3 启动

```bash
# 单机多卡（DeepSpeed ZeRO-2，推荐）
uv run deepspeed scripts/production/train.py

# 不用 DeepSpeed：把 train.yaml 里的 deepspeed 字段去掉再启动
uv run python scripts/production/train.py
```

显式指定配置文件（默认用 `configs/production/{train,model}.yaml`）：

```bash
uv run python scripts/production/train.py \
  --train-config configs/production/train.yaml \
  --model-config configs/production/model.yaml
```

训练结束后，最终权重写到 `./artifacts/production/checkpoints/final/`，并把 tokenizer 一起序列化进去，方便后续 `AutoTokenizer.from_pretrained(...)` 直接加载。

### 6.4 训练日志

`train_runner.py` 内置 `RichLoggingCallback`，每次 Trainer 触发日志（默认每 10 步）时，在原有的 loss/lr/epoch 输出下方额外打印一行：

```
  [train] tok/s=2.34k  progress=12.3%  (246/2000)  ETA=3:42:15  elapsed=0:31:08  gpu_mem=38.2GB
```

| 字段 | 含义 |
|---|---|
| `tok/s` | 从上次日志到本次的区间吞吐量（tokens/sec，取对数 k 为单位） |
| `progress` | 当前 step / 总 step 及百分比 |
| `ETA` | 基于训练开始至今平均速度推算的剩余时间 |
| `elapsed` | 已训练时长 |
| `gpu_mem` | 当前 rank 保留的 GPU 显存峰值（CUDA 可用时输出） |

多卡场景下只有 rank 0 打印，不会重复输出。`tokens_per_step` 自动从 `TrainingArguments.world_size × batch_size × grad_accum × seq_len` 计算，无需额外配置。

### 6.5 Checkpoint 定期上传到 HuggingFace

训练期间在**另一个终端**并行运行上传脚本，每小时把最新完整 checkpoint 推送到 HF Model repo：

```bash
export HF_TOKEN=hf_xxxxxxxxxxxx

python scripts/production/upload_checkpoint_to_hf.py \
    --repo username/lite-llm-checkpoints \
    --interval 3600          # 秒，默认 3600（1 小时）
```

脚本行为：

- 扫描 `output_dir`（默认从 `configs/production/train.yaml` 读取）下 step 编号最大、且 `trainer_state.json` 存在（写完整）的 `checkpoint-*` 目录
- 已上传过同一 step 直接跳过，**幂等**，可重启不重传
- 上传失败只打警告，不阻断训练，下次仍会重试
- `--once` 参数可一次性手动触发（适合 cron 调度）
- `--private` 参数控制首次创建 repo 时的可见性

常用参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--repo` | （必填） | HF model repo id，如 `NoBey/lite-llm-ckpt` |
| `--output-dir` | 读 train.yaml | 本地 checkpoint 目录 |
| `--config` | `configs/production/train.yaml` | 用于读取 output_dir |
| `--token` | `$HF_TOKEN` | HF 写权限 token |
| `--interval` | `3600` | 上传间隔（秒） |
| `--once` | — | 上传一次后退出 |
| `--private` | — | 建 repo 时设为私有 |

> **与 `upload_to_hf.py` 的区别**：`upload_to_hf.py` 是一次性上传 tokenized `.npy` 数据到 HF **Dataset** repo，与 checkpoint 上传完全独立，互不影响。

---

## 7. 推理 / 生成

```python
import torch
from transformers import AutoTokenizer
from lite_llm.modeling import LiteLlmForCausalLM

ckpt = "./artifacts/production/checkpoints/final"
tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
model = LiteLlmForCausalLM.from_pretrained(ckpt).eval()

ids = tok("今天天气", return_tensors="pt", add_special_tokens=False).input_ids
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=64, do_sample=False)
print(tok.decode(out[0], skip_special_tokens=True))
```

`generate` 自动用模型内置的 KV cache（`transformers.cache_utils.Cache` 接口），prefill 之后每步只前传新加入的那 1 个 token，不会全量重算前缀。

> 因为 chat 版 tokenizer 默认会在编码时追加 `<|im_end|>`，**做基座生成时记得 `add_special_tokens=False`**，否则 prompt 末尾会被插一个 EOS、模型可能立刻停。

---

## 8. 模型实现要点（`lite_llm/modeling.py`）

| 组件 | 实现细节 |
|---|---|
| **RMSNorm** | 计算全程 fp32，最后 cast 回原 dtype（Llama 风格） |
| **RoPE** | Llama 风格 half-split；cos/sin 在 `__init__` 时按 `max_position_embeddings` 一次性建好 buffer，运行时直接索引 |
| **MLA** | Q 先经过 `q_down_proj -> q_up_proj`，K/V 先共享 `kv_down_proj` 再分别走 `k_up_proj` / `v_up_proj`，用低秩 latent bottleneck 替代直接的全维 QKV 投影 |
| **QK-Norm** | `qk_norm=true` 时，对每个 head 的 Q、K 各做一次 RMSNorm，再加 RoPE（来自 ViT-22B / Chameleon / Qwen 系列的稳定性 trick） |
| **SwiGLU FFN** | `down_proj(silu(gate_proj(x)) * up_proj(x))` |
| **权重初始化** | `_init_weights`：Linear/Embedding 用 `N(0, initializer_range)`，RMSNorm fill 1。`post_init` 里再把 `o_proj` / `down_proj` 的权重乘 `1/√(2·num_hidden_layers)`（GPT-2 风格残差稳定性） |
| **Weight tying** | 通过 `_tied_weights_keys = ["lm_head.weight"]` 声明，HF 自动 tie |
| **KV cache** | `forward` 接受 `past_key_values: transformers.cache_utils.Cache`；训练时强制 `use_cache=False` 避免存中间 K/V |
| **Attention mask** | `_build_sdpa_mask` 把 HF 的 `[B, S]` padding mask 转成 SDPA 用的 `[B, 1, q_len, kv_len]` bool mask；如果没 padding，直接走 `is_causal=True` 的 SDPA fast path |
| **Gradient checkpointing** | 通过 `supports_gradient_checkpointing=True` + `_gradient_checkpointing_func`，由 `Trainer` 通过 TrainingArguments 启用 |

---

## 9. 数据管线要点（`lite_llm/data_utils.py`）

- **`PretrainDataset`**：把多个 `.npy` / `.bin` 文件**逻辑上拼接**（用 `bisect` 加 cumulative-sum 索引），按 `max_seq_length` 切定长序列。文档边界**不**做 attention mask 隔离 —— 边界靠 prepare 阶段插入的 EOS token 自然学习。
- **`load_tokenized_dataset(data_dir, max_seq_length)`**：一行加载 `data_dir` 下所有 `.npy` / `.bin`（mmap 模式，不爆内存）。
- **`split_train_val(data_dir, max_seq_length, val_fraction, max_val_tokens)`**：从 token 流**尾部**截出 eval 集，确定性、不需要 shuffle、reproducible。
- **`DataCollatorForPretraining`**：`input_ids` 直接复制成 `labels`，loss 在模型内部 shift。
- **`tokenize_and_save(texts, tokenizer, output_path, add_eos=True)`**：单元测试用的小工具，每篇文档结尾插 EOS。

---

## 10. SFT 监督微调

预训练完成后，可以通过 SFT（Supervised Fine-Tuning）让模型学会对话格式。SFT 使用 TRL 库的 `SFTTrainer` + `DataCollatorForCompletionOnlyLM`，只对 assistant 回复计算 loss（loss masking）。

### 10.1 数据格式

SFT 数据采用 OpenAI ChatML 格式（JSONL），每行一条对话：

```json
{"messages": [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！有什么可以帮你的？"}]}
```

支持的字段：`messages`（必需），每条消息包含 `role`（system/user/assistant）和 `content`。Qwen tokenizer 的 `apply_chat_template()` 会自动将其转换为 ChatML 格式（`<|im_start|>system\n...<|im_end|>` 等）。

### 10.2 Loss Masking

`DataCollatorForCompletionOnlyLM` 在 tokenized 序列中搜索 `<|im_start|>assistant\n` 的 token 模式，将所有非 assistant token 的 label 设为 `-100`。模型在 forward 时看到完整上下文，但只在 assistant 回复上计算梯度。

### 10.3 本地 SFT Smoke Test

```bash
uv sync --extra sft

# 生成 tiny JSONL 数据 + dummy 预训练 checkpoint（random init）
uv run python scripts/local/sft_prepare_data.py

# 跑 4 步 SFT 训练（CPU，秒级）
uv run python scripts/local/sft_train.py
```

本地 SFT 使用 `vocab_size=248320`（匹配 Qwen tokenizer）但保持 tiny 架构（2 层、hidden=64），确保 smoke test 自包含，不依赖真实预训练结果。

### 10.4 生产 SFT

```bash
uv sync --extra production --extra sft --frozen

# 下载 SFT 数据集并转换为 JSONL
uv run python scripts/production/sft_prepare_data.py

# 启动 SFT 训练（从预训练 checkpoint 加载）
uv run deepspeed scripts/production/sft_train.py
```

SFT 训练配置（`configs/production/sft_train.yaml`）与预训练的关键差异：

| 配置项 | 预训练 | SFT |
|--------|--------|-----|
| `learning_rate` | 3e-4 | 2e-5 |
| `num_train_epochs` | 1 | 3 |
| `max_seq_length` | 8192 | 4096 |
| `per_device_train_batch_size` | 2 | 1 |
| `warmup_ratio` | 0.05 | 0.1 |
| `pretrained_model_path` | 无（随机初始化） | `./artifacts/production/checkpoints/final` |

数据集规格在 `configs/production/sft_datasets.yaml` 中配置，支持多种 HF 数据集格式（messages 格式直用、OASST 格式自动转换等）。

---

## 11. 测试

```bash
uv run python -m unittest tests.test_training_fixes -v
uv run python -m unittest tests.test_sft -v
```

**预训练测试**（19 个用例）：

- **建模**：weight tying 真共享一份 storage、residual 投影 init 后被缩小、tiny 模型在固定 batch 上 loss 单调下降、`generate` 走 KV cache 路径不报错、RoPE 与手算公式数值一致、RoPE 不改变向量 norm
- **训练管线**：`gradient_checkpointing_enable` 后能正常 backward、`find_last_checkpoint` 按 step 数值排序而非字典序
- **数据**：跨文件 packing 正确、`split_train_val` 从尾部切 eval 且空 val 时返回 `None`、`tokenize_and_save` 在文档间插 EOS、shard 命名 / 续传索引正确、smoke token 全部落在 vocab 内
- **隔离**：local 与 production 配置 YAML 实际加载后能通过对应的 flow validation

**SFT 测试**（16 个用例）：

- **数据加载**：单文件 JSONL、目录批量加载、空目录报错
- **数据拆分**：train/val 比例正确、0 比例返回 None
- **格式校验**：合法数据通过、缺 messages/role/content 报错
- **Flow 隔离**：local SFT 要求 tokenizer、禁止 DeepSpeed、要求 pretrained_model_path；production SFT 要求 DeepSpeed、路径隔离

---

## 12. 许可证与第三方资源

本仓库本身是研究 / 学习用的预训练脚手架。所用的**上游数据集**（FineWeb-Edu, SkyPile, the-Stack v2, FineMath, OpenWebMath 等）和**分词器**（Qwen/Qwen3.5-0.8B）各自遵循其在 HuggingFace Hub 上的 license，商用前请自行核对每个数据源的条款。
