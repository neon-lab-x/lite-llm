# Lite-LLM

从零（from-scratch）预训练一个 **~1.0B 参数的因果语言模型** 的工程脚手架。模型架构是 Llama 风格的 decoder-only Transformer：**Multi-head Latent Attention (MLA) + SwiGLU + RoPE + RMSNorm + QK-Norm**，并接入 HuggingFace `Trainer` 生态，支持单机多卡 + DeepSpeed ZeRO-2 训练，内置基于 `transformers.Cache` 的 KV cache 用于推理加速。SFT 作为一套平行 flow 独立放在 `sft/`，只复用 `lite_llm/` 里的模型实现。

> 主线是**预训练**：模型权重完全从随机初始化训出来，不继承任何已有 checkpoint。复用的只是 Qwen 的 tokenizer（vocab.json + merges.txt），目的是省掉自己训 BPE 的力气，并直接拥有覆盖中英代码数学的多语言词表。`sft/` 中的监督微调 flow 从预训练 checkpoint 继续训练。

---

## 1. 项目布局

```
lite-llm/
├── lite_llm/                    # 模型与训练核心库
│   ├── configuration.py         # LiteLlmConfig（PretrainedConfig 子类）
│   ├── modeling.py              # LiteLlmForCausalLM（PreTrainedModel + GenerationMixin）
│   ├── data_utils.py            # PretrainDataset / DataCollator / split_train_val
│   ├── train_runner.py          # 预训练共享入口（YAML → Trainer）
│   ├── flow_validation.py       # 隔离 local / production 的硬规则
│   ├── token_storage.py         # 生产数据 shard 命名 / 续传辅助
│   └── local_smoke.py           # 本地确定性 smoke 数据生成
├── scripts/
│   ├── local/                   # 本地 CPU smoke flow
│   │   ├── prepare_data.py
│   │   └── train.py
│   └── production/              # 生产 GPU + DeepSpeed flow
│       ├── prepare_data.py
│       ├── download_data.py             # HF 源数据 → 本地筛选 raw parquet
│       ├── tokenize_raw_data.py         # 本地 raw parquet → token shard
│       ├── train.py
│       ├── pack_shards.py               # .npy shard 打包成 ~10GB tar 归档
│       ├── run_prepare_loop.sh          # prepare_data.py 循环包装（失败自动重跑）
│       └── upload_checkpoint_to_hf.py   # 训练中定期上传 checkpoint 到 HF Model
├── sft/                         # 独立 SFT flow（代码/配置/脚本/数据 recipe）
│   ├── collator.py              # ChatML 渲染 + assistant-only loss mask
│   ├── data_utils.py            # SFT JSONL 加载 / 校验 / train-val split
│   ├── runner.py                # SFT Trainer 入口（从预训练 checkpoint 加载）
│   ├── configs/
│   │   ├── local/{model,train}.yaml
│   │   └── production/{datasets,train}_*.yaml
│   └── scripts/
│       ├── local/{prepare_data,train}.py
│       └── production/{download_data,train}.py
├── configs/
│   ├── local/{model,train}.yaml
│   └── production/{model,train,datasets*}.yaml + deepspeed_zero2.json
├── tests/
│   └── test_training_fixes.py   # 预训练测试（24 个）
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

如果你试图把生产数据指向本地路径、把 DeepSpeed 加进本地配置，启动时会立刻报错。

### SFT（Supervised Fine-Tuning）

SFT 不混入 `lite_llm/` 的预训练数据管线，也不新增 `scripts/local/sft_*` 或 `scripts/production/sft_*` 入口；所有 SFT 专属内容都在 `sft/` 目录下：

|                       | Local (smoke test)              | Production (真实 SFT)                  |
|-----------------------|----------------------------------|----------------------------------------|
| 入口                  | `sft/scripts/local/prepare_data.py` + `train.py` | `sft/scripts/production/download_data.py` + `train.py` |
| 配置                  | `sft/configs/local/{model,train}.yaml` | `sft/configs/production/{datasets,train}_*.yaml` |
| 数据格式              | ChatML messages JSONL            | ChatML messages JSONL                  |
| Loss mask             | 只训练 assistant 内容和 assistant `<|im_end|>` | 同左 |
| 训练起点              | tiny smoke checkpoint            | 预训练 checkpoint 的 `final/`          |

本地 smoke：

```bash
uv run python sft/scripts/local/prepare_data.py
uv run python sft/scripts/local/train.py
```

生产 3B recipe SFT：

```bash
uv run python sft/scripts/production/download_data.py \
  --datasets-config sft/configs/production/datasets_zh_first_3b.yaml

uv run deepspeed sft/scripts/production/train.py \
  --train-config sft/configs/production/train_zh_first_3b.yaml \
  --model-config configs/production/model.yaml
```

生产 20B recipe SFT：

```bash
uv run python sft/scripts/production/download_data.py \
  --datasets-config sft/configs/production/datasets_zh_first_20b.yaml

uv run deepspeed sft/scripts/production/train.py \
  --train-config sft/configs/production/train_zh_first_20b.yaml \
  --model-config configs/production/model.yaml
```

---

## 3. 环境

项目用 [**uv**](https://github.com/astral-sh/uv) 管理依赖，Python 版本由 `.python-version` 固定为 `3.11`：

```bash
# 本地开发 / smoke test 用（基础依赖：pytorch, transformers, datasets, pyyaml, accelerate, numpy）
uv sync

# 生产训练用（额外加：deepspeed, wandb）
uv sync --extra production --frozen
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

`scripts/production/prepare_data.py` 按中文优先配方从公开数据集逐文件下载、过滤、tokenize，写成 **int32 扁平 token 数组**，每达到 ~5M token 就刷一个 shard 到磁盘（约 20 MB / shard）。数据集规格可通过 `--datasets-config` 指定，默认使用 `configs/production/datasets.yaml`，也就是 3B MVP 配方。

现有 recipe：

| 配置文件 | 用途 |
|---|---|
| `configs/production/datasets.yaml` | 当前默认，等同 `zh_first_v1_3b` |
| `configs/production/datasets_zh_first_3b.yaml` | 3B token 工程验证版，raw 输出到 `data/production/raw_zh_first_3b` |
| `configs/production/datasets_zh_first_20b.yaml` | 20B 正式中文优先版（19B active + 1B reserved），raw 输出到 `/root/autodl-fs/raw_zh_first_20b` |
| `configs/production/datasets_ablation.yaml` | 小规模消融实验池 |
| `configs/production/datasets_en_first_legacy.yaml` | 原英文主导配方，保留作对比 |

为避免“只取字典序靠前 shard”的偏差，生产脚本会用固定 seed 对每个数据集的文件顺序、parquet row group 顺序、row 顺序做确定性打乱。这样 3B MVP 跑到的是全数据源的均匀样本，而不是某个文件名前缀切片。

磁盘保护默认写在各 dataset recipe 的 `disk` 区块：至少保留 80 GB 空闲空间，临时 HF 下载缓存上限 60 GB。每个源文件处理完后会清理独立下载缓存，避免 HF blob cache 在 400 GB 机器上越积越多。

**3B MVP 配方**（`configs/production/datasets_zh_first_3b.yaml`）：

| # | 数据集名 | 数据源 | 过滤规则 | 目标 tokens |
|---|---|---|---|---|
| 1 | `zh_fineweb_edu_v21` | `opencsg/Fineweb-Edu-Chinese-V2.1` | 中文占比 >= 35%，长度 200-12000，质量分 >= 3 | 1.4B |
| 2 | `baai_cci3_hq` | `BAAI/CCI3-HQ` | 中文占比 >= 35%，长度 200-12000 | 500M |
| 3 | `skypile_150b` | `Skywork/SkyPile-150B` | 中文占比 >= 35%，长度 200-12000 | 300M |
| 4 | `zh_cosmopedia` | `opencsg/chinese-cosmopedia` | 中文占比 >= 40%，长度 300-12000 | 200M |
| 5 | `en_fineweb_edu` | `HuggingFaceFW/fineweb-edu` | 长度 200-12000，质量分 >= 3 | 200M |
| 6 | `finemath_4plus` | `HuggingFaceTB/finemath` (`finemath-4plus`) | 长度 200-16000 | 250M |
| 7 | `github_code_clean` | `codeparrot/github-code-clean` | 长度 100-20000，白名单语言 | 150M |
| | **合计** | | | **3B** |

**20B 正式配方**（`configs/production/datasets_zh_first_20b.yaml`）：同一套来源与过滤规则，active target 为 FineWeb-Edu Chinese 9B、CCI3-HQ 3B、SkyPile 2B、Chinese Cosmopedia 1.2B、English FineWeb-Edu 1.2B、FineMath 1.6B、GitHub Code Clean 1B；另保留 1B `reserved_experiment` 关闭项用于后续领域消融。

`BAAI/CCI3-HQ` 是 gated dataset；第一次使用前需要在 HuggingFace 页面同意条款，`--local-only` 下载时也可以传 `--hf-token` 供读取使用。

推荐两段式生产链路：

1. `scripts/production/download_data.py`：从 HF 巨大源数据中按 recipe 过滤出高质量文档，写成本地 raw parquet，并清理下载缓存。
2. `scripts/production/tokenize_raw_data.py`：完全从本地 raw parquet 做 CPU tokenizer，写 `{name}-00000.npy` 训练 shard，可同时上传 HF Dataset repo。

**Shard 命名 / 续传**：token shard 写成 `{name}-00000.npy`、`{name}-00001.npy`…，下次重跑时自动从最大 shard 索引继续。raw 下载状态写在 raw 目录 `_state/`，tokenize 状态写在 tokenized 目录 `_cache/state/`。

**文档边界**：每篇文档末尾追加一个 `tokenizer.eos_token_id`，用作 packing 阶段的"文档结束"信号。

> **关于 EOS 的取舍**：当前生产配置用的是 chat 版的 `Qwen/Qwen3.5-0.8B`，它的 `eos_token = <|im_end|>`（聊天回合结束符）。chat 版和 base 版（`Qwen/Qwen3.5-0.8B-Base`，`eos_token = <|endoftext|>`）的词表、merges、token id **完全一样**，唯一区别就是 `eos_token` 字段指向哪个 special token。我们已知此处用 chat 版，未来若做 chat-style 微调，需要意识到 `<|im_end|>` 在预训练阶段就被当成"文档分隔符"训练过了。

**常用命令**：

```bash
# 只看计划：远端文件数量/大小、目标 token、预计本地 shard 磁盘
uv run python scripts/production/prepare_data.py --plan-only

# 3B MVP 第一步：只下载/筛选 raw parquet，不 tokenize
uv run python scripts/production/download_data.py \
  --datasets-config configs/production/datasets_zh_first_3b.yaml \
  --hf-token $HF_TOKEN \
  --no-mirror

# 3B MVP 第二步：从本地 raw parquet CPU tokenize，保留本地 shard
uv run python scripts/production/tokenize_raw_data.py \
  --datasets-config configs/production/datasets_zh_first_3b.yaml \
  --local-only

# tokenize 并上传 tokenized shard 到 HF Dataset repo，同时保留本地 shard
uv run python scripts/production/tokenize_raw_data.py \
  --datasets-config configs/production/datasets_zh_first_3b.yaml \
  --hf-token $HF_TOKEN \
  --hf-repo username/lite-llm-tokenized \
  --hf-path zh_first_v1_3b \
  --keep-uploaded

# 消融实验：只跑指定 slice
uv run python scripts/production/download_data.py \
  --datasets-config configs/production/datasets_ablation.yaml \
  --datasets ablate_zh_fineweb_score3 \
  --no-mirror

# 改 raw 输出目录和磁盘水位
uv run python scripts/production/download_data.py \
  --datasets-config configs/production/datasets_zh_first_3b.yaml \
  --download-dir /data/raw_zh_first_3b \
  --min-free-gb 80 \
  --max-cache-gb 60
```

需要调整数据源 / 过滤规则 / 配额，优先复制 `datasets_zh_first_3b.yaml` 或 `datasets_zh_first_20b.yaml` 新建 recipe，不要覆盖 legacy 配方。

旧的 `scripts/fast_prepare.py` 已删除；它使用 streaming 原始顺序，局部下载时容易重新引入数据源顺序偏差。生产数据准备统一走 `download_data.py` + `tokenize_raw_data.py`。

---

## 6. 生产训练

### 6.1 模型配置（`configs/production/model.yaml`）

| 字段 | 值 | 说明 |
|---|---|---|
| `vocab_size` | 248320 | Qwen tokenizer 长度 248077 向上取到 128 倍数（多留一档余量） |
| `hidden_size` | 1536 | |
| `intermediate_size` | 4608 | |
| `num_hidden_layers` | 24 | |
| `attention_type` | `mla` | Multi-head Latent Attention |
| `num_attention_heads` | 12 | |
| `head_dim` | 128 | |
| `q_lora_rank` / `kv_lora_rank` | 384 / 128 | latent bottleneck rank |
| `max_position_embeddings` | 8192 | 上下文长度 |
| `rope_theta` | 10000 | 固定 RoPE base |
| `qk_norm` | true | Q/K 逐 head RMSNorm |
| `tie_word_embeddings` | true | `lm_head` 与 `embed_tokens` 共享权重 |

**实际参数量**（已验证）：

```
total          : 990,199,296  (990 M)
  embed/lm_head: 381,419,520  (381 M, tied)
  non-embed    : 608,779,776  (609 M)
```

### 6.2 训练配置

要点：

- **Optimizer**: AdamW，`lr=3e-4`、`weight_decay=0.1`、`betas=(0.9, 0.95)`、`max_grad_norm=1.0`
- **Scheduler**: cosine，`warmup_ratio=0.05`
- **Batch**: `per_device=2`，`grad_accum=16`；若按 `4 GPUs` 训练，则每次 optimizer step 约 `1.0M tokens`
- **Seq length**: 8192
- **精度**: bf16
- **显存优化**: HF gradient checkpointing 打开（不在 DeepSpeed 里配，由 `train_runner` 通过 TrainingArguments 启用）
- **Eval**: `val_fraction=0.005` 从 token 流尾部留出，每 2000 步算一次 eval loss / perplexity；上限 `max_val_tokens=50_000_000`
- **Checkpointing**: 每 2000 步保存，`save_total_limit=5`
- **断点续训**: `resume_from_last_checkpoint=true`，启动时 `train_runner` 自动找 `output_dir` 下最大 step 的 `checkpoint-*` 恢复
- **DeepSpeed**: ZeRO-2（`configs/production/deepspeed_zero2.json`），优化器和参数都不 offload，开启 `overlap_comm` / `reduce_scatter` / `contiguous_gradients`

训练 recipe：

| 配置文件 | 数据目录 | checkpoint 目录 | eval/save 间隔 |
|---|---|---|---|
| `configs/production/train_zh_first_3b.yaml` | `./data/production/tokenized_zh_first_3b` | `./artifacts/production/checkpoints_zh_first_3b` | 500 steps |
| `configs/production/train_zh_first_20b.yaml` | `/root/autodl-fs/tokenized_zh_first_20b` | `./artifacts/production/checkpoints_zh_first_20b` | 2000 steps |
| `configs/production/train.yaml` | `/root/autodl-fs/tokenized` | `./artifacts/production/checkpoints` | 2000 steps |

### 6.3 启动

```bash
# 3B MVP 训练
uv run deepspeed scripts/production/train.py \
  --train-config configs/production/train_zh_first_3b.yaml \
  --model-config configs/production/model.yaml

# 20B 正式训练
uv run deepspeed scripts/production/train.py \
  --train-config configs/production/train_zh_first_20b.yaml \
  --model-config configs/production/model.yaml
```

不使用 DeepSpeed 时，把对应 train config 里的 `deepspeed` 字段去掉，再用 `uv run python scripts/production/train.py ...` 启动。训练结束后，最终权重写到对应 `output_dir/final/`，并把 tokenizer 一起序列化进去，方便后续 `AutoTokenizer.from_pretrained(...)` 直接加载。

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
    --config configs/production/train_zh_first_3b.yaml \
    --interval 3600          # 秒，默认 3600（1 小时）
```

脚本行为：

- 扫描 `output_dir`（默认从 `configs/production/train.yaml` 读取；3B/20B recipe 建议传对应 `--config`）下 step 编号最大、且 `trainer_state.json` 存在（写完整）的 `checkpoint-*` 目录
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

---

## 10. 测试

```bash
uv run python -m unittest tests.test_training_fixes -v
```

**预训练测试**（24 个用例）：

- **建模**：weight tying 真共享一份 storage、residual 投影 init 后被缩小、tiny 模型在固定 batch 上 loss 单调下降、`generate` 走 KV cache 路径不报错、RoPE 与手算公式数值一致、RoPE 不改变向量 norm
- **训练管线**：`gradient_checkpointing_enable` 后能正常 backward、`find_last_checkpoint` 按 step 数值排序而非字典序
- **数据**：跨文件 packing 正确、`split_train_val` 从尾部切 eval 且空 val 时返回 `None`、`tokenize_and_save` 在文档间插 EOS、shard 命名 / 续传索引正确、生产数据文件顺序可复现打乱、inline 过滤器与 recipe 字段别名正确、text 列 fallback 正确、smoke token 全部落在 vocab 内
- **隔离**：local 与 production 配置 YAML 实际加载后能通过对应的 flow validation

---

## 12. 许可证与第三方资源

本仓库本身是研究 / 学习用的预训练脚手架。所用的**上游数据集**（OpenCSG Chinese FineWeb Edu、BAAI CCI3-HQ、SkyPile、ChineseWebText2.0、Chinese Cosmopedia、FineWeb-Edu、FineMath、GitHub Code 等）和**分词器**（Qwen/Qwen3.5-0.8B）各自遵循其在 HuggingFace Hub 上的 license。部分数据集可能需要登录、同意使用条款或额外商用许可，正式训练前请逐项核对。
