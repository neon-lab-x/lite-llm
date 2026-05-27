# Lite-LLM SFT 架构与部署指南

> 最后更新：2026-05-28

---

## 1. 概述

SFT（Supervised Fine-Tuning）作为独立平行 flow 放在 `sft/` 目录下，只复用 `lite_llm/` 的模型实现（`LiteLlmConfig` + `LiteLlmForCausalLM`），不混入预训练的任何数据管线、脚本或配置。

设计原则：
- **完全隔离**：SFT 有自己的 `flow_validation.py`，local/production 路径、设备、DeepSpeed 配置互相不可越界
- **ChatML 格式统一**：所有数据源在 `download_data.py` 阶段归一化为 messages JSONL，训练时用 `<|im_start|>` / `<|im_end|>` 渲染
- **只训练 assistant 回复**：user/system 的 header + content 全部 mask 为 `-100`，只有 assistant content 和其后的 `<|im_end|>` 参与_loss
- **从预训练 checkpoint 加载**：`runner.py` 通过 `LiteLlmForCausalLM.from_pretrained()` 加载预训练的 `final/` 权重

---

## 2. 目录结构

```
sft/
├── __init__.py
├── README.md                         # SFT flow 自述文档
├── collator.py                       # DataCollatorForChatSFT：ChatML 渲染 + loss mask
├── data_utils.py                     # SftRecord / JSONL 加载 / 校验 / train-val split
├── flow_validation.py                # local/production 隔离校验
├── runner.py                         # SFT Trainer 入口（加载 → 校验 → 训练 → 保存）
├── toy_tokenizer.py                  # 本地 smoke 用离线 tokenizer（不依赖 Qwen 下载）
├── configs/
│   ├── local/
│   │   ├── model.yaml                # tiny MLA 模型（与预训练 local 相同架构，vocab=256）
│   │   └── train.yaml                # CPU smoke：4 步、batch=1、cosine、toy_chatml
│   └── production/
│       ├── datasets_zh_first_3b.yaml # 3B SFT 数据 recipe：~145k 对话、5 个 active 源
│       ├── datasets_zh_first_20b.yaml# 20B SFT 数据 recipe：~550k 对话、6 个 active 源
│       ├── train_zh_first_3b.yaml    # 3B SFT 训练配置：lr=2e-5、seq=4096、1 epoch
│       └── train_zh_first_20b.yaml   # 20B SFT 训练配置：lr=1.5e-5、seq=4096、1 epoch
├── scripts/
│   ├── local/
│   │   ├── prepare_data.py           # 生成 4 条 toy 对话 + 随机初始化 checkpoint
│   │   └── train.py                  # 调 runner.main() + validate_local_sft_config
│   └── production/
│       ├── download_data.py          # HF 数据集 → ChatML JSONL（支持 5 种输入格式）
│       └── train.py                  # 调 runner.main() + validate_production_sft_config
└── tests/
    └── test_sft.py                   # 3 个测试类：数据加载、collator mask、flow validation
```

---

## 3. 核心模块详解

### 3.1 `collator.py` — ChatML 渲染 + Loss Mask

`DataCollatorForChatSFT` 将每条对话渲染为 ChatML 格式：

```
<|im_start|>system\n{content}<|im_end|>\n<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n{content}<|im_end|>\n
```

**Loss mask 规则**：

| 部分 | input_ids | labels |
|---|---|---|
| `<\|im_start\|>role\n` (header) | 实际 token | `-100` |
| system/user content | 实际 token | `-100` |
| `<\|im_end\|>` (非 assistant) | 实际 token | `-100` |
| assistant content | 实际 token | **实际 token（参与 loss）** |
| `<\|im_end\|>` (assistant 后) | 实际 token | **实际 token（`train_on_assistant_end=True`）** |
| padding | `pad_token_id` | `-100` |

左对齐 padding 到 batch 内最长序列，上限 `max_seq_length`。

### 3.2 `data_utils.py` — 数据加载与校验

- `SftRecord(messages, source)`：frozen dataclass，每条训练样本
- `SftJsonlDataset`：PyTorch `Dataset`，返回 `{"messages": ..., "source": ...}`
- `load_sft_records(path)`：从 JSONL/JSON 文件加载所有记录
- `_coerce_messages()`：校验 role ∈ {system, user, assistant}、content 非空、至少含一个 assistant 消息
- `split_sft_records()`：确定性 random shuffle + 按 `val_fraction` 切分

### 3.3 `runner.py` — 训练入口

训练流程：

1. 加载 train YAML + model YAML
2. 调用 `validate_fn` 校验配置隔离性
3. 加载 tokenizer（local 用 `ToyChatTokenizer`，production 用 `AutoTokenizer`）
4. `LiteLlmForCausalLM.from_pretrained(pretrained_model_path, config=config)` 加载预训练权重
5. `load_sft_records()` + `split_sft_records()` 加载并切分数据
6. `DataCollatorForChatSFT` 做在线渲染
7. HF `Trainer` 训练，自动断点续训
8. 保存 final model + tokenizer 到 `output_dir/final/`

`SftLoggingCallback`：与预训练的 `RichLoggingCallback` 格式一致，每 10 步打印 tok/s、progress、ETA、elapsed、gpu_mem。

### 3.4 `flow_validation.py` — 隔离校验

| 校验项 | Local SFT | Production SFT |
|---|---|---|
| 设备 | 必须 `use_cpu=true` | 必须 `use_cpu=false` |
| DeepSpeed | 禁止 | 必须配置 |
| Tokenizer | `toy_chatml` 或指定 | 必须指定 `tokenizer_name` |
| 续训 | 禁止 | 必须 `resume_from_last_checkpoint=true` |
| data_dir | `./data/local_smoke/sft/` | `./data/production/sft/` 或 `/root/autodl-fs/sft/` |
| output_dir | `./artifacts/local/sft/` | `./artifacts/production/sft/` |
| pretrained | `./artifacts/local/sft/` | `./artifacts/production/` |
| vocab_size | 无限制 | 必须 ≥ 10000 |

### 3.5 `download_data.py` — 数据下载与归一化

支持 5 种输入格式的自动转换：

| 格式 | 用途 | 转换逻辑 |
|---|---|---|
| `messages` | 原生多轮对话 | 直接提取 `messages` 字段 |
| `alpaca` | instruction-input-output | `user = instruction + "\n\n" + input`，`assistant = output` |
| `prompt_response` | 单轮 QA | `user = prompt`，`assistant = response` |
| `firefly` | 类 alpaca | `user = input`，`assistant = target` |
| `chosen_messages` | preference 数据 | 提取 `chosen` 字段的 messages |

过滤器：`min_chars`、`max_chars`、`min_assistant_chars`。

支持 resume：已有足够 sample 数量的 JSONL 直接跳过。

### 3.6 `toy_tokenizer.py` — 本地 Smoke Tokenizer

- 固定 special token：pad=0, eos=1, im_start=2, im_end=3
- 普通字符编码：`4 + (ord(c) % (vocab_size - 4))`
- 特殊字符串 `<|im_start|>` / `<|im_end|>` 编码为 id 2/3
- 不依赖任何远程资源，不需要网络

---

## 4. 数据配方

### 4.1 3B SFT Recipe（`datasets_zh_first_3b.yaml`）

| # | 数据集 | 格式 | 目标样本 | License |
|---|---|---|---|---|
| 1 | `BAAI/COIG-PC-core` | alpaca | 45,000 | 待审 |
| 2 | `HuggingFaceH4/ultrachat_200k` | messages | 40,000 | MIT |
| 3 | `HuggingFaceTB/smoltalk` | messages | 35,000 | Apache-2.0 |
| 4 | `HuggingFaceH4/ultrafeedback_binarized` | messages | 15,000 | MIT |
| 5 | `ise-uiuc/Magicoder-OSS-Instruct-75K` | prompt_response | 10,000 | MIT |
| **合计** | | | **~145k** | |

禁用项（留作后续）：
- `open-r1/OpenR1-Math-220k`：推理轨迹，首轮不做以避免模型过度学推理格式
- `BAAI/Infinity-Instruct` (0625)：gated 中文数据集，需先同意条款

### 4.2 20B SFT Recipe（`datasets_zh_first_20b.yaml`）

| # | 数据集 | 目标样本 | 说明 |
|---|---|---|---|
| 1 | COIG-PC-core | 160,000 | 扩大中文 alpaca |
| 2 | ultrachat_200k | 100,000 | 英文多轮 |
| 3 | smoltalk | 150,000 | 综合 instruction |
| 4 | ultrafeedback_binarized | 50,000 | 偏好数据 chosen |
| 5 | Magicoder-OSS-Instruct | 50,000 | 代码 |
| 6 | OpenR1-Math-220k | 40,000 | 数学推理（20B 开启） |
| **合计** | | **~550k** | |

---

## 5. 训练超参数

| 参数 | 3B SFT | 20B SFT |
|---|---|---|
| 学习率 | 2e-5 | 1.5e-5 |
| Scheduler | cosine | cosine |
| Warmup ratio | 5% | 3% |
| Batch / device | 1 | 1 |
| Grad accum | 8 | 16 |
| Seq length | 4096 | 4096 |
| Epochs | 1 | 1 |
| bf16 | yes | yes |
| Gradient checkpointing | yes | yes |
| DeepSpeed | ZeRO-2 | ZeRO-2 |
| Eval interval | 250 steps | 500 steps |
| Save interval | 250 steps | 500 steps |
| Save total limit | 5 | 5 |
| Val fraction | 3% | 2% |
| Weight decay | 0.01 | 0.01 |
| train_on_assistant_end | true | true |

---

## 6. 部署流程

### 6.1 本地 Smoke 验证（CPU，秒级）

```bash
uv run python sft/scripts/local/prepare_data.py
uv run python sft/scripts/local/train.py
```

验证：
- 4 条 toy 对话正确加载
- ChatML 渲染 + loss mask 正确
- tiny 随机模型完成 4 步训练不报错
- checkpoint 保存到 `./artifacts/local/sft/`

### 6.2 生产 3B SFT 部署

**前提**：预训练 3B 已跑完，`./artifacts/production/checkpoints_zh_first_3b/final/` 存在。

```bash
# Step 1：下载 SFT 数据
uv run python sft/scripts/production/download_data.py \
  --datasets-config sft/configs/production/datasets_zh_first_3b.yaml

# Step 2：启动训练
uv run deepspeed sft/scripts/production/train.py \
  --train-config sft/configs/production/train_zh_first_3b.yaml \
  --model-config configs/production/model.yaml
```

数据写入 `./data/production/sft/zh_first_v0_3b/`，checkpoint 写入 `./artifacts/production/sft/checkpoints_zh_first_3b/`。

### 6.3 生产 20B SFT 部署

**前提**：预训练 20B 已跑完，`./artifacts/production/checkpoints_zh_first_20b/final/` 存在。

```bash
# Step 1：下载 SFT 数据
uv run python sft/scripts/production/download_data.py \
  --datasets-config sft/configs/production/datasets_zh_first_20b.yaml

# Step 2：启动训练
uv run deepspeed sft/scripts/production/train.py \
  --train-config sft/configs/production/train_zh_first_20b.yaml \
  --model-config configs/production/model.yaml
```

数据写入 `/root/autodl-fs/sft/zh_first_v0_20b/`，checkpoint 写入 `./artifacts/production/sft/checkpoints_zh_first_20b/`。

---

## 7. 测试

```bash
# 预训练测试
uv run python -m unittest tests.test_training_fixes -v

# SFT 测试
uv run python -m unittest sft.tests.test_sft -v
```

SFT 测试覆盖：
- **SftDataUtilsTest**：JSONL 加载/校验、无效行（无 assistant）报错、train-val split 保持总数
- **SftCollatorTest**：user token 被 mask(-100)、assistant token 保留、input_ids 与 labels 形状一致
- **SftFlowValidationTest**：local/production 配置通过校验、production 不接受 local 数据路径

---

## 8. 与预训练的关系

```
预训练 ──→ final checkpoint ──→ SFT 加载 ──→ SFT final checkpoint
  ↑                               ↑
  随机初始化                    pretrained_model_path
  flat .npy token array          ChatML messages JSONL
  全 token 参与 loss             只 assistant 参与 loss
  所有 token 都是 label          loss mask = -100 / real
```

共享的部分：
- `lite_llm/configuration.py`：模型配置
- `lite_llm/modeling.py`：模型实现（MLA、SwiGLU、RoPE、RMSNorm、QK-Norm、KV cache、weight tying）
- `configs/production/model.yaml`：生产模型架构定义
- `configs/production/deepspeed_zero2.json`：DeepSpeed 配置

不共享的部分：
- 数据管线（预训练用 `.npy`，SFT 用 JSONL）
- 训练入口（预训练用 `train_runner.py`，SFT 用 `sft/runner.py`）
- Loss 计算（预训练是 next-token 全 token，SFT 是 assistant-only mask）
- Flow validation（各自独立校验）
