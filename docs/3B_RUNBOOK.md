# 3B 预训练 Runbook

从已 tokenized 数据下载到训练完成的完整命令。所有路径均为项目相对路径，无需改动任何配置文件。

---

## 1. 下载 tokenized 数据

默认使用 hf-mirror.com，国内服务器直连：

```bash
python scripts/production/download_tokenized_from_hf.py \
  --hf-repo NoBey/l-llm-3b \
  --output-dir data/production/tokenized_zh_first_3b \
  --workers 8
```

海外服务器加 `--no-mirror` 关闭镜像。

数据约 **12 GB**（~600 个 `.npy` shard，共 3B tokens），下载到 `data/production/tokenized_zh_first_3b/`。

---

## 2. 训练

```bash
uv run deepspeed scripts/production/train.py \
  --train-config configs/production/train_zh_first_3b.yaml
```

### 关键配置（`train_zh_first_3b.yaml`）

| 参数 | 值 |
|---|---|
| data_dir | `data/production/tokenized_zh_first_3b` |
| output_dir | `artifacts/production/checkpoints_zh_first_3b` |
| model | `configs/production/model.yaml`（~1B 参数） |
| seq_len | 8192 |
| batch / gpu | 2 |
| grad_accum | 16 |
| lr | 3e-4, cosine |
| bf16 | true |
| DeepSpeed | ZeRO-2 |
| eval | 每 500 步，0.5% 验证集 |
| save | 每 500 步，保留最近 5 个 |

### 单卡显存参考

| GPU | 显存 | 预估吞吐 | 预估训练时间（1 epoch） |
|---|---|---|---|
| RTX 4090 | 24 GB | ~80-120K tok/s | 7-10 小时 |
| A100 40G | 40 GB | ~120-180K tok/s | 5-7 小时 |
| A100 80G | 80 GB | ~150-200K tok/s | 4-6 小时 |

---

## 3. 配置文件清单

| 文件 | 用途 |
|---|---|
| `configs/production/train_zh_first_3b.yaml` | 3B 训练超参 |
| `configs/production/model.yaml` | 模型架构（~1B 参数） |
| `configs/production/deepspeed_zero2.json` | DeepSpeed ZeRO-2 |

---

## 4. 训练完成后

checkpoint 保存在 `artifacts/production/checkpoints_zh_first_3b/`，最终模型在 `artifacts/production/checkpoints_zh_first_3b/final/`。
