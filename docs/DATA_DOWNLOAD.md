# 预训练数据下载指南

本文档说明如何按中文优先配方下载、tokenize、打包预训练数据。默认配置 `configs/production/datasets.yaml` 是 3B MVP；正式 20B 使用 `configs/production/datasets_zh_first_20b.yaml`。

---

## 1. 数据集概览

当前保留四类 recipe：

| 配置文件 | 目标 |
|---|---|
| `datasets_zh_first_3b.yaml` | 3B 工程验证版 |
| `datasets_zh_first_20b.yaml` | 20B 正式中文优先版（19B active + 1B reserved） |
| `datasets_ablation.yaml` | 小规模消融实验池 |
| `datasets_en_first_legacy.yaml` | 旧英文主导配方，用于对比 |

3B MVP 输出约 **12 GB** tokenized `.npy`，20B 正式版输出约 **80 GB**。

### 3B MVP

| # | 数据集名 | HF 数据源 | 目标 tokens | 预估 shard 数 | 预估 tokenized 磁盘 |
|---|---|---|---:|---:|---:|
| 1 | `zh_fineweb_edu_v21` | `opencsg/Fineweb-Edu-Chinese-V2.1` | 1.4B | ~280 | ~5.6 GB |
| 2 | `baai_cci3_hq` | `BAAI/CCI3-HQ` | 500M | ~100 | ~2 GB |
| 3 | `skypile_150b` | `Skywork/SkyPile-150B` | 300M | ~60 | ~1.2 GB |
| 4 | `zh_cosmopedia` | `opencsg/chinese-cosmopedia` | 200M | ~40 | ~0.8 GB |
| 5 | `en_fineweb_edu` | `HuggingFaceFW/fineweb-edu` | 200M | ~40 | ~0.8 GB |
| 6 | `finemath_4plus` | `HuggingFaceTB/finemath` (`finemath-4plus`) | 250M | ~50 | ~1 GB |
| 7 | `github_code_clean` | `codeparrot/github-code-clean` | 150M | ~30 | ~0.6 GB |
| | **合计** | | **3B** | **~600** | **~12 GB** |

### 20B 正式版（19B active + 1B reserved）

| 数据集名 | 目标 tokens |
|---|---:|
| `zh_fineweb_edu_v21` | 9B |
| `baai_cci3_hq` | 3B |
| `skypile_150b` | 2B |
| `zh_cosmopedia` | 1.2B |
| `en_fineweb_edu` | 1.2B |
| `finemath_4plus` | 1.6B |
| `github_code_clean` | 1B |
| `reserved_experiment` | 1B（默认关闭） |

注意：`BAAI/CCI3-HQ` 是 gated dataset，第一次使用前需要在 HuggingFace 页面登录并同意条款；即使是 `--local-only`，下载它时也建议传 `--hf-token`。

---

## 2. 采样策略

生产脚本不再按 HF 文件名排序后从前往后切 token。现在默认：

- `file_order: shuffled`：每个数据集的文件列表按固定 seed 打乱
- `row_group_order: shuffled`：parquet row group 按固定 seed 打乱
- `row_order: shuffled`：row group 内 row 按固定 seed 打乱
- `json_shuffle_buffer: 4096`：JSON/CSV/JSONL 用固定大小 buffer 做局部打乱

这保证 3B MVP 拿到的是全数据源上的确定性随机样本，而不是文件名靠前的一段。

---

## 3. 磁盘保护

默认磁盘水位写在 `configs/production/datasets.yaml`：

```yaml
disk:
  min_free_gb: 80
  max_cache_gb: 60
```

脚本会在下载和写 shard 前检查剩余空间，低于 `min_free_gb` 会直接停止。HF 原始文件下载到 `{output-dir}/_cache/downloads/`，每处理完一个文件就清理该独立缓存；文件级续传状态写到 `{output-dir}/_cache/state/`。

如果某个上游源文件本身超过 60 GB，脚本会在下载前停止；确认磁盘足够后可以临时调大 `--max-cache-gb`。

---

## 4. 运行命令

```bash
# 安装生产依赖
uv sync --extra production --frozen

# 只看计划：列远端文件数量/大小、目标 tokens、预计 shard 磁盘；不下载数据文件
uv run python scripts/production/prepare_data.py --plan-only

# 3B MVP 第一步：下载/筛选 raw parquet 到 data/production/raw_zh_first_3b
uv run python scripts/production/download_data.py \
  --datasets-config configs/production/datasets_zh_first_3b.yaml \
  --hf-token "$HF_TOKEN" \
  --no-mirror

# 3B MVP 第二步：从本地 raw parquet CPU tokenize
uv run python scripts/production/tokenize_raw_data.py \
  --datasets-config configs/production/datasets_zh_first_3b.yaml \
  --local-only

# 正式 20B 第一步：下载/筛选 raw parquet 到 /root/autodl-fs/raw_zh_first_20b
uv run python scripts/production/download_data.py \
  --datasets-config configs/production/datasets_zh_first_20b.yaml \
  --hf-token "$HF_TOKEN" \
  --no-mirror

# 只跑一部分数据集
uv run python scripts/production/download_data.py \
  --datasets-config configs/production/datasets_zh_first_3b.yaml \
  --datasets zh_fineweb_edu_v21,baai_cci3_hq \
  --hf-token "$HF_TOKEN" \
  --no-mirror

# 消融实验
uv run python scripts/production/download_data.py \
  --datasets-config configs/production/datasets_ablation.yaml \
  --datasets ablate_zh_fineweb_score3 \
  --no-mirror

# 自定义输出目录和磁盘水位
uv run python scripts/production/download_data.py \
  --datasets-config configs/production/datasets_zh_first_3b.yaml \
  --download-dir data/production/raw_zh_first_3b \
  --min-free-gb 80 \
  --max-cache-gb 60
```

上传模式仍然可用：

```bash
export HF_TOKEN=hf_xxxxxxxxxxxx
uv run python scripts/production/tokenize_raw_data.py \
  --datasets-config configs/production/datasets_zh_first_3b.yaml \
  --hf-token "$HF_TOKEN" \
  --hf-repo username/lite-llm-tokenized \
  --hf-path zh_first_v1_3b \
  --keep-uploaded
```

---

## 5. 输出文件结构

```
data/production/raw_zh_first_3b/
├── zh_fineweb_edu_v21/
│   ├── zh_fineweb_edu_v21-00000.parquet
│   └── ...
├── baai_cci3_hq/
│   └── baai_cci3_hq-00000.parquet
├── _cache/downloads/            # 临时 HF 下载缓存，单文件处理后清理
└── _state/                      # raw 下载续传状态

data/production/tokenized_zh_first_3b/
├── zh_fineweb_edu_v21-00000.npy
├── zh_fineweb_edu_v21-00001.npy
├── baai_cci3_hq-00000.npy
├── skypile_150b-00000.npy
├── ...
└── _cache/
    └── state/                   # 文件级续传状态
        ├── zh_fineweb_edu_v21_progress.json
        └── ...
```

每个 `.npy` shard 约 5M tokens / 20 MB；每篇文档末尾追加 `tokenizer.eos_token_id` 作为文档边界。

---

## 6. 训练入口

3B MVP 数据准备完成后，用独立训练配置启动：

```bash
uv run deepspeed scripts/production/train.py \
  --train-config configs/production/train_zh_first_3b.yaml \
  --model-config configs/production/model.yaml
```

20B 正式版对应：

```bash
uv run deepspeed scripts/production/train.py \
  --train-config configs/production/train_zh_first_20b.yaml \
  --model-config configs/production/model.yaml
```

两套训练配置的 `data_dir`、`output_dir`、`logging_dir` 都是独立的，不会互相覆盖。

---

## 7. 打包 .npy 文件

`scripts/production/pack_shards.py` 可以把 `.npy` shard 按约 10GB 一组打包成 `.tar` 归档，支持增量打包。

```bash
python scripts/production/pack_shards.py --dataset zh_fineweb_edu_v21
python scripts/production/pack_shards.py --dataset baai_cci3_hq
python scripts/production/pack_shards.py --dry-run
```

打包状态记录在 `{output-dir}/pack_manifest.json` 中。已打包的文件通过文件名和文件大小识别，重跑时会自动跳过。
