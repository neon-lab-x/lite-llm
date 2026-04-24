# 预训练数据下载指南

本文档说明如何下载、tokenize、打包预训练数据。

---

## 1. 数据集概览

共 6 个数据集，总目标约 **19.8B tokens**。数据集定义在 `configs/production/datasets.yaml` 中。

| # | 数据集名 | HF 数据源 | 过滤规则 | 目标 tokens | 预估 shard 数 | 预估磁盘 |
|---|---|---|---|---|---|---|
| 1 | `fineweb_edu` | `HuggingFaceFW/fineweb-edu` | `int_score >= 2` | 8B | ~1,600 | ~32 GB |
| 2 | `fineweb_general` | `HuggingFaceFW/fineweb` | 无 | 4B | ~800 | ~16 GB |
| 3 | `fineweb_edu_chinese` | `opencsg/Fineweb-Edu-Chinese-V2.1` | 中文字符占比 >= 15%，长度 >= 100 | 3.5B | ~700 | ~14 GB |
| 4 | `finemath_4plus` | `HuggingFaceTB/finemath` (config: finemath-4plus) | 无 | 2B | ~400 | ~8 GB |
| 5 | `finemath_3plus` | `HuggingFaceTB/finemath` (config: finemath-3plus) | 无 | 1.5B | ~300 | ~6 GB |
| 6 | `github_code` | `codeparrot/github-code` | 无，`min_doc_length=20` | 800M | ~160 | ~3.2 GB |
| | **合计** | | | **~19.8B** | **~3,960** | **~79 GB** |

**说明**：

- 每个 shard 固定 5M tokens（`FLUSH_EVERY = 5,000,000`），存为 int32 格式，约 **20 MB/shard**
- `github_code` 来源为 GitHub BigQuery 公开数据，覆盖 30 种编程语言（Python、JavaScript、Java、C++、Go、Rust、TypeScript、C、C#、Ruby、PHP、Shell、SQL 等），共 115M 文件 / 873 GB，远超 800M token 目标
- 实际从 HF 下载的原始 parquet 数据量会大于最终输出，因为部分数据会被过滤掉

---

## 2. 环境准备

```bash
# 安装生产依赖（datasets, deepspeed, huggingface_hub, pyarrow, wandb）
uv sync --extra production --frozen
```

Tokenizer 使用 `Qwen/Qwen3.5-0.8B`（vocab = 248,077），首次运行时自动下载。

---

## 3. 下载脚本

### 3.1 脚本入口

```
scripts/production/prepare_data.py
```

**处理流程**：列出 HF parquet 文件 -> 逐个下载 -> pyarrow 读取 -> 过滤 -> tokenize -> 写 .npy shard -> 删除 parquet -> 处理下一个文件

### 3.2 关键参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--datasets` | 全部 | 逗号分隔的数据集名，只处理指定的数据集 |
| `--local-only` | 否 | 仅保存 .npy 到本地，跳过 HF 上传 |
| `--dry-run` | 否 | 只统计 token 数，不写磁盘、不上传 |
| `--verify` | 否 | 每个数据集只处理 2 个 shard，不做续传（快速验证） |
| `--scale` | `1.0` | 目标 token 数的缩放比例（如 `0.01` = 1%） |
| `--output-dir` | `/root/autodl-fs/tokenized` | .npy shard 输出目录 |
| `--cache-dir` | `{output-dir}/_cache` | 临时 parquet 下载缓存目录 |
| `--hf-token` | 无 | HF API token（上传模式必须，`--local-only` 不需要） |
| `--no-mirror` | 否 | 禁用 hf-mirror.com 镜像，直接从 huggingface.co 下载 |

### 3.3 运行模式

**本地模式**（推荐，仅下载到本地）：

```bash
python scripts/production/prepare_data.py --local-only
```

**上传模式**（下载 + 自动上传到 HF Dataset repo）：

```bash
export HF_TOKEN=hf_xxxxxxxxxxxx
python scripts/production/prepare_data.py --hf-token $HF_TOKEN
```

### 3.4 续传机制

- **Shard 级续传**：每个数据集的 shard 按 `{name}-00000.npy`、`{name}-00001.npy` ... 命名，重跑时自动从最大 shard 索引继续，已达标的数据集直接跳过
- **Parquet 级续传**：记录已处理的 parquet 文件路径到 `{name}_progress.json`，重跑时跳过已处理文件
- 缓存的 parquet 文件在处理完成后自动删除，不会占用额外空间

---

## 4. 按数据集单独下载

使用 `--datasets` 参数指定单个数据集名，`--local-only` 跳过上传：

```bash
# 1. fineweb_edu (8B tokens, ~32 GB)
python scripts/production/prepare_data.py --datasets fineweb_edu --local-only

# 2. fineweb_general (4B tokens, ~16 GB)
python scripts/production/prepare_data.py --datasets fineweb_general --local-only  --no-mirror

# 3. fineweb_edu_chinese (3.5B tokens, ~14 GB)
python scripts/production/prepare_data.py --datasets fineweb_edu_chinese --local-only  --no-mirror

# 4. finemath_4plus (2B tokens, ~8 GB)
python scripts/production/prepare_data.py --datasets finemath_4plus --local-only --no-mirror

# 5. finemath_3plus (1.5B tokens, ~6 GB)
python scripts/production/prepare_data.py --datasets finemath_3plus --local-only --no-mirror

# 6. github_code (800M tokens, ~3.2 GB, 30 种编程语言)
python scripts/production/prepare_data.py --datasets github_code --local-only --no-mirror
```

**试跑**（1% 数据量快速验证流程）：

```bash
python scripts/production/prepare_data.py --local-only --scale 0.01
```

**只跑部分数据集**：

```bash
python scripts/production/prepare_data.py --datasets fineweb_edu,fineweb_general --local-only
```

---

## 5. 打包 .npy 文件

### 5.1 脚本入口

```
scripts/production/pack_shards.py
```

将 .npy shard 文件按约 10GB 一组打包成 `.tar` 归档，支持增量打包（已打包的文件自动跳过）。

### 5.2 关键参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--dataset` | 无 | 只打包指定数据集的文件（按文件名前缀过滤） |
| `--output-dir` | `/root/autodl-fs/tokenized` | .npy 文件所在目录 |
| `--size` | `10g` | 每个归档的目标大小（如 `5g`、`20g`） |
| `--prefix` | `shards_pack` | 归档文件名前缀（用 `--dataset` 时自动设为数据集名） |
| `--dry-run` | 否 | 只显示打包计划，不实际创建归档 |

### 5.3 按数据集单独打包

```bash
# 每个数据集单独打包，归档自动命名为 {数据集名}_0000.tar、{数据集名}_0001.tar ...
python scripts/production/pack_shards.py --dataset fineweb_edu
python scripts/production/pack_shards.py --dataset fineweb_general
python scripts/production/pack_shards.py --dataset fineweb_edu_chinese
python scripts/production/pack_shards.py --dataset finemath_4plus
python scripts/production/pack_shards.py --dataset finemath_3plus
python scripts/production/pack_shards.py --dataset github_code
```

**预览打包计划**（不实际执行）：

```bash
python scripts/production/pack_shards.py --dataset fineweb_edu --dry-run
```

**自定义归档大小**：

```bash
python scripts/production/pack_shards.py --dataset fineweb_edu --size 5g
```

**打包全部数据集**（不指定 `--dataset`，所有 .npy 混合打包）：

```bash
python scripts/production/pack_shards.py
```

### 5.4 增量打包

打包状态记录在 `{output-dir}/pack_manifest.json` 中。已打包的文件通过 `文件名+文件大小` 唯一标识，重跑时自动跳过。归档编号从已有最大编号继续递增。

---

## 6. 输出文件结构

下载 + 打包完成后，输出目录结构如下：

```
/root/autodl-fs/tokenized/
├── fineweb_edu-00000.npy          # 各数据集的 token shard
├── fineweb_edu-00001.npy          # 每个 ~20 MB (5M tokens, int32)
├── fineweb_edu-00002.npy
├── ...
├── fineweb_general-00000.npy
├── fineweb_general-00001.npy
├── ...
├── fineweb_edu_0000.tar           # 打包后的归档（按数据集分组）
├── fineweb_edu_0001.tar
├── fineweb_general_0000.tar
├── ...
├── pack_manifest.json             # 打包状态记录
└── _cache/                        # 临时下载缓存（处理完自动清理）
```

---

## 7. 数据格式说明

- **Token shard**（`.npy`）：int32 扁平数组，每 5M tokens 一个文件，每篇文档末尾追加一个 `tokenizer.eos_token_id`（`<|im_end|>`）作为文档边界
- **Tokenizer**：`Qwen/Qwen3.5-0.8B`，chat 版，`eos_token = <|im_end|>`
- **归档**（`.tar`）：普通 tar 格式（非压缩），内部文件 mtime 固定为 0，保证可复现
