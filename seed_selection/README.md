# seed_selection — 种子 Query 筛选模块

从 SAgoge canonical schema 数据中，筛选约 **1M 种子 query**（900k anneal + 100k high-priority），用于后续 SVG teacher distillation。

需求文档：[`docs/task_specs/svg_seed_selection.md`](../docs/task_specs/svg_seed_selection.md)

---

## 流水线概览

```
canonical JSONL (7.7M)
    ↓ extract        抽取 user instruction + svg_len 元数据
    ↓ clean          Unicode 规范化 + strip，过滤空/过短文本
    ↓ dedup_exact    exact dedup（img2svg 优先覆盖 text2svg）
    ↓ dedup_near     MinHash near dedup，分域阈值
    ↓ svg_filter     按 svg_len 过滤 icon 域最简单 10%
    ↓ embed          Qwen3-Embedding-0.6B，256 维，shard 输出
    ↓ cluster        每 bucket MiniBatchKMeans，记录 cluster_id / centrality
    ↓ sample         按比例采样 → pool_1000k / anneal_pool / high_priority_pool
```

### 三桶策略

| bucket_key | 域 | K（聚类数） | near dedup 阈值 |
|---|---|---|---|
| `stage1_icon` | stage1/icon | 10,000 | 0.8 |
| `stage2_icon` | stage2/icon | 2,000 | 0.8 |
| `stage2_illustration` | stage2/illustration | 3,000 | 0.7 |

img2svg 在 exact dedup 中优先于 text2svg（配置文件中 img2svg 文件排在前面）。

---

## 目录结构

```
seed_selection/
  config.py          # YAML 配置加载，强类型 dataclass
  io_utils.py        # JSONL 读写、ID 生成、domain/source 推断
  extract.py         # Step 1：从 canonical schema 抽取 instruction
  clean.py           # Step 2：最小清洗
  dedup_exact.py     # Step 3：精确去重
  dedup_near.py      # Step 4：MinHash 近似去重
  svg_filter.py      # Step 5：SVG 复杂度过滤
  embed.py           # Step 6：Qwen3-Embedding，shard 输出
  cluster.py         # Step 7：MiniBatchKMeans 聚类
  sample.py          # Step 8：分层采样
  main.py            # CLI 入口
  configs/
    default.yaml     # 默认配置模板
  tests/
    fixtures/        # mock JSONL（各域各 source 样本）
    test_extract.py
    test_clean.py
    test_dedup_exact.py
    test_dedup_near.py
    test_embed.py
    test_cluster.py
    test_sample.py
    test_e2e.py
```

---

## 快速开始

### 安装依赖

```bash
pip install datasketch scikit-learn sentence-transformers transformers pyyaml loguru numpy
```

### 配置

编辑 `seed_selection/configs/default.yaml`，至少确认：

```yaml
input_paths:          # 6 个 JSONL 文件，img2svg 在 text2svg 之前
output_root:          # 输出根目录
embedding:
  model_path:         # Qwen3-Embedding-0.6B 路径
  device: cpu         # cpu | cuda | npu
```

### 运行全流水线

```bash
python -m seed_selection.main \
  --config seed_selection/configs/default.yaml \
  run
```

### 从中断处恢复

```bash
python -m seed_selection.main \
  --config seed_selection/configs/default.yaml \
  run --resume
```

### 只运行指定阶段

```bash
python -m seed_selection.main \
  --config seed_selection/configs/default.yaml \
  run --stage embed
```

### dry-run（CPU 验证，跳过真实 embedding）

```bash
python -m seed_selection.main \
  --config seed_selection/configs/default.yaml \
  run --dry-run --dry-run-n 500
```

dry-run 模式：每个输入文件只读前 `--dry-run-n` 行，embedding 使用零向量，全流程约 1 分钟完成。

### 时间估算

```bash
python -m seed_selection.main \
  --config seed_selection/configs/default.yaml \
  estimate
```

---

## CLI 参数

```
python -m seed_selection.main --config CONFIG COMMAND [options]

命令：
  run         运行流水线
  estimate    打印时间估算（不实际运行）

run 选项：
  --resume            跳过已有输出文件的阶段
  --dry-run           快速验证（零向量 + 限制读取行数）
  --dry-run-n N       dry-run 每文件读取行数（默认 1000）
  --stage STAGE       只运行指定阶段（extract/clean/dedup_exact/...）
```

---

## 输出文件

输出根目录（`output_root`）下生成：

| 文件 | 说明 |
|------|------|
| `instruction_pool_raw.jsonl` | extract 阶段原始输出 |
| `instruction_pool_cleaned.jsonl` | clean 阶段输出 |
| `exact_dedup_kept.jsonl` | 精确去重后 |
| `near_dedup_kept.jsonl` | 近似去重后 |
| `svg_filtered_kept.jsonl` | SVG 复杂度过滤后 |
| `embeddings/shard_XXXX.npz` | embedding shard（id + float32 矩阵）|
| `cluster_assignments.jsonl` | 含 bucket_key / cluster_id / cluster_size / distance_to_centroid |
| `pool_1000k.jsonl` | 总采样池（anneal + high-priority）|
| `anneal_pool.jsonl` | ~900k，覆盖优先 |
| `high_priority_pool.jsonl` | ~100k，各 cluster 最中心样本 |
| `run_stats.json` | 运行参数快照 |

### 中间记录 schema

各阶段 JSONL 的每条记录携带：

```json
{
  "id": "stage1_icon/img2svg/data_000000:42",
  "instruction": "Draw a simple house icon",
  "svg_len": 312,
  "domain": "stage1_icon",
  "source": "img2svg",
  "bucket_key": "stage1_icon",
  "cluster_id": 17,
  "cluster_size": 284,
  "distance_to_centroid": 0.043
}
```

---

## 采样策略

**1000k pool**：

1. 各 bucket 按 `bucket_size` 比例分配 quota，总和 = min(1,000,000, 总记录数)
2. bucket 内各 cluster 按 `sqrt(cluster_size)` 分配 budget（最少 1）
3. cluster 内按 `distance_to_centroid` 升序取前 budget 条

**100k high-priority**：

从 1000k 中，每个 `(bucket_key, cluster_id)` 取 distance 最小的 1 条；不足 100k 时按 distance 补充。

**900k anneal**：pool_1000k 去掉 high-priority 的 id。

---

## 配置参考

```yaml
input_paths:
  - /path/to/stage1/icon/img2svg/data_000000.jsonl
  - /path/to/stage2/icon/img2svg/data_000000.jsonl
  - /path/to/stage2/illustration/img2svg/data_000000.jsonl
  - /path/to/stage1/icon/text2svg/data_000000.jsonl
  - /path/to/stage2/icon/text2svg/data_000000.jsonl
  - /path/to/stage2/illustration/text2svg/data_000000.jsonl

output_root: /path/to/output

svg_filter_bottom_pct: 0.10    # icon 域去除最简单的 10%

embedding:
  model_path: /path/to/Qwen3-Embedding-0.6B
  dimension: 256
  batch_size: 16               # CPU；GPU 建议 256
  device: cpu                  # cpu | cuda | npu
  shard_size: 100000

near_dedup:
  num_perm: 128
  char_ngram: 5
  thresholds:
    stage1_icon: 0.8
    stage2_icon: 0.8
    stage2_illustration: 0.7

clustering:
  k_per_bucket:
    stage1_icon: 10000
    stage2_icon: 2000
    stage2_illustration: 3000
  random_seed: 42
  minibatch_size: 50000

sampling:
  total_pool_size: 1000000
  anneal_pool_size: 900000
  high_priority_pool_size: 100000
  random_seed: 42
```

---

## CPU 时间估算（全量 7.7M）

| 阶段 | 数据量 | 估算时间 |
|------|--------|---------|
| extract | 7.7M 条 | ~10 分钟 |
| clean | ~7.7M | ~3 分钟 |
| dedup_exact | ~7.7M | ~5 分钟 |
| dedup_near | ~4M | ~20–40 分钟（按 domain 分批）|
| svg_filter | ~4M | ~3 分钟 |
| embed | ~3M | **18–22 小时**（CPU）/ 1–2 小时（GPU）|
| cluster | ~3M | ~30 分钟 |
| sample | ~3M | ~5 分钟 |

embed 阶段是瓶颈，GPU/NPU 可将总时间降至 2–3 小时。

---

## 测试

```bash
# 运行全部测试（约 1 秒）
python -m pytest seed_selection/tests/ -q

# 只跑单元测试
python -m pytest seed_selection/tests/ -q -k "not e2e"

# 端到端流水线测试（使用 fixtures mock 数据）
python -m pytest seed_selection/tests/test_e2e.py -v
```

---

## Resume 机制

每个阶段以输出文件是否存在且非空为跳过条件：

| 阶段 | 检查文件 |
|------|---------|
| extract | `instruction_pool_raw.jsonl` |
| clean | `instruction_pool_cleaned.jsonl` |
| dedup_exact | `exact_dedup_kept.jsonl` |
| dedup_near | `near_dedup_kept.jsonl` |
| svg_filter | `svg_filtered_kept.jsonl` |
| embed | `embeddings/shard_*.npz`（按 shard 粒度）|
| cluster | `cluster_assignments.jsonl` |
| sample | `pool_1000k.jsonl` |

embed 阶段支持 shard 级 resume：已存在的 shard 不重新计算。
