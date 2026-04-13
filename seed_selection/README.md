# seed_selection — 种子 Query 筛选模块

从 SAgoge canonical schema 数据中，筛选约 **1M 种子 query**，按域和质量分六层输出，用于后续 SVG teacher distillation。

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
    ↓ cluster        KMeans + FPS 多样性排序（两桶策略不同）
                       stage1_icon: KMeans(K=100K) + 类内 FPS → fps_round
                       stage2_illustration: 直接全局 FPS(300K) → fps_rank
    ↓ sample         FPS 顺序优先级 → pool_1000k + 六层分域分级文件
```

### 两桶策略

| bucket_key | 域 | Cluster 策略 | near dedup 阈值 | 说明 |
|---|---|---|---|---|
| `stage1_icon` | stage1/icon + stage2/icon | KMeans(K=100K) + 类内 FPS → `fps_round` | 0.8 | stage2/icon 在 exact dedup 中被全量覆盖，两者合并 |
| `stage2_illustration` | stage2/illustration | 直接全局 FPS(n=300K) → `fps_rank` | 0.7 | 0.32M 条规模，全局 FPS ≈ 1.8 分钟 |

> **为什么 stage2_icon 消失了？**
> input_paths 中 stage1/icon 文件排在 stage2/icon 之前，exact dedup 使用 first-come-wins 策略。stage2/icon 与 stage1/icon 指令完全重叠，因此在 exact dedup 阶段被全量丢弃。downstream 阶段不再出现 `stage2_icon` domain。

### 采样配比与六层分级

| 层 | 域 | 数量 | 用途（FPS 优先级）|
|---|---|---|---|
| `pool_1000k` | 全部 | ~1,000,000 | 总采样池（六层合并） |
| `stage1_icon_high` | stage1_icon | 100,000 | FPS round 1，各 cluster 覆盖最广第一条 |
| `stage1_icon_medium` | stage1_icon | 200,000 | FPS round 2–3 |
| `stage1_icon_low` | stage1_icon | 400,000 | FPS round 4–7 |
| `stage2_illustration_high` | stage2_illustration | **50,000** | 全局 FPS rank 1–50K（最多样） |
| `stage2_illustration_medium` | stage2_illustration | **100,000** | 全局 FPS rank 50K–150K |
| `stage2_illustration_low` | stage2_illustration | **150,000** | 全局 FPS rank 150K–300K |

illustration 数据量约占原始数据 17%，但语义复杂度更高、训练价值更大，通过 `bucket_quota_overrides` 手动提升至 300K（占 30%）。

**常见组合示例**：

| 训练集需求 | 组合方式 | 总量 |
|---|---|---|
| SFT 高质量集 | icon_high + illus_high | 150K |
| 扩展 SFT | icon(高+中) + illus(高+中) | 450K |
| 完整训练池 | 六层全部 | 1000K |

### 并行计算设计

四个计算密集阶段做了并行改造：

| 阶段 | 并行粒度 | 说明 |
|------|---------|------|
| extract | 文件级 | 6 个输入文件各开一个 worker，结果按 img2svg-优先顺序 merge |
| dedup_near | MinHash chunk 级 | 见下方「near dedup 两阶段算法」 |
| cluster | bucket 级 | 2 个 bucket 各开一个 worker，embedding 矩阵以 bytes 传递 |
| embed | 设备级 | shard 按卡数 round-robin 分配，每张卡加载独立模型实例 |

并行进程数由 `num_workers`（extract/dedup_near/cluster）和 `embedding.num_devices`（embed）控制。

### near dedup 两阶段算法

MinHash LSH 去重分为两个性质不同的阶段：

```
Phase 1（并行）：计算 MinHash signature
  对每条文本独立计算 char 5-gram → 更新 128 个 permutation hash
  → 输出 hashvalues (uint64 × 128) 序列化为 bytes
  → 拆成 num_workers 个 chunk，ProcessPoolExecutor 并行执行

Phase 2（顺序，必须）：增量 band hash 去重
  将全部 hashvalues bytes 拼成 (N, 128) numpy 矩阵
  for 每条记录 i（按原始顺序）:
      直接从矩阵计算 9 个 band hash key（byteswap + tobytes）
      if 任一 band hash 已在对应桶中:
          丢弃
      else:
          将 9 个 band hash 插入各自桶
          保留此记录
```

**Phase 2 性能关键**：旧版在 Phase 2 中每条记录都构造一次 `MinHash` 对象（~335μs/次），2.8M 条需 19 分钟。新版直接在 numpy 矩阵上计算 band hash，完全绕过对象构造，速度 **150K rec/s**，2.8M 条仅需 **~19 秒**（62× 加速）。

**Phase 2 必须顺序**：判断记录 i 是否重复，需要查询前 i-1 条已插入的记录，有状态依赖，无法并行。

**num_workers 只影响 Phase 1**：`num_workers` 增大只加速 MinHash 计算（Phase 1），对 Phase 2 无影响。Phase 1 在总耗时中的占比与 `num_workers` 成反比——`num_workers` 足够大后，Phase 2 成为新的顺序瓶颈（约 19 秒，无法进一步缩短）。

**结果确定性**：Phase 1 对每条文本的计算结果与 worker 分配无关，Phase 2 始终顺序执行。**任意 `num_workers` 值的输出完全一致**。

**num_workers 推荐值**：物理核数的 50–75%。超过核数无额外收益；设太大反而增加进程启动和 IPC 开销。

### KMeans 三档后端

聚类阶段支持三档后端，通过 YAML 配置切换，优先级：`use_npu` > `use_faiss` > 默认。

| 后端 | 算法类型 | 安装 | 速度（相对） | 适用场景 |
|------|---------|------|------------|---------|
| sklearn MiniBatchKMeans | 近似（Mini-batch） | 已含（无需额外安装） | 1× | CPU 开发/调试 |
| faiss-cpu Lloyd's | **精确** | `pip install faiss-cpu` | **5–15×** | CPU 生产环境 |
| torch_npu / CUDA Lloyd's | **精确** | `pip install torch torch_npu` | **30–100×** | Ascend 910B / NVIDIA GPU |

**使用 faiss-cpu（CPU 精确 KMeans，推荐 CPU 生产）**：
```yaml
clustering:
  use_faiss: true    # 启用 faiss-cpu Lloyd's（精确算法，5–15× 快于 MiniBatchKMeans）
  use_npu: false     # use_npu 优先于 use_faiss，两者同时为 true 时 NPU 生效
```
安装：`pip install faiss-cpu`

> faiss-gpu 仅支持 CUDA，不支持 Ascend NPU，不要安装 faiss-gpu。

**使用 NPU 加速（Ascend 910B 生产）**：
```yaml
clustering:
  use_npu: true
  npu_devices:
    - "npu:0"             # stage1_icon KMeans(K=100K)
    - "npu:1"             # stage2_illustration 全局 FPS（各自独立，可并行）
  npu_chunk_size: 40000   # K=100K：40K×100K×4B=16GB 峰值（64GB HBM 安全）
                          # 旧版 K=12K 时可设 50000（2.4GB/批）
  n_init: 1               # K=100K 时 1 次初始化已足够
```

多卡配置（2 个 bucket 各占一张卡，并行）：
```yaml
clustering:
  use_npu: true
  npu_devices:
    - "npu:0"   # stage1_icon KMeans → npu:0（stage2 FPS 不使用 KMeans，直接 FPS）
    - "npu:1"   # stage2_illustration 全局 FPS → npu:1
```

8 卡节点（仍是 2 个 bucket，自动 round-robin，npu:0/npu:1 各跑一个 bucket）：
```yaml
clustering:
  use_npu: true
  npu_devices: ["npu:0","npu:1","npu:2","npu:3","npu:4","npu:5","npu:6","npu:7"]
  # 实际只用到 npu:0、npu:1，其余卡闲置；embed 阶段可同时用 num_devices: 8
```

**使用 CUDA GPU**：
```yaml
clustering:
  use_npu: true          # 复用同一后端（torch.cdist 支持 cuda/npu）
  npu_devices:
    - "cuda:0"
```

---

## 目录结构

```
seed_selection/
  config.py          # YAML 配置加载，强类型 dataclass
  io_utils.py        # JSONL 读写、ID 生成、domain/source 推断（两桶）
  extract.py         # Step 1：从 canonical schema 抽取 instruction
  clean.py           # Step 2：最小清洗
  dedup_exact.py     # Step 3：精确去重
  dedup_near.py      # Step 4：MinHash 近似去重
  svg_filter.py      # Step 5：SVG 复杂度过滤
  embed.py           # Step 6：Qwen3-Embedding，shard 输出
  cluster.py         # Step 7：KMeans + FPS（两桶策略）
                     #   stage1_icon:  KMeans(K=100K) + 类内 FPS → fps_round
                     #   stage2_illus: 直接全局 FPS(300K) → fps_rank
  kmeans_npu.py      # NPU/GPU 加速 KMeans（标准 Lloyd's，torch_npu/torch.cuda）
  kmeans_faiss.py    # faiss-cpu 精确 KMeans（CPU BLAS 加速）
  sample.py          # Step 8：FPS 优先级分层采样（pool_1000k + 六层分域分级）
  analyze.py         # 质量报告生成（六层分级分析、层间 distance 单调性验证）
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

# faiss-cpu（可选，CPU 精确 KMeans，5–15× 快于 MiniBatchKMeans）
pip install faiss-cpu

# NPU 加速（可选，需要 Ascend 环境）
pip install torch torch_npu
# CUDA GPU 只需 torch（无需 torch_npu）
pip install torch
```

### 配置

编辑 `seed_selection/configs/default.yaml`，至少确认：

```yaml
input_paths:          # 6 个 JSONL 文件，img2svg 在 text2svg 之前
output_root:          # 输出根目录
embedding:
  model_path:         # Qwen3-Embedding-0.6B 或 Qwen3-Embedding-4B 路径
  device: cpu         # cpu | cuda | npu

# 聚类后端（三选一，优先级：use_npu > use_faiss > 默认）
clustering:
  use_npu: false      # true = NPU/CUDA 精确 KMeans（需要 torch + torch_npu）
  use_faiss: false    # true = faiss-cpu 精确 KMeans（需要 pip install faiss-cpu）
  npu_devices:        # use_npu=true 时生效；单卡填 ["npu:0"]，双卡填 ["npu:0","npu:1"]
    - "npu:0"

# 采样配比
sampling:
  bucket_quota_overrides:
    stage2_illustration: 300000     # illustration 提权到 30%
  tier_sizes:
    stage1_icon:         [100000, 200000, 400000]   # 高100K + 中200K + 低400K
    stage2_illustration: [100000, 100000, 100000]   # 高100K + 中100K + 低100K
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
  analyze     生成质量报告和可视化图表

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
| `pool_1000k.jsonl` | 总采样池（六层合并，~1M）|
| `stage1_icon_high.jsonl` | stage1_icon 高优（100K）|
| `stage1_icon_medium.jsonl` | stage1_icon 中优（200K）|
| `stage1_icon_low.jsonl` | stage1_icon 低优（400K）|
| `stage2_illustration_high.jsonl` | stage2_illustration 高优（100K）|
| `stage2_illustration_medium.jsonl` | stage2_illustration 中优（100K）|
| `stage2_illustration_low.jsonl` | stage2_illustration 低优（100K）|
| `run_stats.json` | 运行参数快照 |

### 中间记录 schema

各阶段 JSONL 的每条记录结构：`instruction` 为唯一顶层 payload 字段，所有流水线元数据统一放在 `_meta` 下。

```json
{
  "instruction": "Draw a simple house icon",
  "_meta": {
    "id": "stage1_icon/img2svg/data_000000:42",
    "domain": "stage1_icon",
    "source": "img2svg",
    "svg_len": 312,
    "bucket_key": "stage1_icon",
    "cluster_id": 17,
    "cluster_size": 284,
    "distance_to_centroid": 0.043
  }
}
```

各字段在流水线中逐步追加：`id/domain/source/svg_len` 由 extract 写入，`bucket_key` 由 cluster 写入，`cluster_id/cluster_size/distance_to_centroid` 由 cluster 写入。

---

## 采样策略详解

### pool_1000k 三层配额分配

**第一层：bucket 间配额**

默认按数据量比例分配（约 icon:illustration = 83:17）；通过 `bucket_quota_overrides` 可手动覆盖：

```yaml
sampling:
  bucket_quota_overrides:
    stage2_illustration: 300000  # 300K，比例提升到 30%
    # stage1_icon 自动获得剩余 700K
```

**第二层：cluster 间按 √(size) 分配 budget（每个至少 1）**

**第三层：cluster 内按 distance_to_centroid 升序选 top-k**

### 六层分级（分域三阶段 Round-Robin）

对 `pool_1000k` 中每个 bucket 独立执行：

```
每个 bucket 独立：
  cluster 内按 distance_to_centroid 升序排列

  Round-Robin 循环遍历所有 cluster，连续前进同一组指针：
    阶段 1：取前 tier_sizes[bucket][0] 条 → 高优（high）
    阶段 2：取下一批 tier_sizes[bucket][1] 条 → 中优（medium）
    阶段 3：剩余所有记录 → 低优（low）
```

- 各 cluster 贡献条数均衡（不因 distance 绝对值倾斜）
- 层间质量单调：高优平均 `distance_to_centroid` < 中优 < 低优
- 指针跨阶段连续：高优选完后中优从高优末尾继续，而非重新排序

**阶段覆盖深度估算（stage1_icon，12K clusters）**：

| 层 | 数量 | 平均轮次/cluster |
|---|---|---|
| 高优 | 100K | ~8 轮 |
| 中优 | 200K | ~17 轮 |
| 低优 | 400K | 剩余 |

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

num_workers: 4                 # extract / dedup_near / cluster 的并行进程数
                               # 建议设为物理 CPU 核数的 50–75%

svg_filter_bottom_pct: 0.10    # icon 域去除最简单的 10%

embedding:
  model_path: /path/to/Qwen3-Embedding-0.6B
  dimension: 256
  batch_size: 16               # CPU；GPU 建议 256
  device: cpu                  # cpu | cuda | npu
  shard_size: 100000
  num_devices: 1               # GPU/NPU 卡数；8 卡 NPU 节点设为 8
                               # CPU 模式固定单进程（忽略此值）

near_dedup:
  num_perm: 128
  char_ngram: 5
  thresholds:
    stage1_icon: 0.8
    stage2_illustration: 0.7

clustering:
  k_per_bucket:
    stage1_icon: 100000         # ~1.44M 记录，平均 ~14 条/cluster
    stage2_illustration: 0      # stage2 不做 KMeans，直接全局 FPS
  random_seed: 42
  minibatch_size: 50000         # MiniBatchKMeans 每批大小（faiss/NPU 模式不生效）
  use_npu: false                # true = torch_npu / CUDA 精确 Lloyd's KMeans
  use_faiss: false              # true = faiss-cpu 精确 Lloyd's（use_npu=true 时忽略）
  npu_devices:                  # use_npu=true 时生效，按 bucket 顺序 round-robin 分配
    - "npu:0"                   # 单卡；多卡示例：["npu:0","npu:1"]
  npu_chunk_size: 40000         # K=100K：40K×100K×4B=16GB 峰值（64GB HBM 安全）

sampling:
  total_pool_size: 1000000
  random_seed: 42
  bucket_quota_overrides:
    stage2_illustration: 300000  # 手动提权到 30%
  tier_sizes:
    stage1_icon:         [100000, 200000, 400000]   # 高100K + 中200K + 低400K = 700K
    stage2_illustration: [100000, 100000, 100000]   # 高100K + 中100K + 低100K = 300K
```

---

## CPU 时间估算（全量 7.7M）

以下估算基于 `num_workers=4`：

| 阶段 | 数据量 | 单进程 | num_workers=4 |
|------|--------|--------|---------------|
| extract | 7.7M 条 | ~10 分钟 | ~3 分钟 |
| clean | ~7.7M | ~3 分钟 | ~3 分钟（无并行）|
| dedup_exact | ~7.7M | ~5 分钟 | ~5 分钟（无并行）|
| dedup_near | ~2.35M | ~80–100 分钟 | **~2–5 分钟**（主要是 Phase 1）|
| svg_filter | ~1.92M | ~3 分钟 | ~3 分钟（无并行）|
| embed | ~1.76M | **12–15 小时**（CPU）| ~40 分钟（GPU ×1）/ ~10 分钟（NPU ×8）|
| cluster | ~1.76M | ~20 分钟（CPU）| ~8 分钟（2 桶并行） / **~2 分钟**（NPU）|
| sample | ~1.76M | ~5 分钟 | ~5 分钟（无并行）|

embed 是最大瓶颈：CPU 不可行（18–22 小时），强烈建议 GPU/NPU。8 卡 NPU 时设 `num_devices: 8` 可将 embed 缩至 ~15 分钟，全流程约 1 小时。

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

---

## 质量分析（analyze 子命令）

```bash
python -m seed_selection.main \
  --config seed_selection/configs/default.yaml \
  analyze
```

分析结果写入 `{output_root}/analysis/`：

| 文件 | 内容 |
|------|------|
| `report.txt` | 文字报告：漏斗统计、六层计数、cluster 覆盖率、层间 distance 分离、source mix |
| `metrics.json` | 所有指标的 JSON 快照 |
| `01_funnel.png` | 各流水线阶段记录数瀑布图 |
| `02_bucket_dist.png` | 每个 bucket 三层分级柱状图（高/中/低优 grouped bars）|
| `03_cluster_size_hist.png` | 各 bucket cluster 大小分布直方图 |
| `04_instruction_len.png` | instruction 长度分布（每个 bucket 一图，三层叠加）|
| `05_distance_hist.png` | distance_to_centroid 分布（每个 bucket 一图，三层叠加，高优峰值应在最左）|
| `06_source_mix.png` | img2svg vs text2svg 比例饼图（pool_1000k）|
| `07_umap.png` | embeddings UMAP 投影（可选，需 `pip install umap-learn`）|

### 核心质量指标说明

**六层不变量**：六层记录数之和应等于 `pool_1000k` 总条数。

**层间 distance 单调性**：对每个 bucket，应满足：

```
高优 mean(distance) < 中优 mean(distance) < 低优 mean(distance)
```

report.txt 会自动标记 `✓ 正常` 或 `⚠ 异常`。

**05_distance_hist.png 解读**：

```
高优（红）峰值 ←── 中优（蓝）峰值 ←── 低优（绿）峰值
```

若三峰叠在一起，说明分层无实质质量区分（通常不会出现，因为 Round-Robin 指针连续前进保证了分离）。

### analyze 所需文件

analyze 模式读取 sample 阶段的输出，所需文件：

```
{output_root}/
  pool_1000k.jsonl
  stage1_icon_high.jsonl
  stage1_icon_medium.jsonl
  stage1_icon_low.jsonl
  stage2_illustration_high.jsonl
  stage2_illustration_medium.jsonl
  stage2_illustration_low.jsonl
  cluster_assignments.jsonl          # 用于 cluster 覆盖率统计
  embeddings/                        # 可选，仅 07_umap.png 使用
```

缺少某个文件时对应指标/图表会被跳过，不影响其他分析。
