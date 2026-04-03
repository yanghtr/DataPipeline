# Seed Query 筛选算法技术报告

## 目录

1. [背景与目标](#1-背景与目标)
2. [数据来源与结构](#2-数据来源与结构)
3. [整体流水线](#3-整体流水线)
4. [Step 1：提取（Extract）](#4-step-1提取-extract)
5. [Step 2：清洗（Clean）](#5-step-2清洗-clean)
6. [Step 3：精确去重（Exact Dedup）](#6-step-3精确去重-exact-dedup)
7. [Step 4：近似去重（Near Dedup）](#7-step-4近似去重-near-dedup)
8. [Step 5：SVG 复杂度过滤（SVG Filter）](#8-step-5svg-复杂度过滤-svg-filter)
9. [Step 6：向量化（Embed）](#9-step-6向量化-embed)
10. [Step 7：聚类（Cluster）](#10-step-7聚类-cluster)
11. [Step 8：采样（Sample）](#11-step-8采样-sample)
12. [数据漏斗与规模估算](#12-数据漏斗与规模估算)
13. [超参汇总与选择依据](#13-超参汇总与选择依据)
14. [工程设计要点](#14-工程设计要点)

---

## 1. 背景与目标

本流水线的目标是从 SAgoge 数据集的 SVG 生成数据中，筛选出一批高质量、多样性强的 **文本指令（instruction）**，作为模型蒸馏阶段的种子 query。

核心设计取向：
- **多样性优先**：避免语义重复的指令大量堆积，通过聚类和基于多样性的采样覆盖尽可能广的指令空间
- **质量保底**：过滤过短指令、过于简单的 SVG 对应的指令、以及语义上几乎相同的近似重复
- **来源优先级**：图片驱动的 img2svg 指令通常语义更具体，优先于文字驱动的 text2svg 指令
- **规模与配比**：最终输出约 100 万条采样池，其中 20 万条为"高质量优先"集合（大模型蒸馏），80 万条为退火集合（开源模型蒸馏）；illustration 数据手动提权（占 30%）以补偿其数量劣势

---

## 2. 数据来源与结构

### 2.1 原始数据分布

原始数据来自 SAgoge 数据集，按两个维度划分：

| 维度 | 类别 | 含义 |
|------|------|------|
| **Domain（任务类型）** | `stage1_icon` | Stage 1 阶段的图标生成数据 |
| | `stage2_icon` | Stage 2 阶段的图标生成数据（更复杂） |
| | `stage2_illustration` | Stage 2 阶段的插图生成数据（最复杂） |
| **Source（指令来源）** | `img2svg` | 从真实图片反推生成的指令 |
| | `text2svg` | 纯文字描述生成的指令 |

共 6 个输入文件，每个 `(domain, source)` 组合对应一个 JSONL 文件。

**有效 domain（精确去重后）**：stage2_icon 的所有 instruction 与 stage1_icon 完全重叠（见 §6.3），exact dedup 后仅剩两个有效 domain：

| Domain | 估算记录数（过滤后） | 占比 |
|--------|-------------------|------|
| `stage1_icon` | ~2.25M | ~83% |
| `stage2_illustration` | ~0.45M | ~17% |

### 2.2 原始数据格式

原始数据为 panguml 统一格式，每条记录包含一次对话：

```json
{
  "meta_prompt": [""],
  "data": [
    {
      "role": "user",
      "content": [{"type": "text", "text": {"string": "Draw a simple house icon"}}]
    },
    {
      "role": "assistant",
      "content": [{"type": "text", "text": {"string": "```svg\n<svg ...>...</svg>\n```"}}]
    }
  ]
}
```

### 2.3 SVG 长度特征

不同 domain 的 SVG 复杂度差异显著：

| Domain | SVG 字符数范围 | 指令长度（中位数） | 特征 |
|--------|--------------|-----------------|------|
| stage1_icon | 101–271 字符 | ~79 字符 | 极简，多为基础几何形状 |
| stage2_icon | ~100–2000 字符 | ~79 字符 | 与 stage1_icon 完全重叠 |
| stage2_illustration | 3,605–29,191 字符 | ~408 字符 | 高复杂度，包含大量路径 |

SVG 字符数作为 **复杂度代理指标**，是后续过滤阶段的关键依据。

---

## 3. 整体流水线

```
原始 JSONL（6 个文件）
        │
        ▼
  ① Extract          提取 instruction 文本和基础元数据
        │
        ▼
  ② Clean            Unicode 规范化 + 过短过滤
        │
        ▼
  ③ Exact Dedup      字符串精确去重，img2svg 优先
        │             （stage2_icon 在此阶段被全量覆盖）
        ▼
  ④ Near Dedup       MinHash LSH 近似去重（按 domain 分组）
        │
        ▼
  ⑤ SVG Filter       移除过度简单的 icon SVG（底部 10%）
        │
        ▼
  ⑥ Embed            Qwen3-Embedding-4B，256 维 L2 归一化向量
        │
        ▼
  ⑦ Cluster          KMeans（两桶）
        │             stage1_icon: K=12,000
        │             stage2_illustration: K=6,000
        ▼
  ⑧ Sample           两层配额 + 两类输出池
        │
  ┌─────┼──────────────┐
  ▼     ▼              ▼
pool_1000k   high_priority_pool   anneal_pool
（~100万）      （20万）               （~80万）
```

**中间记录 schema** — 各阶段逐步追加字段，顶层只有 `instruction`，所有流水线元数据统一存入 `_meta`：

```json
{
  "instruction": "Draw a simple house icon",
  "_meta": {
    "id":                  "stage1_icon/img2svg/data_000000:42",
    "domain":              "stage1_icon",
    "source":              "img2svg",
    "svg_len":             271,
    "gt_svg":              "```svg\n<svg viewBox=\"0 0 24 24\">...</svg>\n```",
    "bucket_key":          "stage1_icon",      ← cluster 阶段写入
    "cluster_id":          17,                  ← cluster 阶段写入
    "cluster_size":        284,                 ← cluster 阶段写入
    "distance_to_centroid": 0.043              ← cluster 阶段写入
  }
}
```

`gt_svg` 字段存储原始 GT SVG 文本（含 ` ```svg ` 包裹符），贯穿所有阶段保留至最终输出文件，供蒸馏完成后与模型生成的 SVG 做并排对比可视化。

---

## 4. Step 1：提取（Extract）

### 4.1 功能

从 6 个原始 JSONL 文件中，提取每条记录的 `instruction` 文本和 SVG 字符数，构建统一的中间格式。

### 4.2 算法逻辑

**字段提取**：
- `instruction`：取 `data[0]`（user turn）中第一个 `type=="text"` 的 `text.string`
- `svg_len`：取 `data[1]`（assistant turn）中第一个 `type=="text"` 的 `text.string` 的字符数（含 ` ```svg ` 包裹符）
- `gt_svg`：同上 `text.string` 的完整内容（含 ` ```svg ` 包裹符），存入 `_meta.gt_svg`，贯穿全流程保留至最终采样输出，供蒸馏后与模型生成 SVG 并排对比

**ID 生成**：每条记录分配全局唯一 ID：

```
id = "{domain}/{source}/{file_stem}:{line_no}"
例如：stage1_icon/img2svg/data_000000:42
```

设计关键：id 中包含 domain + source + 文件名，确保不同文件中相同行号的记录不会产生 id 碰撞（6 个文件均命名为 `data_000000.jsonl`，若仅用文件名会产生全局 id 重复）。

**文件顺序**：配置中 img2svg 文件**严格排在** text2svg 之前：

```yaml
input_paths:
  - .../stage1/icon/img2svg/data_000000.jsonl   # img2svg 优先
  - .../stage2/icon/img2svg/data_000000.jsonl
  - .../stage2/illustration/img2svg/data_000000.jsonl
  - .../stage1/icon/text2svg/data_000000.jsonl  # text2svg 在后
  - .../stage2/icon/text2svg/data_000000.jsonl
  - .../stage2/illustration/text2svg/data_000000.jsonl
```

这个顺序由后续精确去重中的 first-come-wins 策略所依赖。

**并行**：6 个文件并行提取（`ProcessPoolExecutor`），merge 时严格按原始顺序拼接，保证 img2svg 在前的全局顺序。

### 4.3 过滤条件

| 过滤条件 | 处理 |
|---------|------|
| user turn 无文本 | skip |
| assistant turn 无 SVG | skip |
| JSON 解析失败 | skip |

---

## 5. Step 2：清洗（Clean）

### 5.1 功能

对 instruction 文本做轻量化规范化，过滤无效记录。

### 5.2 算法逻辑

**Unicode NFC 规范化**：

```python
unicodedata.normalize("NFC", text).strip()
```

NFC 将分解的 Unicode 组合字符（如 `é` = `e` + `´`）合并为预组合形式，消除因编码差异导致的字符串不等价问题，提高后续精确去重的召回率。

**长度过滤**：清洗后字符数 `< 3` 的记录被丢弃（`MIN_INSTRUCTION_LEN = 3`）。此阈值极宽松，主要过滤因解析错误产生的空字符串或单字母残留。

---

## 6. Step 3：精确去重（Exact Dedup）

### 6.1 功能

基于完全相同的 instruction 字符串进行去重，同一指令出现多次时保留质量最优的版本。

### 6.2 算法逻辑

**单遍扫描 + 优先级替换**：

```python
seen: dict[str, (priority, record)] = {}

for rec in records:
    instr = rec["instruction"]
    priority = SOURCE_PRIORITY[rec["_meta"]["source"]]
    # img2svg: priority=0（优先）
    # text2svg: priority=1

    if instr not in seen:
        seen[instr] = (priority, rec)
    elif priority < seen[instr][0]:
        seen[instr] = (priority, rec)  # img2svg 替换 text2svg
    # 相同 source 时 first-come-wins（即靠前的行保留）
```

**输出顺序**：Python 3.7+ 的 `dict` 保持插入顺序，因此输出中 img2svg 记录在前的全局顺序得以保持。

### 6.3 stage2_icon 被全量覆盖的原因

这是本流水线的关键观察：**stage2_icon 的所有 instruction 都在 stage1_icon 中出现过**。

原因：
1. stage1/icon 的 img2svg 文件（2,791,284 条）排在 stage2/icon 的 img2svg 文件（555,000 条）之前
2. exact dedup 使用 first-come-wins：stage1_icon 记录先被处理，占据了 dict 中的位置
3. stage2_icon 的指令集是 stage1_icon 的子集（相同的图标生成任务，更复杂的 SVG 对应相同 instruction）
4. 因此 stage2_icon 的全部 555,000 条记录在 exact dedup 后被丢弃

**下游影响**：
- near_dedup、svg_filter、cluster、sample 阶段均不再出现 `stage2_icon` domain
- 有效 domain 从三桶变为两桶：`stage1_icon` + `stage2_illustration`
- `infer_domain()` 将 stage2/icon 路径也映射到 `"stage1_icon"`

### 6.4 img2svg 与 text2svg 的重叠比例

这是一个容易被忽视的关键现象：**img2svg 与 text2svg 的 instruction 重叠率接近 100%**。

证据：原始文件行数几乎 1:1：

| 文件 | 行数 |
|------|------|
| stage1/icon/img2svg | 2,791,284 |
| stage1/icon/text2svg | 2,791,323 |
| stage2/illustration/img2svg | 517,788 |
| stage2/illustration/text2svg | 518,792 |

两种来源使用**相同的底层 SVG 图像**：img2svg 从图像反推 instruction，text2svg 从文字描述生成 instruction，由于底层图像相同，大量 instruction 文本完全一致。

**对下游的影响**：
- exact dedup 后，text2svg 几乎全部被 img2svg 覆盖（img2svg 优先）
- 有效记录集 ≈ img2svg 记录集（text2svg 仅作为潜在备用，在 img2svg 记录损坏/缺失时生效）
- 实际进入 embed/cluster/sample 的记录全部唯一，无重复问题

**注**：stage2_icon 被 stage1_icon 全量覆盖（§6.3）是另一维度的重叠（跨 stage），与 img2svg/text2svg 的同 domain 内重叠属于两个独立现象，均通过 exact dedup 的 first-come-wins 解决。

### 6.5 设计原理

**为什么 img2svg 优先？**
img2svg 的指令是从真实图片反推出来的，语义上与具体视觉内容更紧密绑定，信息密度更高。text2svg 是纯文字创作，质量参差不齐。当两条指令文本完全相同时，保留 img2svg 版本确保后续蒸馏时，模型看到的是与真实图片对应的 SVG，而非纯文字生成的 SVG。

---

## 7. Step 4：近似去重（Near Dedup）

### 7.1 功能

基于 MinHash LSH 去除语义上高度相似（但不完全相同）的 instruction，按 domain 分组，不同 domain 使用不同相似度阈值。

### 7.2 算法逻辑

**两阶段设计**：

```
Phase 1（并行）：MinHash 计算
    对每条文本：
        计算字符 5-gram 序列
        用 128 个独立 hash 函数映射 → MinHash signature（128 × uint64）
        序列化为 bytes 传回主进程

    → 将所有 bytes 拼成 (N, 128) uint64 矩阵

Phase 2（顺序，有状态依赖）：增量 LSH 去重
    从 dummy MinHashLSH 获取 band 分割参数（threshold=T, num_perm=128）
    → 9 个 band，每个 band 13 列

    for 每条记录 i（按原始顺序，即 img2svg 在前）:
        keys[b] = hv_matrix[i, start_b:end_b].byteswap().tobytes()  # 9 个 band hash
        if 任一 band hash 已在对应桶中:
            丢弃（找到近似重复）
        else:
            将 9 个 band hash 插入各自桶
            保留此记录
```

**为什么不并行 Phase 2？**
LSH 去重需要查询「之前所有已保留记录」的 band hash，是有状态的增量过程，任何并行化都会漏掉跨分片的近似重复对。

**img2svg 优先性的传递**：Phase 2 按顺序处理，img2svg 记录排在前面，因此当 img2svg 和 text2svg 互为近似重复时，img2svg 先被插入桶，text2svg 后续被判定为重复而丢弃。

### 7.3 关键超参

#### 字符级 n-gram（`char_ngram=5`）

MinHash 的输入是文本的字符 5-gram 集合。对于短文本（如 icon 指令，通常 10–50 个词）：
- 字符级 n-gram 比词级 n-gram 细粒度更高，对语序变化不敏感（"red circle" vs "circle red" 有大量共享 5-gram）
- 5-gram 是短文本去重的经验最优：太短（2-gram）区分度弱，太长（8-gram）召回率低

#### MinHash 排列数（`num_perm=128`）

MinHash 估计 Jaccard 相似度的方差：

$$\text{Var}(\hat{J}) = \frac{J(1-J)}{m}$$

其中 $m$ 为排列数。$m=128$ 时，$J=0.8$ 处的标准差约为 $0.035$，估计误差在可接受范围内，同时内存占用 $128 \times 8B \times 3M \approx 3\text{ GB}$ 可控。

#### 相似度阈值（`threshold`）

| Domain | 阈值 | 原因 |
|--------|------|------|
| `stage1_icon` | **0.8** | 短文本（10–30 字符），字符 5-gram 重叠度高，需严格阈值 |
| `stage2_illustration` | **0.7** | 长文本天然 Jaccard 值较低，0.8 会漏掉实质重复 |

阈值 $T$ 与 LSH band 参数的关系：

$$P(\text{至少一个 band 匹配}) \approx 1 - (1 - J^r)^b$$

其中 $r$ 为每 band 的行数，$b$ 为 band 数（$b \times r \leq 128$）。$T=0.8$ 对应 datasketch 自动选择 $b=9, r=13$，在 $J=0.8$ 处命中概率 $\approx 0.99$，$J=0.7$ 处约 $0.50$（较好区分）。

### 7.4 性能优化

Phase 2 的原始实现在每条记录上构造 `MinHash()` 对象（~335μs/次），2.8M 条约 19 分钟。

优化方案：将全部 hashvalues 预先拼成 numpy 矩阵，直接在矩阵行上进行 band hash 计算，完全绕过 MinHash 对象构造：

```python
keys = [hv[s:e].byteswap().tobytes() for s, e in hashranges]
```

优化后 Phase 2 速度约 150K rec/s，2.8M 条仅需 **~19 秒**（62× 加速）。

---

## 8. Step 5：SVG 复杂度过滤（SVG Filter）

### 8.1 功能

去除 `stage1_icon` 中对应 SVG 过于简单的指令（底部 10%）。`stage2_illustration` 全部保留。

### 8.2 算法逻辑

**两遍算法**：

```
第一遍：收集各 domain 的 svg_len 分布
        → 计算 10th percentile 作为截断值

第二遍：重新读文件
        对 stage1_icon：移除 svg_len <= cutoff 的记录
        对 stage2_illustration：全部保留
```

过滤条件：`svg_len <= P10(domain)`，即严格小于等于（boundary 值也被移除）。

### 8.3 设计原理

**为什么用 svg_len 作为复杂度代理指标？**

SVG 文本长度与图形复杂度高度正相关：极简 SVG（如单个矩形）字符数仅百余个，而复杂图标可达数千字符。过短的 SVG 意味着：
1. 指令对应的视觉内容过于简单（"Draw a dot"），对蒸馏模型学习 SVG 结构价值低
2. 这类样本在训练中可能形成捷径，导致模型输出过于简单

**为什么 illustration 不过滤？**

插图数据量本就稀少（相对于 icon），且其指令描述场景更复杂，即使对应 SVG 字符数处于低位，也不代表指令质量差。

**为什么是 10%（`bottom_pct=0.10`）？**

基于对 stage1_icon 数据分布的观察：最底部 10% 的 SVG 长度集中在约 101–120 字符，主要为单一几何形状。该比例保守，仅移除明显过度简单的样本，不影响数据分布主体。

---

## 9. Step 6：向量化（Embed）

### 9.1 功能

用预训练语言模型对每条 instruction 生成语义向量，作为后续聚类的输入。

### 9.2 模型选择：Qwen3-Embedding 系列

| 属性 | 值 |
|------|-----|
| 推荐模型 | `Qwen3-Embedding-4B`（生产）/ `Qwen3-Embedding-0.6B`（快速验证） |
| 输出维度 | 原始 ≥ 256，截断到 **256 维** |
| 归一化 | L2 归一化（便于余弦相似度 = 点积） |
| 量化 | float32 |

**为什么截断到 256 维？**
Qwen3-Embedding 全系列支持 **Matryoshka Representation Learning**，可将输出截断到任意低维（128/256/512/...）并重新 L2 归一化，低维向量是高维向量的前缀子集，语义信息已充分保留。256 维在语义区分能力与 KMeans 计算效率（N×K 距离矩阵大小）之间取得平衡。截断后重新归一化以保证向量模长一致。

**为什么从 0.6B 升级到 4B？**
- 0.6B 适合快速验证（CPU 可运行，约 30–60 samples/s）
- 4B 语义质量更高，尤其对 illustration 的长文本（中位数 408 字符）区分效果更好
- 两者均可截断到 256 维，**配置无需修改**（只需更换 `embedding.model_path`）
- 4B 推理速度约为 0.6B 的 1/7，NPU × 8 卡下 embed 阶段仍可在合理时间内完成

### 9.3 算法逻辑

**Shard 机制**：

```
总记录数 N → 切分为若干 shard，每 shard 100,000 条
每个 shard 输出一个 .npz 文件（ids + embeddings 矩阵）
```

**断点续传**：启动时检查 shard 文件是否已存在且非空，若是则跳过，支持中断后恢复。

**多卡并行**（GPU/NPU 环境）：

```
shard 0 → device:0
shard 1 → device:1
shard 2 → device:2
...（round-robin 分配）
```

每个 worker 独立加载模型到指定设备，互不干扰。CPU 模式固定单进程（CPU 内部已有 OpenMP 并行）。

**归一化流程**：

```python
embs = embs[:, :dimension]           # 截断到 256 维
norms = L2_norm(embs, axis=1)
embs = embs / clip(norms, 1e-8, ∞)  # 重归一化，防止零向量除法错误
```

---

## 10. Step 7：聚类（Cluster）

### 10.1 功能

对每个 domain 的向量空间独立做 K-Means 聚类，将每条指令归入某个语义簇，并记录到质心距离，作为采样阶段的质量代理指标。

### 10.2 算法选择

支持两种后端，通过配置切换：

| 后端 | 算法 | 速度 | 质量 | 适用场景 |
|------|------|------|------|---------|
| sklearn MiniBatchKMeans | Mini-batch 近似 K-Means | 基准（CPU） | 近似（inertia↑1–5%） | CPU 环境 |
| `kmeans_npu`（torch_npu） | 标准 Lloyd's K-Means | **~30–100×**（Ascend 910B） | 精确 | NPU/GPU 生产环境 |

**MiniBatchKMeans 参数**：
```python
MiniBatchKMeans(
    n_clusters=K,
    batch_size=50000,   # 每批处理条数
    random_state=42,
    n_init=3,           # 3次随机初始化取最优
    max_iter=100,
)
```

**NPU Lloyd's 算法（`kmeans_npu.py`）**：

```
初始化：K-Means++ 方式选取初始质心（概率正比于距最近质心的距离）
Lloyd's 迭代（最多 100 轮）：
  1. 分配：分批计算 N×K 距离矩阵
     for i in range(0, N, chunk_size=50,000):
         dists[i:i+chunk] = torch.cdist(X[i:i+chunk], C)  # NPU 矩阵乘加速
     labels = dists.argmin(dim=1)
  2. 更新：new_C[k] = mean(X[labels==k])
     空 cluster 随机重新初始化（防止退化）
  3. 收敛判断：||new_C - C||₂ / K < tol（默认 1e-4）
重复 n_init=3 次，保留 inertia 最小的结果
```

**显存估算**（D=256, FP32，chunk=50K）：
- stage1_icon：N=2.25M, K=12,000 → 完整矩阵 108GB，分批后每批 **2.4GB**（单卡 64GB 可容）
- stage2_illustration：N=0.45M, K=6,000 → 每批 **0.5GB**

### 10.3 K 值选择

| Domain | K | 记录数（过滤后） | 平均 cluster 大小 | 选择理由 |
|--------|---|----------------|-----------------|---------|
| `stage1_icon` | **12,000** | ~2,250,000 | ~188 | 数据量大，需足够 K 覆盖多样图标风格 |
| `stage2_illustration` | **6,000** | ~450,000 | ~75 | 语义空间远大于 icon（构图×风格×主题），需更细粒度分区 |

**K 值选择原则**：
- K 远小于 N（N/K ≈ 75–188），每个 cluster 有足够数量的成员，使后续 cluster-level 采样有意义
- K 足够大（> 1000），确保语义空间被充分划分，不同主题/风格的指令落在不同 cluster 中
- illustration K 与 icon K 之比（6K:12K = 1:2）远高于其数据量之比（1:5），这是因为 illustration 的语义空间相对于数据量更大

### 10.4 输出字段

聚类完成后，每条记录追加以下 `_meta` 字段：

| 字段 | 含义 |
|------|------|
| `cluster_id` | 所属 cluster 编号（0 起始） |
| `cluster_size` | 该 cluster 的成员总数 |
| `distance_to_centroid` | 该记录的 embedding 与 cluster 质心的 L2 距离 |
| `bucket_key` | 所属 domain（与 domain 值相同，供采样索引） |

`distance_to_centroid` 是后续采样中"选最具代表性样本"的核心依据：距离越小，该指令越接近该语义簇的中心，代表性越强。

### 10.5 各 domain 独立聚类的原因

两个 domain 的指令在语义空间上分布差异极大（icon 指令短、抽象；illustration 指令长、描述性强），混合聚类会导致跨 domain 的大型异质 cluster，失去语义连贯性。各 domain 独立聚类保证每个 cluster 内的指令在语义上同质。

---

## 11. Step 8：采样（Sample）

### 11.1 功能

从聚类结果中采样约 **100 万条**记录（`pool_1000k`），并进一步划分为：
- `high_priority_pool`（20 万条）：每个语义簇均有覆盖，多样性最高 → **SFT（监督微调）阶段**使用，用大模型（Gemini 等）蒸馏，作为冷启动训练集
- `anneal_pool`（~80 万条）：在各语义簇中心周边展开，数量更多 → **退火微调（Annealing）阶段**使用，用开源模型蒸馏，扩大数据规模

**训练策略含义**：SFT 阶段（high_priority）优先使用多样性最高的 200K，使模型快速建立对各语义类型的覆盖；退火阶段（anneal）补充 800K 增量数据，在 SFT 基础上进一步强化各语义类型的细节学习。

### 11.2 Bucket 配额分配（第一层）

**默认：按数据量比例**

```
stage1_icon quota    ≈ 700,000  （~83% × 1M，覆盖大量图标样本）
stage2_illustration  ≈ 300,000  （显式提权，见下）
```

**illustration 手动提权**

illustration 原始数据量占比约 17%，但具有更高的训练价值：
- 指令更长、语义更丰富（中位数 408 字符 vs icon 的 79 字符）
- SVG 更复杂（3K–30K 字符），是更难的生成任务
- 每条样本消耗更多 teacher model tokens，稀缺性更高

配置 `bucket_quota_overrides: {stage2_illustration: 300000}` 将 illustration 提升到 30%，其余 700K 自动分配给 stage1_icon。

### 11.3 Cluster 预算分配（第二层）

**按 √(cluster_size) 分配 budget，每个 cluster 至少 1**

```python
raw_budget[c] = bucket_quota × √cluster_size[c] / Σ√cluster_size
budget[c] = max(1, floor(raw_budget[c]))
```

sqrt 权重的含义：

| 方案 | 大 cluster 获得 | 小 cluster 获得 |
|------|---------------|---------------|
| 均匀分配（size⁰） | 与小 cluster 相同 | 过多（相对占比大） |
| 正比分配（size¹） | 过多 | 极少（极端不均） |
| **sqrt 分配（size^0.5）** | **适中，但相对占比低** | **相对更多，增强多样性** |

sqrt 权重在"完全均匀"与"完全正比"之间取得平衡：大 cluster 多采，但不会压制小 cluster（每个 cluster 保证至少 1 条）。

### 11.4 Cluster 内选取（第三层）

```python
sorted_recs = sorted(recs, key=lambda r: distance_to_centroid)
pool_records.extend(sorted_recs[:budget])
```

优先选最靠近质心的记录（最具代表性），避免选取 cluster 边界的噪声点。

> **注**：此三层分配机制导致 `pool_1000k` 实际条数可能略小于 100 万，原因是当某 cluster 的 budget 超过其实际 size 时，deficit 不做补偿。全量数据上预计偏差 ~0.5%，属于设计预期内。

### 11.5 high_priority_pool 选取（两阶段 Round-Robin）

从 `pool_1000k` 中选出最具代表性的 20 万条：

**Phase 1（one-per-cluster，约 18K 条）**：

```python
for cluster_key in cluster_keys:
    # 从每个 cluster 取 distance_to_centroid 最小的 1 条
    hp_records.append(cluster_sorted_records[cluster_key][0])
```

覆盖所有 cluster（stage1_icon 约 12K 个 + stage2_illustration 约 6K 个 = 约 18K 个）。

**Phase 2（Round-Robin 补充至 20 万）**：

```
while len(hp_records) < 200,000:
    for cluster_key in cluster_keys（轮转）:
        取该 cluster 下一条未被选中的 distance 最小记录
        加入 hp_records
```

**为什么用 Round-Robin 而不是全局 distance 排序？**

| 方案 | 行为 | 问题 |
|------|------|------|
| 全局 distance 排序（旧） | 按绝对距离从小到大补充 | dense cluster（distance 绝对值小）在 Phase 2 中贡献过多记录，稀疏 cluster 贡献极少，语义多样性下降 |
| **Round-Robin（当前）** | 各 cluster 轮流贡献下一条最近记录 | 各 cluster 贡献条数均衡（约 200K / 18K ≈ 11 条/cluster），不因 distance 绝对值而倾斜 |

**cluster 内多条记录是否会重复？**

不会。经过 exact dedup 和 near dedup 后，同一 cluster 内的所有记录都是语义相近但文本不同的独立 instruction。11 条记录覆盖了该语义簇内的多种表达方式，是 SFT 训练所期望的多样性（例如"画一个简单的房屋图标"有多种英文表达，均属同一语义簇但文本各异）。

**为什么不对 pool_1000k 重新聚类（K=200K）再选质心？**

理论上，对 pool_1000k（1M 条）做 K=200K 的 KMeans，取每个 micro-cluster 的质心最近 1 条，可以得到更精细的 coreset。分析如下：

| 考量 | 重新聚类（K=200K） | Round-Robin（当前） |
|------|-------------------|-------------------|
| 多样性 | 略优：每个 micro-cluster 保证 1 条 | 每个 cluster 约 11 条，稍有语义重叠 |
| 计算代价 | K=200K, N=1M, D=256 → 可接受（NPU 约 10 分钟），但需额外流程 | 无额外计算 |
| 工程复杂度 | 需新增聚类步骤，管道更长 | 原生支持 |
| 实际收益 | 每个原始 cluster 仅约 55 条，分成 11 个 sub-cluster 粒度极细，sub-cluster 内部几乎无重复可消除 | 同 pool_1000k 级别的语义覆盖 |

**结论**：对于 SFT 数据需求，每 cluster 11 条已足够，Round-Robin 是最优的工程实用方案。若未来需要极致多样性（如 high_priority_pool 缩小到 50K），可引入 K=50K 重聚类方案。

### 11.6 anneal_pool 构成

```python
hp_ids = set(r["_meta"]["id"] for r in hp_records)
anneal = [r for r in pool_records if r["_meta"]["id"] not in hp_ids]
```

`pool_1000k = high_priority ∪ anneal`，两者无交集，总量一致。

---

## 12. 数据漏斗与规模估算

以全量 SAgoge 数据（~7.7M 条）为基准的典型漏斗：

```
阶段                    数据量          说明
──────────────────────────────────────────────────────────────────
原始输入               ~7,729,000      6 个文件合并
↓ Extract              ~7,729,000      极少量 skip（格式完整率高）
↓ Clean                ~7,728,000      近乎无损（过短指令极少）
↓ Exact Dedup          ~3,320,000      移除 ~57% 精确重复
                                       · img2svg 与 text2svg 同 domain 内重叠率 ~100%
                                         （相同底层 SVG 图像生成几乎相同的 instruction）
                                         → text2svg 几乎全被 img2svg 覆盖，有效记录集 ≈ img2svg 集
                                       · stage2_icon（~1.11M）被 stage1_icon 全量覆盖（跨 stage 重叠）
                                       · 有效 domain 缩为两桶：stage1_icon + stage2_illustration
↓ Near Dedup           ~2,860,000      移除 ~14% 近似重复
                                       stage1_icon（阈值 0.8）去重比例高于 illustration（0.7）
↓ SVG Filter           ~2,700,000      移除 icon 域底部 10% 极简 SVG（~160K 条）
                                       illustration 全部保留
↓ Embed                ~2,700,000      无损（仅向量化，不过滤）
↓ Cluster              ~2,700,000      无损（仅附加 cluster 信息）
                                       stage1_icon: K=12,000
                                       stage2_illustration: K=6,000
↓ Sample               ~1,000,000      从 2.7M 中采 1M（约 37% 采样率）
──────────────────────────────────────────────────────────────────
pool_1000k             ~1,000,000      总采样池（stage1_icon 700K + illustration 300K）
high_priority_pool       200,000       SFT 阶段（大模型蒸馏，Round-Robin，cluster 覆盖率 100%）
anneal_pool            ~800,000        退火阶段（开源模型蒸馏，pool_1000k 去掉 high_priority）
```

**domain 配比**（最终 pool_1000k）：

| Domain | 数量 | 占比 | 配比策略 |
|--------|------|------|---------|
| stage1_icon | ~700,000 | 70% | 自动分配（原始比例 83%，因 illustration 提权而降低） |
| stage2_illustration | 300,000 | 30% | 手动提权（原始比例 17%，提权因语义价值高） |

---

## 13. 超参汇总与选择依据

| 超参 | 值 | 模块 | 选择依据 |
|------|-----|------|---------|
| `svg_filter_bottom_pct` | 0.10 | SVG Filter | 移除 stage1_icon 最底部 10% 过简单样本 |
| `near_dedup.num_perm` | 128 | Near Dedup | MinHash 方差可控（σ≈0.035 @J=0.8），内存 3GB |
| `near_dedup.char_ngram` | 5 | Near Dedup | 短文本字符 5-gram 的经验最优 |
| `near_dedup.threshold.stage1_icon` | 0.8 | Near Dedup | 短文本严格去重 |
| `near_dedup.threshold.stage2_illustration` | 0.7 | Near Dedup | 长文本 Jaccard 值天然偏低，放宽阈值 |
| `embedding.dimension` | 256 | Embed | Matryoshka 截断维度，语义信息/计算开销平衡点（4B/0.6B 均适用） |
| `embedding.model` | Qwen3-Embedding-4B（生产）/ 0.6B（验证） | Embed | 支持 Matryoshka，只需换 model_path，其余配置不变 |
| `embedding.shard_size` | 100,000 | Embed | 单 shard 内存 100k×256×4B = 100MB，可控 |
| `clustering.k_per_bucket.stage1_icon` | 12,000 | Cluster | N≈2.25M，K/N=0.53%，平均 188条/cluster |
| `clustering.k_per_bucket.stage2_illustration` | 6,000 | Cluster | N≈0.45M，语义丰富，K/N=1.33%，平均 75条/cluster |
| `clustering.minibatch_size` | 50,000 | Cluster | MiniBatchKMeans 批大小；NPU 模式下为 cdist 分批大小 |
| `clustering.n_init` | 3 | Cluster | 3次随机初始化取最优，避免局部最优 |
| `clustering.npu_chunk_size` | 50,000 | Cluster | NPU 模式：chunk=50K 时单批显存 ≤2.4GB（单卡 64GB 可容） |
| `sampling.total_pool_size` | 1,000,000 | Sample | 蒸馏阶段目标规模 |
| `sampling.high_priority_pool_size` | 200,000 | Sample | SFT 训练集（大模型蒸馏，Round-Robin 覆盖全部 cluster） |
| `sampling.anneal_pool_size` | 800,000 | Sample | 退火训练集（开源模型蒸馏） |
| `sampling.bucket_quota_overrides.stage2_illustration` | 300,000 | Sample | illustration 提权至 30%（原比例 17%，价值更高） |
| `num_workers` | 4（默认）/ 128（生产） | 全局 | 对 extract/near_dedup/cluster 阶段并行 |

---

## 14. 工程设计要点

### 14.1 img2svg 优先性的传递

img2svg 优先级通过两个机制在整个流水线中得以保持：
1. **文件顺序**：input_paths 中 img2svg 在前，extract 阶段保持此顺序 merge
2. **first-come-wins**：exact dedup 和 near dedup 的 Phase 2 均为顺序处理，先处理的 img2svg 自然胜出

### 14.2 stage2_icon 两桶合并

全量数据验证发现 stage2_icon 在 exact dedup 后无存活记录，原因已在 §6.3 详述。设计应对：
- `DOMAINS = ("stage1_icon", "stage2_illustration")`，去除 `stage2_icon`
- `infer_domain()` 将 stage2/icon 路径映射到 `"stage1_icon"`
- near_dedup.thresholds 和 clustering.k_per_bucket 中去除 `stage2_icon` key
- 此变更不影响 extract 阶段（stage2/icon 文件仍读取，只是 exact dedup 后消失）

### 14.3 断点续传（Resume）

Embed 阶段支持 shard 级别的断点续传（已存在的 shard 文件跳过），其他阶段每阶段输出为独立文件，主程序通过 `--resume` 检查输出文件是否存在来跳过已完成阶段。

### 14.4 `_meta` 统一封装

所有流水线元数据（id、domain、source、svg_len、cluster 信息等）均封装在 `_meta` 子对象中，顶层只保留 `instruction`，避免字段污染，便于后续转换为 panguml 统一格式。

### 14.5 OpenBLAS 多核崩溃防护

在 192 核等超多核机器上，`ProcessPoolExecutor` fork 出的子进程中，OpenBLAS 会尝试创建与 CPU 数量匹配的线程数，超过其编译时上限（通常 64）时会触发 `double free` 崩溃。通过在 worker 启动时（`initializer` 参数）将 `OPENBLAS_NUM_THREADS`、`OMP_NUM_THREADS` 等全部限制为 1 解决。

### 14.6 Near Dedup Phase 2 性能

旧版每条记录构造 `MinHash()` 对象耗时 ~335μs，2.8M 条需 19 分钟。新版直接在预拼接的 numpy 矩阵上做 band hash（`hv[s:e].byteswap().tobytes()`），速度提升 62×，2.8M 条 Phase 2 仅需 19 秒。`num_workers` 仅加速 Phase 1（MinHash 计算），Phase 2 的瓶颈已通过算法优化解决。

### 14.7 NPU KMeans 显存管理

standard Lloyd's K-Means 的完整 N×K 距离矩阵（N=2.25M, K=12K, FP32）约 108GB，超出单卡显存。通过分批 `torch.cdist`（`chunk_size=50,000`）将每步峰值显存降至 2.4GB，在 Ascend 910B（64GB）上安全运行。空 cluster 随机重初始化防止质心退化。

### 14.8 采样比例的数量一致性

`pool_1000k ≡ high_priority_pool ∪ anneal_pool`（集合不相交，union 等于 pool_1000k），通过 id 集合差保证。`anneal_pool_size` 配置值为软约束，实际条数由 pool_1000k 减去 high_priority 决定。

### 14.9 gt_svg 字段的存储与用途

`_meta.gt_svg` 在 extract 阶段写入（存储原始 assistant turn 的完整文本，含 ` ```svg ` 包裹），通过所有下游阶段原样透传，最终保留在 `pool_1000k.jsonl`、`high_priority_pool.jsonl`、`anneal_pool.jsonl` 中。

用途：
1. **蒸馏后对比可视化**：将 teacher model 生成的 SVG 与 `gt_svg` 并排渲染，评估蒸馏质量
2. **来源溯源**：可从最终采样记录直接查看原始 GT SVG，无需反查原始文件

存储代价：icon domain `gt_svg` 约 100–2000 字符/条（可接受），illustration domain 约 3K–30K 字符/条（1M 条总量约增加 5–15 GB 中间文件）。若存储受限，可通过配置 `extract.save_gt_svg: false` 关闭（当前默认开启）。
