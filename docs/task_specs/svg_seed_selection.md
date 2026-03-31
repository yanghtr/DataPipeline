# 种子 Query 筛选模块需求文档（统一格式数据版）

## 1. 背景

我们已经有统一的数据格式 skill，且原始数据已经被转换为统一 canonical schema。

当前需要实现一个**种子 Query 筛选模块**，用于从统一格式样本中抽取 user instruction 文本，并依次完成：

1. 抽 instruction 内容
2. 最小清洗
3. exact dedup
4. MinHash near dedup
5. Qwen3-Embedding
6. 聚类 / 采样
7. 导出 seed query 池

本模块的目标是为后续 SVG teacher distillation 提供一个：

- 覆盖面广
- 重复低
- 噪声低
- 尽量少人工 bias

的种子 query 集合。

---

## 2. 模块目标

本模块的目标是：

1. 从统一格式样本中抽取 user 侧 instruction 文本；
2. 对 instruction 做最小必要清洗；
3. 去除完全重复和近重复 instruction；
4. 通过 text embedding 建立语义空间；
5. 通过分桶、聚类、采样来保留覆盖面；
6. 产出两类 seed pool：
   - **anneal pool**：大规模覆盖优先，目标约 `900k`
   - **high-priority pool**：更适合优先蒸馏的候选池，目标约 `100k`

注意：
- 当前 `100k` 只是高优先级候选池，不等同于最终 platinum SFT 数据。
- 最终真正高质量的 `image/SVG` 数据，还应依赖下游 teacher 输出结果的 SVG / render 质检。

---

## 3. 非目标

本模块**不做**以下事情：

- 不做 render 质量评估；
- 不做 teacher model 推理；
- 不做 image 侧打分；
- 不做多模态 embedding；
- 不做人手语义词表评分；
- 不做 LLM judge；
- 不做最终训练数据打包；
- 不做多轮对话处理。

---

## 4. 输入前提

输入数据已经满足统一 schema，具体格式约束见：

- `.claude/skills/panguml-format/SKILL.md`

本模块不负责“把原始数据转成统一格式”，只负责：

- 从统一格式样本中抽 instruction
- 清洗 instruction
- 去重
- embedding
- 聚类 / 采样

---

## 5. 输入输出

## 5.1 输入

输入为统一格式样本集合。
每条记录为一个 canonical sample。

全部数据来源于`/home/yanghaitao/Projects/Data/processed/SAgoge`中的`.jsonl`文件，包括：

./stage1/icon/generation/img2svg/data_000000.jsonl
./stage1/icon/generation/text2svg/data_000000.jsonl
./stage2/illustration/img2svg/data_000000.jsonl
./stage2/illustration/text2svg/data_000000.jsonl
./stage2/icon/generation/img2svg/data_000000.jsonl
./stage2/icon/generation/text2svg/data_000000.jsonl

---

## 5.2 输出

输出文件的根目录在`/home/yanghaitao/Projects/Data/query_seed/SAgoge`，建议包括：

1. `instruction_pool_cleaned.*`
   - 抽取后的 instruction 池（最小清洗后）
2. `exact_dedup_removed.*`
   - exact dedup 的去重映射/日志
3. `near_dedup_removed.*`
   - near dedup 的去重映射/日志
4. `embeddings/`
   - embedding 结果或其引用
5. `cluster_assignments.*`
   - 每条 instruction 的 bucket / cluster / centrality
6. `anneal_pool.*`
   - 约 900k
7. `high_priority_pool.*`
   - 约 100k
8. `stats.json`
   - 全流程统计
9. `run_config.json`
   - 运行配置快照

- 最终输出格式是 JSONL 

---

## 6. 总体流水线

模块必须按以下顺序处理：

- 读取统一格式样本
- 从样本中抽取 instruction 文本
- 最小清洗
- exact dedup
- near dedup（MinHash）
- SVG复杂度分层抽样
- text embedding（Qwen3-Embedding）
- 分桶
- 桶内聚类
- 采样生成 900k / 100k
- 导出结果与日志

建议数据清洗过程使用统一格式，方便统一各个清洗规则的接口，同时保留每条数据原始信息。

---

## 7. 核心原则

### 7.1 少规则、少 bias
尽量避免手工语义词表、手工复杂度打分。

### 7.2 覆盖优先
核心目标不是挑“最好看”的 prompt，而是构造覆盖广、重复低、噪声低的 seed query 池。

### 7.3 最小清洗
清洗只做低风险、白盒、确定性的处理。

### 7.4 可追溯
每一步都应有明确输入输出和日志。

### 7.5 可恢复
大规模数据处理必须支持中断恢复。

---

## 8. instruction 抽取规则

本模块首先要从统一格式 sample 中抽取 instruction 文本。 只从：

- `data[0]`
- `role == "user"`

的 `content` 中提取 `type == "text"` 的文本。

说明：
- 本模块的目标是筛选 **text instruction seeds**

---

## 9. 最小清洗规则

本模块只对 instruction 文本做最小清洗。
对抽出的 instruction 执行：

1. 类型检查：
   - 必须是字符串，否则丢弃
2. Unicode 规范化
3. 去首尾空白：
   - `strip()`

---

## 10. 去重

### 10.1 exact dedup

最小清洗后的 instruction 必须做 exact dedup。

应输出：

- `exact_dedup_kept`
- 保存相关统计信息

---

### 10.2 near dedup（MinHash）

exact dedup 之后，必须进行 near dedup。

#### 目标

去掉语义非常接近、或只在局部词序/冠词/少量修饰上不同的 instruction。

#### 推荐算法

v1 推荐：

- shingling
- MinHash
- LSH

#### 推荐库

优先使用：

- `datatrove`

备选：

- `datasketch`

#### 阈值

- 给出阈值设定的逻辑和方案

#### 输出

应输出：

- `near_dedup_removed`
- `near_dedup_kept`
- near dup 相关统计

---

## 11. 根据SVG复杂度进行分层抽样

对于现在仍然保留下来的每一条数据的`instruction`，提取其对应的SVG。分别对`stage1/icon`和`stage2/icon`的SVG文本字符长度进行统计，对于每一类，单独统计后，去除SVG文本字符长度最小的10%那部分，因为这部分的SVG过于简单，不利于模型学习。我们保留全部留存下来的 `stage2/illustration` 。

---

## 12. Embedding

接着，需对 instruction 池做 text embedding。

### 12.1 目标

embedding 仅用于：

- 建立语义空间
- 聚类
- 采样

不用于语义质量打分。

### 12.2 推荐模型

目前默认使用模型 `Qwen3-Embedding-0.6B`，其ckpt路径为：`/home/yanghaitao/Projects/CKPT/Qwen3-Embedding-0.6B`。用户可以指定输入更大的模型的路径。在代码测试时，默认没有GPU/NPU，只能使用CPU进行代码验证。同时需要保留命令接口，可以指定命令行参数调用GPU/NPU。

### 12.3 embedding 输入

使用当下仍留存下来的数据。

- 判断我们当前任务是否需要加固定 embedding instruction prefix。若加，具体内容是什么？如果差别不大v1版本可以不加

### 12.4 维度

- 维度默认取 `256`（判断v1版本是否足够？如果实在不够才考虑用512） 

### 12.5 批处理要求

embedding 阶段必须支持：

- batch inference
- shard 处理
- resume
- 输出缓存

---

## 13. 分桶（Bucketing）

为了保留 coarse coverage，embedding / clustering 前应先分桶。

bucket_key 推荐 bucket 由以下字段：

- `domain`

例如：

- `icon`
- `illustration`

---

## 14. 聚类

embedding 后，必须在每个 bucket 内做聚类。

### 14.1 推荐算法

v1 默认推荐：

- `MiniBatchKMeans`或者更加高效的聚类算法

### 14.2 聚类范围

- 每个 bucket 单独聚类
- 不对全量混在一起直接聚类

### 14.3 K 的设定

需要合理设置并给出理由

### 14.4 输出

每条 instruction 至少要有：

- `bucket_key`
- `cluster_id`
- `cluster_size`
- `distance_to_centroid` 或等价 centrality

---

## 15. 采样

最终总体必须输出：

- 约 `900k` anneal pool
- 约 `100k` high-priority pool

## 15.1 总体目标

### anneal pool
- 覆盖面优先
- 长尾保留
- 允许质量一般但干净、非重复的 query

### high-priority pool
- 更中心、更稳定、更代表性的 query
- 仍需保证跨 bucket 覆盖
- 当前只是高优先级候选，不是最终 platinum 数据

---

## 15.2 bucket 配额

必须先给每个 bucket 分配 quota。

- quota 按 bucket size 比例分配

---

## 15.3 cluster 内预算

同一个 bucket 内，不应只从最大 cluster 取样。

推荐：

- 每个 cluster 至少取 1 条（若 bucket quota 允许）
- larger cluster 的额外预算按次线性增长

审视方案，是否需要推荐启发式：

- `budget(cluster) ∝ sqrt(cluster_size)`
- 可能需要统计`cluster_size`的分布情况进行分析判断。

---

## 15.4 cluster 内排序

v1 推荐不使用人工语义评分，只用：

- `centrality`
- stable tie-break

排序优先级：

1. 更靠近 cluster centroid
2. stable tie-break（如 id）

---

## 15.5 900k 采样规则

推荐先采样生成1000k，后面采样更高质量的100k后，剩余的就是900k：

1. 按 bucket 分 quota
2. bucket 内按 cluster 分 budget
3. cluster 内按 centrality 排序取样

---

## 15.6 100k 采样规则

- `100k` 直接从前面先采的1000k中筛选出更高质量的100k。比如：

1. 更多样性、更高质量的采样
2. 使用更严格的 quota / centrality 规则
3. 保留跨 bucket 覆盖
或者更好的方法，如果没有也可以随机采


---

## 16. 配置要求

必须有统一配置文件。
推荐支持YAML

### 必须可配置项
至少包括：
- input path(s)
- output path(s)
- input/output format
- 清洗开关
- 长度阈值
- exact dedup 配置
- near dedup 配置
- embedding model
- embedding dimension
- batch size
- bucket fields
- clustering 参数
- sampling 参数
- anneal pool size
- high-priority pool size
- random seed

---

## 17. CLI 要求

代码必须提供 CLI。

同时提供统一script可以读入YAML配置进行自动化运行：

- `run`

要求：
- 支持从中间阶段 resume
- 支持 dry-run / small subset (重要，我们默认是CPU环境，只能小规模调用embedding模型，同时需要估计全量运行的时间并说明)

---

## 18. 推荐代码结构

建议代码放置在一个新的文件夹，生成如下模块：

- `extract.py`
- `clean.py`
- `dedup_exact.py`
- `dedup_near.py`
- `embed.py`
- `bucket.py`
- `cluster.py`
- `sample.py`
- `io_utils.py`
- `config.py`
- `main.py`

### 推荐函数

- `extract_instruction_from_sample(sample) -> dict | None`
- `clean_instruction(text: str) -> str | None`
- `run_exact_dedup(records) -> ...`
- `run_near_dedup(records, config) -> ...`
- `embed_instructions(records, config) -> ...`
- `build_bucket_key(record, config) -> str`
- `cluster_bucket(records, config) -> ...`
- `sample_anneal_pool(records, config) -> ...`
- `sample_high_priority_pool(records, config) -> ...`

---

## 19. 日志与统计

必须输出详细统计。

至少包括：

- 总样本数
- 抽出 instruction 的样本数
- 清洗后保留数
- exact dedup 去掉数
- near dedup 去掉数
- 最终 deduped pool 数
- bucket 数
- 每个 bucket 大小
- cluster 数
- anneal pool 大小
- high-priority pool 大小

并保存：

- `stats.json`
- `run_config.json`

---

## 20. skip / remove reason

每一步被丢弃或合并的样本都应记录 reason。

推荐 reason：

### extract 阶段
- `missing_instruction_text`

### clean 阶段

### exact dedup 阶段
- `exact_duplicate_removed`

### near dedup 阶段
- `near_duplicate_removed`

### sampling 阶段
- 不算 remove，可记为未入选

---

## 21. 测试要求

必须写测试。

至少包括：

1. instruction 抽取测试
2. 最小清洗测试
3. exact dedup 测试
4. near dedup 测试
5. embedding 接口测试（mock 即可，只能使用CPU）
6. bucket 构造测试
7. cluster / sample 基本流程测试
8. determinism 测试

---

## 22. 交付物

生成代码时，至少应包含：

1. 完整 Python 源码
2. 配置模板
3. CLI 入口
4. README / 使用说明
5. 测试
6. 小型 mock 数据示例

---

## 23. 给 Claude Code 的明确要求

请根据本需求文档生成代码时，严格遵守以下要求：

1. 输入是已经符合统一 canonical schema 的样本；
2. 本模块只处理 text instruction seed selection；
3. 不要引入人手语义词表打分；
4. 不要引入 LLM judge；
5. 去重必须包含 exact dedup 和 MinHash near dedup；
6. embedding 默认使用 Qwen3-Embedding；
7. 采样必须体现 bucket coverage + cluster coverage；
8. 所有关键参数都要可配置；
9. 全流程必须支持日志、统计、resume；
10. 代码要模块化，方便后续替换 embedding / clustering / sampling 实现。

