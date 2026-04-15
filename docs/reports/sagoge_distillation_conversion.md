# SAgoge 蒸馏数据转换技术报告

## 目录

1. [背景与目标](#1-背景与目标)
2. [整体架构：两阶段设计](#2-整体架构两阶段设计)
3. [输入格式](#3-输入格式)
4. [输出格式：canonical schema](#4-输出格式canonical-schema)
5. [Convert 阶段详解](#5-convert-阶段详解)
6. [Filter 阶段详解](#6-filter-阶段详解)
7. [_meta 字段说明](#7-_meta-字段说明)
8. [图片目录结构](#8-图片目录结构)
9. [性能设计](#9-性能设计)
10. [CLI 参数参考](#10-cli-参数参考)
11. [典型调用示例](#11-典型调用示例)

---

## 1. 背景与目标

将 SAgoge 蒸馏数据集的原始 JSONL（模型蒸馏响应）转换为统一多模态 canonical schema 格式，供后续训练流程直接消费。

核心需求：
- 从模型响应中提取 SVG 代码，渲染为 PNG 图片
- 构建 `[image, instruction] → svg_code` 的多模态对话样本
- 保留完整过程元数据（`_meta`），支持事后分析与二次过滤
- 大规模（100K+ 条）并行处理，稳定不卡死

---

## 2. 整体架构：两阶段设计

```
原始 JSONL
    │
    ▼
┌─────────────────────────┐
│  convert 阶段            │
│  并行渲染 SVG → PNG      │
│  写入中间 JSONL（含_meta）│
└─────────────────────────┘
    │
    ▼
中间 JSONL（全量，含失败记录）
    │
    ▼
┌─────────────────────────┐
│  filter 阶段             │
│  按 _meta 过滤            │
│  写入最终 JSONL（不含_meta）│
└─────────────────────────┘
    │
    ▼
最终 JSONL（仅成功记录）
```

两阶段分离的好处：
- convert 产出的中间 JSONL 包含全部记录（含失败），可独立分析各类失败比例，无需重跑渲染
- filter 无需任何渲染，纯内存操作，可随时调整过滤策略重新生成最终数据集

---

## 3. 输入格式

原始 JSONL，每行一条记录，字段如下：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | str | 记录唯一标识 |
| `instruction` | str | 用户指令文本（作为训练样本的 user text） |
| `response` | str | 模型生成的完整响应，通常包含一个或多个 SVG 代码块 |
| `model` | str | 生成响应的模型名称 |
| `prompt_tokens` | int | 提示词 token 数 |
| `completion_tokens` | int | 补全 token 数 |
| `finish_reason` | str | 停止原因（`stop` / `length` 等） |

---

## 4. 输出格式：canonical schema

最终 JSONL 每行是一个完整的多模态对话样本：

```json
{
  "meta_prompt": [""],
  "data": [
    {
      "role": "user",
      "content": [
        {
          "type": "image",
          "image": {
            "type": "relative_path",
            "format": "image/png",
            "relative_path": "00000/000000001.png",
            "width": 800,
            "height": 600
          }
        },
        {
          "type": "text",
          "text": {
            "type": "string",
            "format": "utf-8",
            "string": "Draw a simple house icon in flat design style"
          }
        }
      ]
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "text",
          "text": {
            "type": "string",
            "format": "utf-8",
            "string": "```svg\n<svg ...>...</svg>\n```"
          }
        }
      ]
    }
  ]
}
```

关键设计：
- **user content 顺序**：图片在前，文字指令在后
- **assistant text**：将最后一个 SVG 包裹在 ` ```svg\n...\n``` ` 中
- **relative_path**：相对于调用方传入的 `--image-root`，可被下游训练框架直接拼接为绝对路径
- **train_mode**：`sft` 对应 `meta_prompt: [""]`，`pretrain` 对应 `meta_prompt: ["", ""]`

---

## 5. Convert 阶段详解

### 5.1 处理流程（单条记录）

```
原始 record
    │
    ├─ 1. 提取 instruction（空则跳过）
    │
    ├─ 2. 从 response 提取最后一个 SVG 代码块
    │       优先匹配 ```svg...``` fenced block
    │       fallback：裸 <svg>...</svg>
    │
    ├─ 3. 解析 SVG 渲染尺寸
    │       width/height 属性 → viewBox → (0,0) 交由 cairosvg 决定
    │
    ├─ 4. cairosvg 渲染为 PNG（带白色背景）
    │       超时则标记 render_failed
    │
    ├─ 5. 全白检测（记录到 _meta，不用于 convert 阶段跳过）
    │
    ├─ 6. 构建 canonical sample
    │
    └─ 7. schema 校验（validate_sample）
```

### 5.2 SVG 提取策略

`extract_last_svg` 取响应中**最后一个**有效 SVG，理由：模型在思考过程中可能输出多个中间版本，最后一个通常是最终版本。

### 5.3 并行实现

使用 `ProcessPoolExecutor` 多进程并行，worker 函数 `_process_record` 为模块级函数（满足 pickle 序列化要求）。所有参数通过 tuple 传入。

结果按原始 idx 顺序写入中间 JSONL，保证输出与输入行序一致。

---

## 6. Filter 阶段详解

### 6.1 过滤条件（按优先级）

| 条件 | 过滤原因标签 | 说明 |
|------|------------|------|
| `_meta.skip_reason` 非空 | 原值（如 `render_failed`） | convert 阶段已标记为跳过 |
| `svg_format_valid == False` | `invalid_svg_format` | 响应中找不到任何 SVG 代码 |
| `render_success == False` | `render_failed` | cairosvg 渲染失败或超时 |
| `is_all_white == True` | `all_white_image` | 渲染结果为全白图（无效输出） |

满足任一条件则过滤，写入统计后跳过；否则去掉 `_meta` 字段输出到最终 JSONL。

### 6.2 全白检测的设计取舍

全白检测（`is_all_white`）在 convert 阶段**只记录不跳过**，原因：

1. 全白图的比例和分布本身是有价值的分析指标
2. 过滤策略可能随需求变化（如放宽阈值），分离后无需重跑渲染
3. filter 阶段可随时调整是否过滤全白图

---

## 7. _meta 字段说明

中间 JSONL 每条记录（含失败记录）均携带 `_meta`：

| 字段 | 类型 | 说明 |
|------|------|------|
| `input_id` | str\|None | 原始记录的 `id` |
| `model` | str\|None | 生成模型名称 |
| `response` | str\|None | 完整原始响应文本 |
| `prompt_tokens` | int\|None | 提示词 token 数 |
| `completion_tokens` | int\|None | 补全 token 数 |
| `finish_reason` | str\|None | 停止原因 |
| `svg_format_valid` | bool | 是否成功从响应中提取到 SVG |
| `render_success` | bool | cairosvg 渲染是否成功 |
| `render_error` | str | 渲染失败原因（仅失败时存在） |
| `is_all_white` | bool | 渲染结果是否为全白图 |
| `skip_reason` | str\|None | 跳过原因；`None` 表示该记录成功转换 |

失败记录的 `_meta` 会附在一个只含 `_meta` 键的对象上写入中间 JSONL，filter 阶段可识别并计入统计。

---

## 8. 图片目录结构

为避免单一目录下文件数过多导致文件系统性能退化（ext4 dentry cache 在 5–10 万文件后命中率下降），PNG 按 shard 分子目录存放：

```
images_dir/
  00000/          ← shard 0，idx 0–4999
    000000000.png
    000000001.png
    ...
  00001/          ← shard 1，idx 5000–9999
    000005000.png
    ...
  00019/          ← shard 19，idx 95000–99999（100K 数据共 20 个 shard）
```

命名规则：
- **子目录**：`{shard_idx:05d}`，shard_idx = `idx // shard_size`，5 位补零，支持最多 99999 个 shard
- **文件名**：`{idx:09d}.png`，全局唯一 ID，跨 shard 不重名，便于从文件名反查原始记录
- `relative_path` 格式：`00000/000000001.png`（相对于 `--image-root`）

`--shard-size 0` 可禁用分目录（兼容旧版行为）。

---

## 9. 性能设计

### 9.1 PIL Image 资源管理

所有 PIL Image 对象均使用 `with` 语句管理生命周期，确保文件描述符及时释放，防止 worker 进程内 FD 泄漏和 GC 压力积累。

### 9.2 避免二次磁盘读取

`render_svg` 成功后将原始 `png_bytes` 存入 `RenderResult.png_bytes`；`check_all_white` 直接接收 bytes，不再重新读取已写入磁盘的文件：

```
cairosvg.svg2png() → png_bytes
    ├─ Image.open(BytesIO(png_bytes)).save(output_path)   # 写磁盘
    └─ check_all_white(png_bytes)                          # 直接用内存，不读磁盘
```

### 9.3 渲染超时保护

部分 SVG（含复杂 `<filter>`、大量 `<path>`、递归 `<use>`）会导致 cairosvg 内部 C 层无限阻塞，Python 层无法通过异常或线程中断。

解决方案：在 worker 进程内使用 `signal.SIGALRM`（Linux/macOS），超时后直接打断 C 层调用：

- 默认超时：60 秒
- 超时记录：`render_error = "render timeout after Ns"`，`skip_reason = "render_failed"`
- ProcessPoolExecutor 的每个 worker 是独立进程的主线程，满足 `SIGALRM` 的使用条件

---

## 10. CLI 参数参考

### convert 子命令

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` / `-i` | ✓ | — | 原始输入 JSONL 文件路径 |
| `--images-dir` | ✓ | — | PNG 输出目录 |
| `--image-root` | ✓ | — | canonical schema 中 relative_path 的计算基准目录 |
| `--inter-jsonl` | ✓ | — | 中间 JSONL 输出路径（含 `_meta`） |
| `--train-mode` | | `sft` | 训练模式：`sft` 或 `pretrain` |
| `--workers` | | CPU 核数 | 并行 worker 进程数 |
| `--shard-size` | | `5000` | 每个子目录最多存放的图片数；`0` 禁用分目录 |
| `--render-timeout` | | `60` | 单条 SVG 渲染超时秒数；`0` 不限时 |
| `--log-path` | | — | 日志文件路径（可选，同时输出到 stderr） |

### filter 子命令

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` / `-i` | ✓ | — | 中间 JSONL 文件路径（convert 产出） |
| `--output-jsonl` | ✓ | — | 最终 JSONL 输出路径（不含 `_meta`） |
| `--log-path` | | — | 日志文件路径（可选） |

---

## 11. 典型调用示例

```bash
# Step 1: Convert（渲染 SVG → PNG，生成中间 JSONL）
python -m converters.convert_sagoge_distill convert \
    --input       /data/raw/sagoge/distill.jsonl \
    --images-dir  /data/processed/sagoge/images \
    --image-root  /data/processed/sagoge \
    --inter-jsonl /data/processed/sagoge/inter.jsonl \
    --train-mode  sft \
    --workers     32 \
    --shard-size  5000 \
    --render-timeout 60 \
    --log-path    /data/logs/sagoge_convert.log

# Step 2: Filter（按 _meta 过滤，生成最终 JSONL）
python -m converters.convert_sagoge_distill filter \
    --input        /data/processed/sagoge/inter.jsonl \
    --output-jsonl /data/processed/sagoge/final.jsonl \
    --log-path     /data/logs/sagoge_filter.log
```

Convert 结束时日志会打印统计：

```
============================================================
  Statistics (pre-filter)
  Total:   100000
  Success: 94821
  Skipped: 5179
  Skip reasons:
    render_failed:       3102
    missing_svg_code:    1844
    missing_instruction:  233
============================================================
```

Filter 结束时进一步统计（含全白过滤）：

```
============================================================
  Statistics (filter pass)
  Total:   100000
  Success: 93417
  Skipped: 6583
  Skip reasons:
    render_failed:   3102
    missing_svg_code: 1844
    all_white_image:  1404
    ...
============================================================
```
