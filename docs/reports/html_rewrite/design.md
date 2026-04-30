# html_rewrite 模块设计文档

## 1. 背景与目标

**目标**：从真实网站 HTML 批量生产"dirty → clean HTML"改写训练数据。

**输入**：FineWebEdu JSONL，每条含原始网页 `html` 及元信息（`url`、`final_url`、`crawl_time`、`page_type`、`part` 等）。

推荐输入方式：
- `input_dir + input_filename_template + [input_start_index, input_end_index_exclusive)` 自动展开

兼容输入方式：
- 单文件：`input_path`
- 显式列表：`input_paths`

**输出**：默认按 shard 分目录输出。每个输入 shard 对应一组 Stage 1 / Stage 2 / 日志文件；同时写 `manifest.json` 和 run 级 aggregate summary，便于大规模 resume、排错与回溯。

**核心设计原则**：
- 预处理只做最小化处理（路径替换、超长截断、格式标准化），不重写结构、不删除 class/nav/sidebar/footer
- 样本过滤放在 Stage 1 最后，只决定“是否进入 Stage 2”，不回写或重构 HTML 结构
- 模型负责真正的语义改写
- 两阶段解耦：Stage 1（离线预处理）可单独重跑调参，Stage 2（模型改写）不重跑预处理
- 工程上复用 `distillation/` 已验证的并发/retry/resume/日志/结果保存模式

---

## 2. 整体架构

```
input shard JSONL(s)
  │
  ▼
[Stage 1: Preprocess]                html_rewrite/stage1_preprocess.py
  ├─ 媒体路径替换为 placeholder
  ├─ 超长 inline script / JSON payload / style / hidden input / comment 截断
  ├─ HTML 格式标准化（解析 + prettify + 去多余空行）
  ├─ 记录详细预处理统计
  ├─ 样本级过滤（超长 / 过空 / 结构异常）
  └─ 写出 keep / reject / summary
  │
  ▼  (中间文件，可调整阈值后 --no-resume 重跑)
run_root_dir/stage1/<shard>/
  ├─ preprocessed.jsonl
  ├─ stats.jsonl
  ├─ rejects.jsonl
  ├─ summary.json
  ├─ summary_reject_reasons.json
  └─ plots/
  │
  ▼
[Stage 2: Rewrite]                   html_rewrite/stage2_rewrite.py
  ├─ 并发调用 OpenAI-compatible 模型
  ├─ 提取模型输出中的 clean HTML
  └─ retry / resume / append-only 写出
  │
  ▼
run_root_dir/stage2/<shard>/
  ├─ output.jsonl
  └─ api_calls.jsonl

run_root_dir/
  ├─ manifest.json
  └─ aggregate/
      ├─ stage1_summary.json
      └─ stage2_summary.json
```

两个阶段通过中间文件完全解耦，各自支持独立 resume。Stage 1 无网络依赖，Stage 2 只读取对应 shard 的 Stage 1 keep 样本，不重跑预处理。reject 样本不进入 Stage 2，但必须保留日志，避免“静默丢数”。

补充说明：
- Stage 1 和 Stage 2 现在都采用 append-only、无序写最终 JSONL。
- 每条记录都会写出：
  - `record_uid`：流水线内部唯一键，格式为 `<input_shard>:<input_index>`
  - `id`：原始网页 id / url
  - `_meta`：原始网页元信息
  - `input_shard`、`input_index`、`source_input_path`：源输入定位信息
- resume / 去重统一基于 `record_uid`；不再依赖输出是否与输入同序，也不依赖 `id` 在多输入文件场景下天然唯一。

补充约束：
- `lxml` 是 Stage 1 的硬依赖。
- 如果当前环境缺少 `lxml`，应直接报错停止，不允许静默 fallback 到 `html.parser` 或其它 parser。

---

## 3. 记录唯一性与 Resume 机制

**核心变化**：当前设计不再要求 Stage 1 / Stage 2 输出与输入严格同序，而是统一使用稳定唯一的 `record_uid` 做断点恢复和去重。

```
record_uid = f"{input_shard}:{input_index:08d}"
```

其中：
- `input_shard`：由 `output_shard_name_template` 生成的 shard 名
- `input_index`：该记录在原始输入文件中的 0-based 行号

这样做的好处：
- 不依赖网页 URL 是否唯一
- 多输入文件场景下天然稳定
- 两个阶段都可以直接 append-only 写最终文件
- 中途停止后，只要扫描已有 JSONL 里的 `record_uid`，就能跳过已完成样本

**Resume 规则**：
- run 级别仍用 `manifest.json` 固定“输入 shard → 输出 shard”映射，防止错误 resume 到另一批数据
- shard 内 resume 时扫描已有 keep / reject / output 文件，建立 `{record_uid → record}` 映射
- `--no-resume` 时，清空该 shard 现有输出后重新跑

**为什么保留 `id` 和 `_meta`**：
- `id` 仍保留原始网页 id / url，方便下游按网页语义使用
- `_meta` 继续原样保留，方便回溯 `final_url`、`page_type`、`crawl_time` 等信息
- 但断点恢复和去重不再依赖它们，而是依赖 `record_uid`

---

## 4. 目录结构

```
html_rewrite/
├── __init__.py
├── config.py                 # HtmlRewriteConfig dataclass + load_config()
├── stage1_preprocess.py      # Stage 1 批量预处理引擎
├── stage2_rewrite.py         # Stage 2 批量模型改写引擎
├── main.py                   # CLI 入口 (--stage preprocess|rewrite|all)
├── demo.py                   # 单条 debug 工具
├── run_layout.py             # 输入 shard 展开、run 目录与 manifest 管理
├── README.md                 # 使用说明
├── preprocess/
│   ├── __init__.py           # 导出 preprocess(html, cfg) -> (str, PreprocessStats)
│   ├── preprocessor.py       # 主编排器
│   ├── media.py              # 媒体路径替换
│   ├── scripts.py            # inline script / JSON payload 截断
│   ├── styles.py             # inline <style> 截断
│   ├── forms.py              # hidden input value 截断
│   ├── comments.py           # HTML comment 截断
│   ├── formatter.py          # HTML 格式标准化
│   ├── filtering.py          # Stage 1 keep/reject gate
│   ├── analysis.py           # Stage 1 汇总统计 + 分布图输出
│   ├── parser.py             # lxml 依赖检查与统一解析入口
│   ├── text.py               # 可见文本 / 声明语言抽取
│   ├── language.py           # 英文主语言检测与过滤
│   └── stats.py              # PreprocessStats dataclass
├── prompts/
│   ├── __init__.py
│   └── html_rewrite.py       # SYSTEM_PROMPT + build_user_content()
└── configs/
    └── default_local.yaml
```

---

## 5. 配置（HtmlRewriteConfig）

```yaml
# ── API
url: "http://localhost:8000/v1/chat/completions"
api_key: "your-key"
model: "your-model"
timeout: 120.0
max_retries: 3
ssl_verify: true
log_user: "html_rewrite"

# ── 路径（推荐：模板展开 + run 级分片输出）
input_dir: "/path/to/raw_dir"
input_filename_template: "part-{index:05d}.jsonl"
input_start_index: 0
input_end_index_exclusive: 100
run_root_dir: "/path/to/run_20260501"
output_shard_name_template: "part-{index:05d}"

# 兼容旧模式：单文件或显式列表输入（二选一；不要与上面的模板模式混用）
# input_path: "/path/to/raw.jsonl"
# input_paths:
#   - "/path/to/part-00000.jsonl"
#   - "/path/to/part-00001.jsonl"
#
# 旧模式输出路径（仅当未设置 run_root_dir 时使用）
preprocessed_path: "/path/to/preprocessed.jsonl"
output_path: "/path/to/output.jsonl"
call_log_path: "logs/api_calls.jsonl"
stats_log_path: "logs/preprocess_stats.jsonl"
reject_log_path: "logs/preprocess_rejects.jsonl"
summary_log_path: "logs/preprocess_summary.json"
stats_plot_dir: "logs/preprocess_plots"
stats_hist_bins: 120

# ── 预处理阈值（对应规范固定值，可按实际分布调整）
inline_script_max_chars: 4096
json_payload_max_chars: 4096
hidden_input_max_chars: 4096
html_comment_max_chars: 1024
inline_style_max_chars: 32768
max_preprocessed_chars: 65536
min_preprocessed_chars: 1024
fetch_media_size: false

# ── 英文主语言过滤（已实现）
enable_language_filter: true
allowed_languages: ["en"]
language_detector: "langid"
language_min_visible_text_chars: 200
language_min_letter_chars: 100
language_sample_max_chars: 12000
language_min_latin_ratio: 0.6
language_min_detector_margin: 3.0

# ── 生成参数
generation_params: {}
prompt_module: "html_rewrite"

# ── 运行时
num_workers: 16
resume: true
```

规则：
- 推荐模式下，输入路径由 `input_dir / input_filename_template.format(index=i)` 自动展开
- `input_start_index` 为包含式，`input_end_index_exclusive` 为不包含式
- `input_path`、`input_paths`、模板展开模式三者只能选一种，避免歧义
- 若 `run_root_dir` 非空，则启用 run 级 shard 输出模式：
  - Stage 1 写 `run_root_dir/stage1/<shard>/...`
  - Stage 2 写 `run_root_dir/stage2/<shard>/...`
  - 同时维护 `run_root_dir/manifest.json` 与 `run_root_dir/aggregate/*.json`
- 若未设置 `run_root_dir`，则回退到旧单文件输出模式，使用 `preprocessed_path`、`output_path` 等字段
- 不论是 `input_paths` 还是模板展开模式，处理顺序都严格按 shard 列表顺序执行
- `--limit` 仍基于拼接后的全局输入顺序
- `demo --index` 在 `preprocess` 阶段表示原始输入第 `N` 条；Stage 1 / Stage 2 输出文件本身不再保证与输入同序
- `stats_hist_bins` 控制 Stage 1 直方图 bins 数；默认建议 `120`，如果分布仍不够清楚可以继续调大
- 每个 shard 的 Stage 1 `summary.json`，以及 run 级 `aggregate/stage1_summary.json` / `aggregate/stage2_summary.json`，都会输出 `record_uid_check`：
  - `total_records`
  - `unique_record_uids`
  - `duplicate_record_uids`
  - `duplicate_records`
  - `duplicate_samples`
- 如果某个 shard 存在重复记录，aggregate summary 还会单独列出 `shards_with_duplicates`

---

## 6. 预处理规范对照表

| 处理项 | 阈值 | 替换方式 |
|--------|------|----------|
| 媒体路径（img/video/audio/iframe/embed/object/url()/base64） | 全部替换 | `__MEDIA_PLACEHOLDER__/media__width{W}__height{H}.ext` |
| inline script 内容 | > 4096 chars | 清空内容，加 `data-inline-script-truncated="true" data-original-chars="{N}"` |
| JSON/hydration payload | > 4096 chars | 清空内容，加 `data-json-payload-truncated="true" data-original-chars="{N}"` |
| inline `<style>` 内容 | > 32768 chars | 替换为注释占位，加 `data-inline-style-truncated="true"` 属性 |
| hidden input value | > 4096 chars | value 替换为 `__LONG_HIDDEN_VALUE_TRUNCATED_CHARS_{N}__` |
| HTML comment | > 1024 chars | 替换为 `<!-- original comment truncated, chars={N} -->` |
| HTML 格式 | 全部 | BeautifulSoup lxml 解析 + prettify + 去连续多余空行 |

不处理的内容（与规范一致）：外部 script src、外部 CSS link、class/id/data-*、nav/sidebar/footer、可见表单字段、DOM 结构。

补充说明：
- 上表仍属于“单点预处理规则”；完成这些规则后，还应执行一次“样本级过滤”，决定该样本是否进入 Stage 2。
- 样本级过滤看的是**整条预处理后 HTML** 的整体质量，而不是某一个 `<script>` / `<style>` 是否超阈值。

---

## 7. 媒体 placeholder 详细设计

### 7.1 Placeholder 格式

```
__MEDIA_PLACEHOLDER__/media__width{W}__height{H}.ext
```

示例：
- `__MEDIA_PLACEHOLDER__/media__width640__height480.jpg`
- `__MEDIA_PLACEHOLDER__/media__widthunknown__heightunknown.png`
- `__MEDIA_PLACEHOLDER__/media__widthunknown__heightunknown.media`（无法识别扩展名）

### 7.2 宽高获取策略

优先级：**① 标签属性 → ② base64 头部解析 → ③ unknown**

| 来源 | 实现 | 适用场景 |
|------|------|----------|
| 标签 width/height 属性 | 直接读取 `tag.get("width")` | img/video/embed 等已声明尺寸 |
| base64 PNG 头部解析 | stdlib `struct`，解析 IHDR chunk（字节 16-24）| data URI 图片 |
| base64 JPEG 头部解析 | stdlib `struct`，扫描 SOF0/SOF1/SOF2 marker | data URI JPEG |
| unknown | 无法获取任何尺寸时 | 外链图片（不下载）|

**不下载任何媒体资源**（`fetch_media_size=false` 默认关闭），原因见第 7.3 节。

### 7.3 是否下载图片获取尺寸（fetch_media_size）

**默认关闭（false）**，理由：

| 问题 | 说明 |
|------|------|
| 网络稳定性 | 真实网站图片链接大量失效、限流、CDN 跨域拒绝、需要 Cookie |
| 速度 | 每条 HTML 平均 50-200 个媒体资源，全量下载严重拖慢 Stage 1 |
| 离线性 | Stage 1 设计为纯离线可复现，不引入网络不确定性 |
| 必要性 | 后续模型主要关注资源角色和位置，精确像素尺寸收益有限 |
| 覆盖率 | 真实数据中约 60-70% 的 img 没有 width/height 属性，下载也未必能批量解决 |

**如何开启**（`fetch_media_size: true`）：

当前代码预留了 `cfg.fetch_media_size` 字段，若开启，`preprocess/media.py` 应在标签无 width/height 属性时，对外链图片发起 HEAD 请求或下载前若干字节（建议 4KB），尝试从 Content-Length / 图片头部解析尺寸。约束：
- 仅对图片（img/source），不对视频、音频、iframe 开启
- 短 timeout（建议 3-5s）
- 最大下载字节数限制（建议 8KB，覆盖 PNG IHDR + JPEG SOF）
- 失败静默忽略，不影响主流程
- 不保存文件
- 统计 fetch_attempted / fetch_ok / fetch_failed / fetch_timeout

实现建议：在 `media.py` 的 `_make_placeholder()` 中，当 `tag_w/tag_h` 均为 None 且非 base64 时，调用独立的 `_fetch_image_size(url, timeout, max_bytes)` 函数。

### 7.4 扩展名提取

**外链资源**：从 URL path 提取（`urlparse(url).path`），去 query string 后取最后一个 `.` 后缀，与白名单比对：

```
.jpg .jpeg .png .webp .gif .svg .mp4 .webm .mp3 .wav .ogg .pdf
```

无法识别时用 `.media`。

**base64 data URI**：从 `data:{mime};...` 中提取 mime type，映射表：

```
image/jpeg → .jpg   image/png → .png   image/webp → .webp
image/gif  → .gif   image/svg+xml → .svg
video/mp4  → .mp4   video/webm → .webm
audio/mpeg → .mp3   audio/wav → .wav   audio/ogg → .ogg
application/pdf → .pdf
其余 → .media
```

### 7.5 srcset 处理

`<img srcset="...">` 整个属性替换为单个 placeholder，取 srcset 中第一个 URL 的扩展名。有 src 的 img 同时处理 src；无 src 只有 srcset 的 img 也正确处理。

---

## 8. 数据流与中间文件格式

### Stage 1 输出（preprocessed JSONL）

说明：该文件现在是 append-only、无序输出；下游不要按行号与原始输入对齐，而要按 `record_uid` 或 `id` 使用。

```json
{
  "record_uid": "part-00000:00001234",
  "id": "https://example.com/",
  "input_shard": "part-00000",
  "input_index": 1234,
  "source_input_path": "/path/to/raw_dir/part-00000.jsonl",
  "_meta": {
    "url": "https://example.com/",
    "final_url": "https://example.com/",
    "crawl_time": 1711152000,
    "page_type": ["HOME_PAGE"],
    "part": "part2026-03-23-00000",
    "crawl_type": "..."
  },
  "preprocessed_html": "<!DOCTYPE html><html>...</html>",
  "preprocess_stats": {
    "original_chars": 114017,
    "cleaned_chars": 65456,
    "visible_text_chars": 18240,
    "compression_ratio": 0.5741,
    "media": {
      "total": 104, "images": 86, "videos": 0, "audios": 0, "iframes": 18,
      "base64": 0, "regular": 104, "with_size": 38, "without_size": 66, "replaced": 104,
      "fetch_attempted": 0, "fetch_ok": 0, "fetch_failed": 0, "fetch_timeout": 0
    },
    "scripts": { "external": 19, "inline_total": 1, "inline_truncated": 0, "inline_chars": [312] },
    "json_payloads": { "total": 0, "truncated": 0, "chars": [] },
    "styles": { "external_links": 5, "inline_total": 3, "inline_truncated": 0, "inline_chars": [284, 1200, 890] },
    "hidden_inputs": { "total": 0, "truncated": 0, "value_chars": [] },
    "comments": { "total": 3, "truncated": 0, "chars": [42, 15, 8] },
    "formatter": {
      "parse_ok": true,
      "node_count_before": 1504, "node_count_after": 1596,
      "tag_counts_before": { "img": 86, "script": 20, "nav": 2, ... },
      "tag_counts_after":  { "img": 86, "script": 20, "nav": 2, ... }
    },
    "language": {
      "declared_lang": "en",
      "detected_lang": "en",
      "detected_lang_score": -92.1314,
      "detected_lang_top2": "la",
      "detected_lang_top2_score": -98.4421,
      "detector_margin": 6.3107,
      "sample_text_chars": 12000,
      "letter_chars": 14221,
      "latin_letter_chars": 14180,
      "latin_ratio": 0.9971,
      "passed": true,
      "reason": "allowed_language"
    }
  }
}
```

### Stage 1 reject 输出（preprocess_rejects JSONL）

```json
{
  "record_uid": "part-00000:00004567",
  "id": "https://example.com/very-long-article",
  "input_shard": "part-00000",
  "input_index": 4567,
  "source_input_path": "/path/to/raw_dir/part-00000.jsonl",
  "_meta": {
    "url": "https://example.com/very-long-article",
    "final_url": "https://example.com/very-long-article",
    "page_type": ["ARTICLE"]
  },
  "reject_reason": "too_long_after_preprocess",
  "preprocess_stats": {
    "original_chars": 182341,
    "cleaned_chars": 88452,
    "visible_text_chars": 56021,
    "compression_ratio": 0.4851
  }
}
```

reject 文件只保留回溯所需字段，不写入完整 `preprocessed_html`，避免把本来要省下的超长内容再次写回磁盘。

### Stage 1 reject 原因明细（`*_reject_reasons.json`）

reject summary 不再使用 `p50/p90/p95` 这类摘要描述失败样本，而是按 reason 直接列出规则阈值和失败记录，便于排查环境差异或规则误杀。

```json
{
  "total_rejected": 3,
  "reasons": {
    "too_long_after_preprocess": {
      "count": 3,
      "threshold_field": "max_preprocessed_chars",
      "threshold_value": 65536,
      "records": [
        {
          "record_uid": "part-00000:00004567",
          "id": "https://example.com/a",
          "input_shard": "part-00000",
          "input_index": 4567,
          "source_input_path": "/path/to/raw_dir/part-00000.jsonl",
          "actual_cleaned_chars": 80958,
          "visible_text_chars": 4104,
          "original_chars": 121004,
          "details": {
            "threshold_field": "max_preprocessed_chars",
            "threshold_value": 65536,
            "actual_cleaned_chars": 80958
          }
        }
      ]
    }
  }
}
```

### Stage 2 输出（output JSONL）

在 Stage 1 字段基础上新增：

```json
{
  "record_uid": "part-00000:00001234",
  "id": "https://example.com/",
  "input_shard": "part-00000",
  "input_index": 1234,
  "source_input_path": "/path/to/raw_dir/part-00000.jsonl",
  "_meta": { "url": "...", "final_url": "...", "page_type": ["HOME_PAGE"] },
  "output_html": "<!DOCTYPE html>...",
  "model": "your-model",
  "prompt_tokens": 12000,
  "completion_tokens": 8000,
  "finish_reason": "stop"
}
```

---

## 9. Stage 1 样本级过滤建议

### 9.1 为什么要在 Stage 1 末尾加过滤

需要。只做局部截断还不够，因为很多页面即使去掉超长脚本和 payload，仍然会因为正文极长、表格极长或 DOM 过深，导致 `preprocessed_html` 依然非常大，直接推高 Stage 2 的 prompt tokens 和成本。

因此更合理的策略是：

1. 先完成所有最小预处理；
2. 再基于最终 `preprocessed_html` 做一次样本级 gate；
3. keep 样本进入 Stage 2，reject 样本单独记日志，不静默丢弃。

### 9.2 推荐的必选 gate

**A. 超长 gate（必选）**

- 判断字段：`cleaned_chars`
- 默认阈值：`max_preprocessed_chars = 65536`
- 触发动作：不写入 `preprocessed.jsonl`，而是写入 `preprocess_rejects.jsonl`
- reject reason：`too_long_after_preprocess`

为什么先推荐 64K：
- 这是一个简单、稳定、与模型无关的硬阈值，足够先挡掉最极端样本；
- 对 HTML 这种“标签 + 文本”混合内容，64K chars 往往已经对应相当高的 token 成本；
- 作为第一版阈值，它比直接按原始 HTML 长度过滤更合理，因为前面的截断可能已经显著缩小输入。

但 64K 不是通用最优值。更稳妥的经验是：
- 如果 Stage 2 使用的是 **32K context** 模型，建议把阈值收紧到 `32K-48K chars`；
- 如果使用的是 **128K context** 模型，`64K chars` 通常可以作为保守起点；
- 真正最终值应结合你们实际 `prompt_tokens` 分布回调。

**B. 过空 gate（建议也默认开启）**

- 判断字段：`cleaned_chars`
- 默认阈值：`min_preprocessed_chars = 1024`
- 触发动作：reject
- reject reason：`too_short_after_preprocess`

目的：
- 过滤只有壳子、跳转页、错误页、极短占位页或几乎无内容的样本；
- 避免 Stage 2 花成本改写低价值页面。

**C. 结构异常 gate（必选）**

- 判断条件：格式化后缺少有效 HTML 骨架，例如无 `<html>` / 无 `<body>` / 序列化结果为空
- 触发动作：reject
- reject reason：`invalid_or_empty_html`

目的：
- 防止解析异常、极脏输入或清洗后空文档进入 Stage 2。

### 9.3 推荐但可以第二步再做的 gate

**D. 模型预算 gate（强烈建议）**

如果后续愿意在 Stage 1 增加一个轻量 tokenizer 估算，可以再加：

- 判断字段：`estimated_prompt_tokens`
- 默认策略：`estimated_prompt_tokens > model_context_limit * 0.45 ~ 0.60` 时 reject
- reject reason：`prompt_budget_exceeded`

这样会比单纯按 chars 更稳，因为不同语言和不同 HTML 结构的 token/chars 比例差异很大。  
如果暂时不想引入 tokenizer，先只做 `max_preprocessed_chars` 也完全合理。

**E. 英文主语言 gate（已实现）**

目标不是判断“这个域名是否是英文站”，而是判断**当前页面的主要可见文本是否以英文为主**。  
因此应按页面过滤，而不是按站点域名过滤。

#### 为什么不建议只用规则

纯规则方法（只看 `html lang`、URL 中的 `/en/`、域名后缀、ASCII 比例）实现简单，但鲁棒性不足：

- `html lang` 经常缺失、乱填，或模板页统一写死；
- 多语言站点同一域名下会混有英文页和非英文页；
- 页面可能主要是英文正文，但导航、cookie banner、页脚混入多语；
- 仅看 ASCII / 拉丁字母比例，会把法语、德语、西班牙语等也误判成英文。

规则信号适合作为辅助，不适合作为主判据。

#### `langid.py` 是什么

`langid.py` 不是纯规则方法，也不是需要联网调用的大模型。  
它是一个**内置预训练语言识别模型**（传统统计模型，基于字符 n-gram / 朴素贝叶斯风格方法），随 Python 包一起分发：

- 运行时不需要再下载模型；
- 不需要 GPU；
- 推理成本低，适合放在 Stage 1 批量过滤；
- 输入一段文本，输出预测语言和一个相对分数。

工程上只需要安装 `langid>=1.1.6` 这个 Python 包，不需要额外下载模型文件。

需要注意：
- 它输出的 score 不是严格校准后的概率；
- 因此更稳妥的做法不是只看 `classify()` 的单个分数，而是比较 top1 / top2 的差距（margin）。

#### 当前实现方案

推荐采用“**规则信号 + 轻量模型**”的多信号方案：

1. 从预处理后的 DOM 中抽取 `visible_text`
2. 若 `visible_text_chars < language_min_visible_text_chars`，直接 reject  
   reject reason：`language_detection_insufficient_text`
3. 统计字母字符数；若字母数 `< language_min_letter_chars`，直接 reject  
   reject reason：`language_detection_insufficient_letters`
4. 做快速脚本检查：计算拉丁字母占全部字母的比例  
   若 `< language_min_latin_ratio`，直接 reject  
   reject reason：`language_not_mainly_latin_script`
5. 收集声明性信号：
   - `<html lang>`
   - `content-language`
   - `og:locale`
6. 对可见文本做采样后送入 `langid`：
   - 不用整页全文，建议截取前/中/后拼接，总长最多 `language_sample_max_chars`
   - 这样比只取开头更不容易被导航和页头污染
7. 取 `langid.rank(text)` 的 top1 / top2：
   - 若 `top1 != "en"`，reject  
     reject reason：`non_english_page`
   - 若 `top1 == "en"` 但 `top1_score - top2_score < language_min_detector_margin`，reject  
     reject reason：`language_detection_low_margin`
8. 若声明性信号明确是非英文，且 detector 也不是 `en`，直接 reject
9. 最终只保留 `detected_lang == "en"` 且 margin 足够的页面

#### 为什么这套方案更鲁棒

- 先用 `visible_text`，避免让 HTML 标签、脚本和样式干扰检测；
- 先做最便宜的“文本足够长 / 字母足够多 / 拉丁脚本占比”过滤，减少 detector 的误判；
- 再用 `langid` 做主判据，避免纯规则误杀；
- 用 top1/top2 margin，而不是迷信单个 score；
- 用“前中后采样”，避免整页被单一导航区域主导。

#### 推荐初始阈值

- `language_min_visible_text_chars = 200`
- `language_min_letter_chars = 100`
- `language_sample_max_chars = 12000`
- `language_min_latin_ratio = 0.6`
- `language_min_detector_margin = 3.0`

这些值适合作为第一版默认值，后续根据 reject 明细和抽样质检再调。

#### 当前写出的统计字段

当前会在 `preprocess_stats.language` 中记录：

- `declared_lang`
- `detected_lang`
- `detected_lang_score`
- `detected_lang_top2`
- `detected_lang_top2_score`
- `detector_margin`
- `sample_text_chars`
- `letter_chars`
- `latin_letter_chars`
- `latin_ratio`
- `passed`
- `reason`

### 9.4 reject 文件格式建议

```json
{
  "record_uid": "part-00000:00004567",
  "id": "https://example.com/very-long-article",
  "input_shard": "part-00000",
  "input_index": 4567,
  "source_input_path": "/path/to/raw_dir/part-00000.jsonl",
  "_meta": { "url": "...", "final_url": "...", "page_type": ["ARTICLE"] },
  "reject_reason": "too_long_after_preprocess",
  "preprocess_stats": {
    "original_chars": 182341,
    "cleaned_chars": 88452,
    "compression_ratio": 0.4851
  }
}
```

关键点：
- reject 文件必须保留 `record_uid`、`id`、`_meta`、`reject_reason` 和核心 stats；
- 不建议把完整 `preprocessed_html` 再写进 reject 文件，否则会把“省 tokens”的收益又写回磁盘；
- keep / reject 都要统计数量，方便看过滤比例是否异常。

## 10. 统计分析建议

Stage 1 完成后，可通过每个 shard 的 `stats.jsonl`、`summary.json`、`plots/`，以及 run 级 `aggregate/stage1_summary.json` 分析阈值合理性，重点关注：

```python
import json, statistics

stats = [json.loads(l)["stats"] for l in open("run_root_dir/stage1/part-00000/stats.jsonl")]

# 压缩比分布
ratios = [s["compression_ratio"] for s in stats]
print(f"压缩比 median={statistics.median(ratios):.2f}  p90={sorted(ratios)[int(len(ratios)*0.9)]:.2f}")

# 清洗后长度分布（决定 max_preprocessed_chars）
cleaned = [s["cleaned_chars"] for s in stats]
print(f"cleaned_chars p50={sorted(cleaned)[len(cleaned)//2]:,}  p90={sorted(cleaned)[int(len(cleaned)*0.9)]:,}  max={max(cleaned):,}")

# 媒体资源宽高覆盖率
with_sz = sum(s["media"]["with_size"] for s in stats)
total_m = sum(s["media"]["total"] for s in stats)
print(f"有宽高的媒体 {with_sz}/{total_m} = {with_sz/total_m:.1%}")

# inline script 长度分布（判断 4096 阈值是否合适）
all_script_chars = [c for s in stats for c in s["scripts"]["inline_chars"]]
print(f"inline script 长度 p50={sorted(all_script_chars)[len(all_script_chars)//2]:,}  max={max(all_script_chars):,}")
```

每个 shard 的 `summary.json` 建议至少包含：

```json
{
  "total_input": 10000,
  "kept": 9123,
  "rejected": 877,
  "reject_reasons": {
    "too_long_after_preprocess": 612,
    "too_short_after_preprocess": 201,
    "invalid_or_empty_html": 64
  },
  "cleaned_chars": { "p50": 8421, "p90": 31104, "p95": 48762, "p99": 70211 },
  "compression_ratio": { "p50": 0.61, "p90": 0.94 }
}
```

同时输出 `summary_reject_reasons.json`，专门用于排查“为什么这一批全被 reject”这类问题。

建议同时输出这些图，便于直接观察阈值位置，而不是只看分位数：

- `01_original_chars_hist.png`
- `02_cleaned_chars_hist.png`，标出 `min_preprocessed_chars` / `max_preprocessed_chars`
- `03_visible_text_chars_hist.png`
- `04_compression_ratio_hist.png`
- `05_inline_script_chars_hist.png`
- `06_json_payload_chars_hist.png`
- `07_inline_style_chars_hist.png`
- `08_hidden_input_chars_hist.png`
- `09_html_comment_chars_hist.png`
- `10_cleaned_chars_keep_vs_reject.png`
- `11_reject_reason_counts.png`

如果统计显示阈值偏高/偏低，修改配置后 `--no-resume` 重跑 Stage 1 即可。

---

## 11. 与 distillation 模块的关系

| 组件 | distillation/ 来源 | html_rewrite/ 用法 |
|------|-------------------|-------------------|
| API 调用 + retry | `utils/api_client.call_chat_completion()` | Stage 2 直接复用 |
| 并发模型 | `distill.py:ThreadPoolExecutor` | Stage 1 / Stage 2 均使用，改为 append-only 写出 |
| Resume | `distill.py:done_ids 扫描` | 升级为：扫描已有 JSONL 中的 `record_uid` 去重 |
| Config 加载 | `config.py:load_config(Path)` | 镜像实现，新增预处理阈值字段 |
| Prompt 模块接口 | `prompts/svg.py: SYSTEM_PROMPT + build_user_content()` | `prompts/html_rewrite.py` 遵循同一接口 |
| 日志约定 | `[distill]` tag | 改为 `[preprocess]` / `[rewrite]` |

**distillation 没有的新增内容**：
- `preprocess/` 子模块（媒体替换、截断、格式化、统计）
- Stage 1/2 解耦中间文件
- `record_uid` 驱动的 append-only / resume 机制
- 新依赖：`beautifulsoup4 + lxml`

---

## 12. 运行命令速查

```bash
# Stage 1：预处理（离线，可重复运行调参）
python -m html_rewrite.main --config html_rewrite/configs/default_local.yaml --stage preprocess
# 产物：run_root_dir/manifest.json + run_root_dir/stage1/<shard>/... + run_root_dir/aggregate/stage1_summary.json

# 调整阈值后强制重跑（忽略已有输出）
python -m html_rewrite.main --config html_rewrite/configs/default_local.yaml --stage preprocess --no-resume

# 仅处理前 5 条（快速验证）
python -m html_rewrite.main --config html_rewrite/configs/default_local.yaml --stage preprocess --limit 5

# Stage 2：模型改写
python -m html_rewrite.main --config html_rewrite/configs/default_local.yaml --stage rewrite

# 全流程
python -m html_rewrite.main --config html_rewrite/configs/default_local.yaml --stage all

# 单条 debug
python -m html_rewrite.demo --config html_rewrite/configs/default_local.yaml --stage preprocess --index 0
python -m html_rewrite.demo --config html_rewrite/configs/default_local.yaml --stage rewrite --index 0
```

当使用 `input_paths` 或模板展开模式时：
- `--limit N` 表示只处理拼接后前 `N` 条
- `demo --index N` 在 `preprocess` 阶段也是按拼接后的总顺序取第 `N` 条
- Stage 1 / Stage 2 输出文件本身不再保证与原始输入同序

---

## 13. 新增依赖

```
beautifulsoup4>=4.12    # HTML 解析和序列化
lxml>=5.0              # BeautifulSoup 后端，比 html.parser 更鲁棒（处理真实脏 HTML）
matplotlib>=3.8        # Stage 1 分布图输出
langid>=1.1.6          # 英文主语言检测；包内自带轻量模型，无需额外下载文件
```

其余依赖（loguru、requests、pyyaml）沿用项目现有版本。
