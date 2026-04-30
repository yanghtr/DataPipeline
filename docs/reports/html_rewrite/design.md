# html_rewrite 模块设计文档

## 1. 背景与目标

**目标**：从真实网站 HTML 批量生产"dirty → clean HTML"改写训练数据。

**输入**：FineWebEdu JSONL，每条含原始网页 `html` 及元信息（`url`、`final_url`、`crawl_time`、`page_type`、`part` 等）。

**输出**：以“可进入 Stage 2 的有效样本”为主输出，JSONL 每条包含预处理后 HTML、模型改写的 clean HTML、预处理统计及原始 meta 信息；同时额外输出 Stage 1 reject 日志和汇总统计，便于调参与回溯。

**核心设计原则**：
- 预处理只做最小化处理（路径替换、超长截断、格式标准化），不重写结构、不删除 class/nav/sidebar/footer
- 样本过滤放在 Stage 1 最后，只决定“是否进入 Stage 2”，不回写或重构 HTML 结构
- 模型负责真正的语义改写
- 两阶段解耦：Stage 1（离线预处理）可单独重跑调参，Stage 2（模型改写）不重跑预处理
- 工程上复用 `distillation/` 已验证的并发/retry/resume/日志/结果保存模式

---

## 2. 整体架构

```
raw JSONL
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
preprocessed JSONL  +  preprocess_stats JSONL  +  preprocess_rejects JSONL  +  preprocess_summary JSON
  │
  ▼
[Stage 2: Rewrite]                   html_rewrite/stage2_rewrite.py
  ├─ 并发调用 OpenAI-compatible 模型
  ├─ 提取模型输出中的 clean HTML
  └─ retry / resume / 有序写出
  │
  ▼
output JSONL  (含 meta + preprocessed_html + output_html + preprocess_stats)
```

两个阶段通过中间文件完全解耦，各自支持独立 resume。Stage 1 无网络依赖，Stage 2 只读取 Stage 1 的 keep 样本，不重跑预处理。reject 样本不进入 Stage 2，但必须保留日志，避免“静默丢数”。

补充约束：
- `lxml` 是 Stage 1 的硬依赖。
- 如果当前环境缺少 `lxml`，应直接报错停止，不允许静默 fallback 到 `html.parser` 或其它 parser。

---

## 3. 顺序保证机制

**问题**：`ThreadPoolExecutor` 的 `as_completed` 按完成顺序返回，导致输出无序，两阶段结果无法逐行对应。

**方案**：并发计算 + 按原始索引排序 + 原子写出。

```
for i, rec in enumerate(records):          # 保留原始索引
    futures.append(exe.submit(process, i, rec))

# 结果收集进 dict {orig_idx: result}（线程安全）
# 全部完成后，sorted(results_by_idx.keys()) 写出
# 写入临时文件 .tmp，os.replace() 原子覆盖目标文件
```

**Resume 与顺序兼容**：
- Resume 时读取已有 keep 输出文件，建立 `{id → record}` 映射
- 如启用 reject 输出，也读取已有 reject 文件，建立 `{id → reject_info}` 映射
- 只处理未完成的 `todo` 记录（仍记录原始索引）
- 全部处理完成后，将已有结果和新结果合并成 `{orig_idx → record}`，统一排序写出
- keep 文件与 reject 文件都各自保持原始顺序；Stage 2 只消费 keep 文件

**原子性**：写入 `.tmp` 文件，`os.replace()` 原子覆盖，不会产生半完成文件。

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

# ── 路径
input_path: "/path/to/raw.jsonl"
preprocessed_path: "/path/to/preprocessed.jsonl"   # Stage 1 输出 / Stage 2 输入
output_path: "/path/to/output.jsonl"               # Stage 2 输出
call_log_path: "logs/api_calls.jsonl"
stats_log_path: "logs/preprocess_stats.jsonl"
reject_log_path: "logs/preprocess_rejects.jsonl"
summary_log_path: "logs/preprocess_summary.json"
stats_plot_dir: "logs/preprocess_plots"

# ── 预处理阈值（对应规范固定值，可按实际分布调整）
inline_script_max_chars: 4096
json_payload_max_chars: 4096
hidden_input_max_chars: 4096
html_comment_max_chars: 1024
inline_style_max_chars: 32768
max_preprocessed_chars: 65536
min_preprocessed_chars: 1024
fetch_media_size: false

# ── 生成参数
generation_params: {}
prompt_module: "html_rewrite"

# ── 运行时
num_workers: 16
resume: true
```

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

```json
{
  "id": "https://example.com/",
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
    }
  }
}
```

### Stage 1 reject 输出（preprocess_rejects JSONL）

```json
{
  "id": "https://example.com/very-long-article",
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
          "id": "https://example.com/a",
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

### 9.4 reject 文件格式建议

```json
{
  "id": "https://example.com/very-long-article",
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
- reject 文件必须保留 `id`、`_meta`、`reject_reason` 和核心 stats；
- 不建议把完整 `preprocessed_html` 再写进 reject 文件，否则会把“省 tokens”的收益又写回磁盘；
- keep / reject 都要统计数量，方便看过滤比例是否异常。

## 10. 统计分析建议

Stage 1 完成后，通过 `stats_log_path`（逐条统计 JSONL）、`summary_log_path`（聚合统计 JSON）和 `stats_plot_dir`（PNG 分布图目录）分析阈值合理性，重点关注：

```python
import json, statistics

stats = [json.loads(l)["stats"] for l in open("logs/preprocess_stats.jsonl")]

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

建议额外输出一个 `summary_log_path`，至少包含：

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

同时输出 `*_reject_reasons.json`，专门用于排查“为什么这一批全被 reject”这类问题。

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
| 并发模型 | `distill.py:ThreadPoolExecutor` | Stage 1 / Stage 2 均使用，新增有序写出 |
| Resume | `distill.py:done_ids 扫描` | 升级为：读已有结果 → merge → 有序覆盖写出 |
| Config 加载 | `config.py:load_config(Path)` | 镜像实现，新增预处理阈值字段 |
| Prompt 模块接口 | `prompts/svg.py: SYSTEM_PROMPT + build_user_content()` | `prompts/html_rewrite.py` 遵循同一接口 |
| 日志约定 | `[distill]` tag | 改为 `[preprocess]` / `[rewrite]` |

**distillation 没有的新增内容**：
- `preprocess/` 子模块（媒体替换、截断、格式化、统计）
- Stage 1/2 解耦中间文件
- 有序输出机制（并发 + 排序 + 原子写）
- 新依赖：`beautifulsoup4 + lxml`

---

## 12. 运行命令速查

```bash
# Stage 1：预处理（离线，可重复运行调参）
python -m html_rewrite.main --config html_rewrite/configs/default_local.yaml --stage preprocess
# 产物：preprocessed.jsonl + preprocess_stats.jsonl + preprocess_rejects.jsonl + preprocess_summary.json + preprocess_plots/

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

---

## 13. 新增依赖

```
beautifulsoup4>=4.12    # HTML 解析和序列化
lxml>=5.0              # BeautifulSoup 后端，比 html.parser 更鲁棒（处理真实脏 HTML）
matplotlib>=3.8        # Stage 1 分布图输出
```

其余依赖（loguru、requests、pyyaml）沿用项目现有版本。
