# html_rewrite 模块设计文档

## 1. 背景与目标

**目标**：从真实网站 HTML 批量生产"dirty → clean HTML"改写训练数据。

**输入**：FineWebEdu JSONL，每条含原始网页 `html` 及元信息（`url`、`final_url`、`crawl_time`、`page_type`、`part` 等）。

**输出**：JSONL，每条包含预处理后 HTML、模型改写的 clean HTML、预处理统计及原始 meta 信息，两阶段输出与输入严格同序、可逐行对应。

**核心设计原则**：
- 预处理只做最小化处理（路径替换、超长截断、格式标准化），不重写结构、不删除 class/nav/sidebar/footer
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
  └─ 记录详细预处理统计
  │
  ▼  (中间文件，可调整阈值后 --no-resume 重跑)
preprocessed JSONL  +  preprocess_stats JSONL
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

两个阶段通过中间文件完全解耦，各自支持独立 resume。Stage 1 无网络依赖，Stage 2 读中间文件、不重跑预处理。

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
- Resume 时读取已有输出文件，建立 `{id → record}` 映射
- 只处理未完成的 `todo` 记录（仍记录原始索引）
- 全部处理完成后，将已有结果和新结果合并成 `{orig_idx → record}`，统一排序写出
- 已有结果被重新包含进有序输出，最终文件始终是完整有序的

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

# ── 预处理阈值（对应规范固定值，可按实际分布调整）
inline_script_max_chars: 4096
json_payload_max_chars: 4096
hidden_input_max_chars: 4096
html_comment_max_chars: 1024
inline_style_max_chars: 32768
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

## 9. 统计分析建议

Stage 1 完成后，通过 `stats_log_path`（逐条统计 JSONL）分析阈值合理性，重点关注：

```python
import json, statistics

stats = [json.loads(l)["stats"] for l in open("logs/preprocess_stats.jsonl")]

# 压缩比分布
ratios = [s["compression_ratio"] for s in stats]
print(f"压缩比 median={statistics.median(ratios):.2f}  p90={sorted(ratios)[int(len(ratios)*0.9)]:.2f}")

# 媒体资源宽高覆盖率
with_sz = sum(s["media"]["with_size"] for s in stats)
total_m = sum(s["media"]["total"] for s in stats)
print(f"有宽高的媒体 {with_sz}/{total_m} = {with_sz/total_m:.1%}")

# inline script 长度分布（判断 4096 阈值是否合适）
all_script_chars = [c for s in stats for c in s["scripts"]["inline_chars"]]
print(f"inline script 长度 p50={sorted(all_script_chars)[len(all_script_chars)//2]:,}  max={max(all_script_chars):,}")
```

如果统计显示阈值偏高/偏低，修改配置后 `--no-resume` 重跑 Stage 1 即可。

---

## 10. 与 distillation 模块的关系

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

## 11. 运行命令速查

```bash
# Stage 1：预处理（离线，可重复运行调参）
python -m html_rewrite.main --config html_rewrite/configs/default_local.yaml --stage preprocess

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

## 12. 新增依赖

```
beautifulsoup4>=4.12    # HTML 解析和序列化
lxml>=5.0              # BeautifulSoup 后端，比 html.parser 更鲁棒（处理真实脏 HTML）
```

其余依赖（loguru、requests、pyyaml）沿用项目现有版本。
