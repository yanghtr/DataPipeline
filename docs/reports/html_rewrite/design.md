# html_rewrite 模块设计文档

## 1. 背景与目标

**目标**：从真实网站 HTML 批量生产"dirty → clean HTML"改写训练数据。

输入：FineWebEdu JSONL，每条含原始网页 `html` 及元信息（`url`、`final_url`、`crawl_time`、`page_type`、`part` 等）。

输出：JSONL，每条包含预处理后 HTML、模型改写的 clean HTML、预处理统计、及原始 meta 信息。

**核心约束**：
- 预处理只做最小化处理（路径替换、超长截断、格式标准化），不重写结构、不删除 class/nav/sidebar/footer
- 模型负责真正的语义改写
- 工程上复用 `distillation/` 已验证的并发/retry/resume/日志/结果保存模式

---

## 2. 整体架构

```
raw JSONL
  ↓
[Stage 1: Preprocess]           html_rewrite/stage1_preprocess.py
  - 媒体路径替换为 placeholder
  - 超长 inline script / JSON payload / style / hidden input / comment 截断
  - HTML 格式标准化（解析+重新序列化）
  - 记录详细预处理统计
  ↓
preprocessed JSONL              (中间文件，可用于调整阈值后重跑)
+ preprocess_stats JSONL        (逐条统计，用于分析阈值合理性)
  ↓
[Stage 2: Rewrite]              html_rewrite/stage2_rewrite.py
  - 并发调用 OpenAI-compatible 模型
  - 提取模型输出中的 clean HTML
  - retry / resume / 流式写出
  ↓
output JSONL                    (含 meta + preprocessed_html + output_html + stats)
```

两个阶段通过中间文件完全解耦，可独立运行、独立 resume。Stage 1 不依赖网络，Stage 2 不重跑预处理。

---

## 3. 目录结构

```
html_rewrite/
├── __init__.py
├── config.py                 # HtmlRewriteConfig dataclass + load_config()
├── stage1_preprocess.py      # Stage 1 批量预处理引擎
├── stage2_rewrite.py         # Stage 2 批量模型改写引擎
├── main.py                   # CLI 入口 (--stage preprocess|rewrite|all)
├── demo.py                   # 单条 debug 工具
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

## 4. 配置（HtmlRewriteConfig）

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

# ── 预处理阈值（对应规范固定值，可覆盖）
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

## 5. 预处理规范对照表

| 处理项 | 阈值 | 替换方式 | 来源规范 §|
|--------|------|----------|-----------|
| 媒体路径（img/video/audio/iframe/embed/object/url()/base64） | 全部替换 | `__MEDIA_PLACEHOLDER__/media__width{W}__height{H}.ext` | §1 |
| inline script 内容 | > 4096 chars | 清空内容，加 `data-inline-script-truncated="true" data-original-chars="{N}"` | §3 |
| JSON/hydration payload | > 4096 chars | 清空内容，加 `data-json-payload-truncated="true" data-original-chars="{N}"` | §4 |
| inline `<style>` 内容 | > 32768 chars | 替换为注释，加属性标记 | §5 |
| hidden input value | > 4096 chars | value 替换为 `__LONG_HIDDEN_VALUE_TRUNCATED_CHARS_{N}__` | §6 |
| HTML comment | > 1024 chars | 替换为 `<!-- original comment truncated, chars={N} -->` | §7 |
| HTML 格式 | 全部 | 解析 + prettify + 去多余空行 | §8 |

**媒体 placeholder 格式细节**：
- 宽高来源：① 标签 width/height 属性 → ② base64 头部解析（PNG/JPEG，stdlib struct）→ ③ `unknown`
- 扩展名：从原始 URL path 提取（去 query string），支持 `.jpg .jpeg .png .webp .gif .svg .mp4 .webm .mp3 .wav .ogg .pdf`；base64 从 mime type 推断；无法识别用 `.media`
- srcset：整个属性替换为单个 placeholder

---

## 6. 中间文件格式

### Stage 1 输出（preprocessed JSONL）

```json
{
  "id": "https://example.com/page",
  "_meta": {
    "url": "https://example.com/page",
    "final_url": "https://example.com/page",
    "crawl_time": 1711152000,
    "page_type": ["HOME_PAGE"],
    "part": "part2026-03-23-00000"
  },
  "preprocessed_html": "<!DOCTYPE html><html>...</html>",
  "preprocess_stats": {
    "original_chars": 114000,
    "cleaned_chars": 48000,
    "compression_ratio": 0.42,
    "media": { "total": 23, "images": 18, "videos": 2, "audios": 0, "iframes": 3, "base64": 1, "with_size": 15, "without_size": 8, "replaced": 23 },
    "scripts": { "external": 5, "inline": 3, "truncated": 1 },
    "json_payloads": { "total": 2, "truncated": 2 },
    "styles": { "external_links": 3, "inline": 2, "truncated": 1 },
    "hidden_inputs": { "total": 4, "truncated": 1 },
    "comments": { "total": 7, "truncated": 2 },
    "formatter": { "parse_ok": true, "node_count_before": 892, "node_count_after": 892 }
  }
}
```

### Stage 2 输出（final output JSONL）

```json
{
  "id": "https://example.com/page",
  "_meta": { "url": "...", "final_url": "...", "crawl_time": ..., "page_type": [...], "part": "..." },
  "preprocessed_html": "...",
  "output_html": "<!DOCTYPE html><html>...</html>",
  "preprocess_stats": { ... },
  "model": "your-model",
  "prompt_tokens": 12000,
  "completion_tokens": 8000,
  "finish_reason": "stop"
}
```

---

## 7. 与 distillation 模块的关系

| 组件 | distillation/ 来源 | html_rewrite/ 用法 |
|------|-------------------|-------------------|
| API 调用 + retry | `utils/api_client.call_chat_completion()` | 直接复用，stage2 调用 |
| 并发模型 | `distill.py:ThreadPoolExecutor` | stage1 / stage2 均使用相同模式 |
| 线程安全写 JSONL | `distill.py:write_lock + append` | 直接复用 |
| Resume | `distill.py:done_ids 扫描` | stage1 按 url 做 id，stage2 同理 |
| Config 加载 | `config.py:load_config(Path)` | 镜像实现 |
| Prompt 模块接口 | `prompts/svg.py: SYSTEM_PROMPT + build_user_content()` | prompts/html_rewrite.py 遵循同一接口 |
| 日志约定 | `[distill]` tag prefix | 改为 `[preprocess]` / `[rewrite]` |

**新增（distillation 没有的）**：
- `preprocess/` 子模块：HTML 解析、媒体替换、截断逻辑
- Stage 1/2 解耦中间文件
- 预处理统计（`PreprocessStats`）
- 新依赖：`beautifulsoup4 + lxml`

---

## 8. 运行方式

```bash
# Stage 1：预处理（离线，无网络依赖）
python -m html_rewrite.main --config html_rewrite/configs/default_local.yaml --stage preprocess

# 检查统计分布，调整阈值后重跑
python -m html_rewrite.main --config html_rewrite/configs/default_local.yaml --stage preprocess --no-resume

# Stage 2：模型改写
python -m html_rewrite.main --config html_rewrite/configs/default_local.yaml --stage rewrite

# 全流程（preprocess → rewrite）
python -m html_rewrite.main --config html_rewrite/configs/default_local.yaml --stage all

# 调试单条（默认取第一条）
python -m html_rewrite.demo --config html_rewrite/configs/default_local.yaml --stage preprocess
python -m html_rewrite.demo --config html_rewrite/configs/default_local.yaml --stage rewrite
```

---

## 9. 新增依赖

在 `requirements.txt` 中添加：

```
beautifulsoup4>=4.12
lxml>=5.0
```
