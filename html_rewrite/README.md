# html_rewrite

将真实网站原始 HTML 批量改写为干净、单文件、可渲染 HTML 的数据生产流水线。

## 流程概述

```
raw JSONL  →  [Stage 1: 预处理 + 过滤 + 统计可视化]  →  preprocessed JSONL  →  [Stage 2: 模型改写]  →  output JSONL
```

两个阶段通过中间文件解耦，可独立运行、独立 resume。输出结果与输入严格同序，两阶段结果可逐行对应比较。

## 快速开始

### 1. 安装依赖

```bash
pip install beautifulsoup4>=4.12 lxml>=5.0 matplotlib>=3.8
# 其余依赖见项目根目录 requirements.txt
```

### 2. 配置文件

默认配置文件路径：

```bash
html_rewrite/configs/default_local.yaml
```

直接编辑这个文件即可；下面的命令也默认使用它。

必填字段：

```yaml
url: "http://localhost:8000/v1/chat/completions"   # OpenAI-compatible API endpoint
api_key: "your-api-key"
model: "your-model-name"
input_path: "/path/to/raw.jsonl"                   # FineWebEdu 原始 JSONL
preprocessed_path: "/path/to/preprocessed.jsonl"  # Stage 1 输出 / Stage 2 输入
output_path: "/path/to/output.jsonl"               # Stage 2 最终输出
```

### 3. 运行

```bash
# Stage 1：仅预处理（离线，无需 API，可调整阈值反复运行）
python -m html_rewrite.main --config html_rewrite/configs/default_local.yaml --stage preprocess

# 检查 logs/html_rewrite_stats.jsonl / logs/html_rewrite_summary.json / logs/html_rewrite_plots/ 后，如需调整阈值：
python -m html_rewrite.main --config html_rewrite/configs/default_local.yaml --stage preprocess --no-resume

# Stage 2：模型改写（需要 API 可用）
python -m html_rewrite.main --config html_rewrite/configs/default_local.yaml --stage rewrite

# 全流程一次跑完
python -m html_rewrite.main --config html_rewrite/configs/default_local.yaml --stage all

# 调试单条（取第 0 条，默认）
python -m html_rewrite.demo --config html_rewrite/configs/default_local.yaml --stage preprocess
python -m html_rewrite.demo --config html_rewrite/configs/default_local.yaml --stage rewrite --index 2
```

Stage 1 跑完后，通常会看到这些文件或目录：

- `preprocessed_path`：仅 keep 样本，供 Stage 2 使用
- `stats_log_path`：逐条统计和 `status/reject_reason`
- `reject_log_path`：被过滤掉的样本
- `summary_log_path`：聚合统计摘要
- `stats_plot_dir`：直方图和 reject 原因柱状图
- `summary_log_path` 同目录下的 `*_reject_reasons.json`：按 reject 原因展开的明细日志

常用调试方式：

```bash
# 先只跑前 5 条，确认过滤和分布图输出正常
python -m html_rewrite.main --config html_rewrite/configs/default_local.yaml --stage preprocess --limit 5

# 查看单条预处理结果和统计
python -m html_rewrite.demo --config html_rewrite/configs/default_local.yaml --stage preprocess --index 0
```

## 配置参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `url` | — | API endpoint（必填） |
| `api_key` | — | API key（必填） |
| `model` | — | 模型名（必填） |
| `input_path` | — | 原始 JSONL 路径（必填） |
| `preprocessed_path` | `preprocessed.jsonl` | Stage 1 输出路径 |
| `output_path` | `html_rewrite_output.jsonl` | Stage 2 输出路径 |
| `call_log_path` | `logs/html_rewrite_api_calls.jsonl` | API 原始调用日志 |
| `stats_log_path` | `logs/html_rewrite_stats.jsonl` | 预处理统计日志 |
| `reject_log_path` | `logs/html_rewrite_rejects.jsonl` | 被 Stage 1 过滤掉的样本日志 |
| `summary_log_path` | `logs/html_rewrite_summary.json` | Stage 1 汇总统计 |
| `stats_plot_dir` | `logs/html_rewrite_plots` | Stage 1 分布图输出目录 |
| `inline_script_max_chars` | `4096` | inline script 截断阈值 |
| `json_payload_max_chars` | `4096` | JSON payload 截断阈值 |
| `hidden_input_max_chars` | `4096` | hidden input value 截断阈值 |
| `html_comment_max_chars` | `1024` | HTML comment 截断阈值 |
| `inline_style_max_chars` | `32768` | inline style 截断阈值 |
| `min_preprocessed_chars` | `1024` | Stage 1 过空 gate，低于该长度直接 reject |
| `max_preprocessed_chars` | `65536` | Stage 1 超长 gate，高于该长度直接 reject |
| `fetch_media_size` | `false` | 是否下载图片头部以获取尺寸（默认关闭） |
| `num_workers` | `16` | 并发线程数 |
| `resume` | `true` | 是否断点续跑（跳过已完成条目） |
| `generation_params` | `{}` | 透传给 API 的生成参数（temperature、max_tokens 等） |
| `prompt_module` | `html_rewrite` | prompts/ 下的 prompt 模块名 |

## 输出格式

### Stage 1 输出（preprocessed JSONL）

每行一条，仅包含通过 Stage 1 gate 的 keep 样本，字段：

```json
{
  "id": "https://example.com/",
  "_meta": { "url": "...", "final_url": "...", "crawl_time": 1711152000, "page_type": ["HOME_PAGE"], "part": "..." },
  "preprocessed_html": "<!DOCTYPE html>...",
  "preprocess_stats": {
    "original_chars": 114000,
    "cleaned_chars": 65000,
    "visible_text_chars": 18240,
    "compression_ratio": 0.57,
    "media": { "total": 104, "images": 86, "videos": 0, "audios": 0, "iframes": 18, "base64": 0, "with_size": 38, "without_size": 66, "replaced": 104 },
    "scripts": { "external": 19, "inline_total": 1, "inline_truncated": 0 },
    "json_payloads": { "total": 0, "truncated": 0 },
    "styles": { "external_links": 5, "inline_total": 3, "inline_truncated": 0 },
    "hidden_inputs": { "total": 0, "truncated": 0 },
    "comments": { "total": 3, "truncated": 0 },
    "formatter": { "parse_ok": true, "node_count_before": 1504, "node_count_after": 1596 }
  }
}
```

`preprocess_stats` 中重点建议关注：

- `cleaned_chars`：最终进入 gate 的 HTML 长度
- `visible_text_chars`：页面可见文本长度，后续如果要加双 gate 会很有用
- `compression_ratio`：判断当前预处理是否真的压缩了上下文
- `scripts/json_payloads/styles/hidden_inputs/comments`：各类局部截断的次数和长度分布

### Stage 1 reject 输出（reject JSONL）

被过滤的样本单独写出，字段：

```json
{
  "id": "https://example.com/very-long-article",
  "_meta": { "url": "...", "final_url": "...", "page_type": ["ARTICLE"] },
  "reject_reason": "too_long_after_preprocess",
  "preprocess_stats": {
    "original_chars": 182341,
    "cleaned_chars": 88452,
    "visible_text_chars": 56021,
    "compression_ratio": 0.4851
  }
}
```

常见 `reject_reason`：
- `too_short_after_preprocess`
- `too_long_after_preprocess`
- `invalid_or_empty_html`
- `preprocess_exception`

`*_reject_reasons.json` 会把 reject 按 reason 分组，并直接列出：

- `threshold_field`
- `threshold_value`
- `records`
  每条记录包含 `id`、`actual_cleaned_chars`、`visible_text_chars`、`original_chars` 和 `details`

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

## Stage 1 统计与可视化

Stage 1 会额外生成：

- `stats_log_path`：逐条统计 JSONL，包含 `status=kept/rejected` 与 `reject_reason`
- `summary_log_path`：聚合统计 JSON，包含 keep/reject 数量、原因分布和长度分位数
- `stats_plot_dir`：若环境已安装 `matplotlib`，自动输出多张分布图 PNG

`summary_log_path` 中会包含：

- `total_input / kept / rejected`
- `reject_reasons`
- `thresholds`
- `cleaned_chars / original_chars / visible_text_chars / compression_ratio` 的分布摘要
- `rule_counts`，用于看各类局部规则的触发频率

默认会画出这些分布：

- `original_chars`
- `cleaned_chars`，并标出 `min/max_preprocessed_chars` 阈值
- `visible_text_chars`
- `compression_ratio`
- `inline_script_chars`
- `json_payload_chars`
- `inline_style_chars`
- `hidden_input value chars`
- `html_comment_chars`
- `cleaned_chars keep vs reject`
- `reject_reason` 柱状图

如果环境里没有 `matplotlib`，Stage 1 仍然会成功完成，只是跳过图片输出。

如果环境里缺少 `lxml`，Stage 1 会直接报错停止；不会静默 fallback 到其他 parser。

## 媒体 placeholder 格式

所有媒体资源路径替换为：

```
__MEDIA_PLACEHOLDER__/media__width{W}__height{H}.ext
```

示例：
- `<img src="https://cdn.com/photo.jpg?v=1" width="640" height="480">` → `src="__MEDIA_PLACEHOLDER__/media__width640__height480.jpg"`
- `<img src="https://cdn.com/banner.png">` → `src="__MEDIA_PLACEHOLDER__/media__widthunknown__heightunknown.png"`
- `<video src="clip.mp4" poster="thumb.jpg" width="1280" height="720">` → src 和 poster 分别替换，继承 video 宽高

扩展名从原始 URL path 提取（去掉 query string），支持 `.jpg .jpeg .png .webp .gif .svg .mp4 .webm .mp3 .wav .ogg .pdf`。无法识别时用 `.media`。

## 目录结构

```
html_rewrite/
├── config.py                 # HtmlRewriteConfig dataclass
├── stage1_preprocess.py      # Stage 1 批量预处理（并发 + resume + 有序输出）
├── stage2_rewrite.py         # Stage 2 批量模型改写（并发 + resume + 有序输出）
├── main.py                   # CLI 入口
├── demo.py                   # 单条 debug 工具
├── preprocess/
│   ├── media.py              # 媒体路径替换
│   ├── scripts.py            # inline script / JSON payload 截断
│   ├── styles.py             # inline style 截断
│   ├── forms.py              # hidden input 截断
│   ├── comments.py           # HTML comment 截断
│   ├── formatter.py          # HTML 格式标准化
│   ├── filtering.py          # Stage 1 keep/reject gate
│   ├── analysis.py           # Stage 1 汇总统计 + 直方图输出
│   ├── preprocessor.py       # 编排器
│   └── stats.py              # PreprocessStats dataclass
├── prompts/
│   └── html_rewrite.py       # System prompt + user content builder
└── configs/
    └── default_local.yaml    # 配置模板
```

## 自定义 Prompt

在 `prompts/` 下新建模块，实现以下接口：

```python
SYSTEM_PROMPT: str = "..."

def build_user_content(preprocessed_html: str) -> list[dict]:
    ...
    return [{"type": "text", "text": prompt}]
```

在配置中指定：

```yaml
prompt_module: "my_custom_prompt"
```

## 设计文档

详见 `docs/reports/html_rewrite/design.md`。
