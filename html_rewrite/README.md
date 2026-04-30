# html_rewrite

将真实网站原始 HTML 批量改写为干净、单文件、可渲染 HTML 的数据生产流水线。

## 流程概述

```
raw JSONL  →  [Stage 1: 预处理]  →  preprocessed JSONL  →  [Stage 2: 模型改写]  →  output JSONL
```

两个阶段通过中间文件解耦，可独立运行、独立 resume。输出结果与输入严格同序，两阶段结果可逐行对应比较。

## 快速开始

### 1. 安装依赖

```bash
pip install beautifulsoup4>=4.12 lxml>=5.0
# 其余依赖见项目根目录 requirements.txt
```

### 2. 配置文件

复制并编辑配置：

```bash
cp html_rewrite/configs/default_local.yaml html_rewrite/configs/my_config.yaml
```

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
python -m html_rewrite.main --config html_rewrite/configs/my_config.yaml --stage preprocess

# 检查 logs/html_rewrite_stats.jsonl 中的统计分布后，如需调整阈值：
python -m html_rewrite.main --config html_rewrite/configs/my_config.yaml --stage preprocess --no-resume

# Stage 2：模型改写（需要 API 可用）
python -m html_rewrite.main --config html_rewrite/configs/my_config.yaml --stage rewrite

# 全流程一次跑完
python -m html_rewrite.main --config html_rewrite/configs/my_config.yaml --stage all

# 调试单条（取第 0 条，默认）
python -m html_rewrite.demo --config html_rewrite/configs/my_config.yaml --stage preprocess
python -m html_rewrite.demo --config html_rewrite/configs/my_config.yaml --stage rewrite --index 2
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
| `inline_script_max_chars` | `4096` | inline script 截断阈值 |
| `json_payload_max_chars` | `4096` | JSON payload 截断阈值 |
| `hidden_input_max_chars` | `4096` | hidden input value 截断阈值 |
| `html_comment_max_chars` | `1024` | HTML comment 截断阈值 |
| `inline_style_max_chars` | `32768` | inline style 截断阈值 |
| `fetch_media_size` | `false` | 是否下载图片头部以获取尺寸（默认关闭） |
| `num_workers` | `16` | 并发线程数 |
| `resume` | `true` | 是否断点续跑（跳过已完成条目） |
| `generation_params` | `{}` | 透传给 API 的生成参数（temperature、max_tokens 等） |
| `prompt_module` | `html_rewrite` | prompts/ 下的 prompt 模块名 |

## 输出格式

### Stage 1 输出（preprocessed JSONL）

每行一条，字段：

```json
{
  "id": "https://example.com/",
  "_meta": { "url": "...", "final_url": "...", "crawl_time": 1711152000, "page_type": ["HOME_PAGE"], "part": "..." },
  "preprocessed_html": "<!DOCTYPE html>...",
  "preprocess_stats": {
    "original_chars": 114000,
    "cleaned_chars": 65000,
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
