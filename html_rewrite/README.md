# html_rewrite

将真实网站原始 HTML 批量改写为干净、单文件、可渲染 HTML 的数据生产流水线。

## 流程概述

```
input shard JSONL(s)  →  [Stage 1: 预处理 + 过滤 + 统计可视化]  →  stage1/<shard>/preprocessed.jsonl  →  [Stage 2: 模型改写]  →  stage2/<shard>/output.jsonl
```

两个阶段通过中间文件解耦，可独立运行、独立 resume。输出结果与输入严格同序，两阶段结果可逐行对应比较。

## 快速开始

### 1. 安装依赖

```bash
pip install beautifulsoup4>=4.12 lxml>=5.0 matplotlib>=3.8 langid>=1.1.6
# 其余依赖见项目根目录 requirements.txt
```

`langid` 只需要安装 Python 包本身，不需要额外下载模型文件；语言识别模型已经随包分发。

### 2. 配置文件

默认配置文件路径：

```bash
html_rewrite/configs/default_local.yaml
```

直接编辑这个文件即可；下面的命令也默认使用它。

默认配置已启用“只保留英文主语言页面”的过滤；如需关闭或调阈值，直接编辑这个文件即可。

默认推荐配置（模板展开 + run 级分片输出）：

```yaml
url: "http://localhost:8000/v1/chat/completions"   # OpenAI-compatible API endpoint
api_key: "your-api-key"
model: "your-model-name"
input_dir: "/path/to/raw_dir"
input_filename_template: "part-{index:05d}.jsonl"
input_start_index: 0
input_end_index_exclusive: 100
run_root_dir: "/path/to/run_20260501"
output_shard_name_template: "part-{index:05d}"
```

这会按顺序展开：

- `/path/to/raw_dir/part-00000.jsonl`
- `/path/to/raw_dir/part-00001.jsonl`
- ...
- `/path/to/raw_dir/part-00099.jsonl`

其中：

- `input_start_index` 是包含式
- `input_end_index_exclusive` 是不包含式

如果你更喜欢手动指定，也仍然支持：

- `input_path`
- `input_paths`

但默认推荐模板展开模式。

### 3. 运行

```bash
# Stage 1：仅预处理（离线，无需 API，可调整阈值反复运行）
python -m html_rewrite.main --config html_rewrite/configs/default_local.yaml --stage preprocess

# 检查 run_root_dir/stage1/<shard>/summary.json 与 plots/ 后，如需调整阈值：
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

- `run_root_dir/manifest.json`：输入分片与输出 shard 的固定映射，保证 resume 安全
- `run_root_dir/stage1/<shard>/preprocessed.jsonl`：该 shard 的 keep 样本
- `run_root_dir/stage1/<shard>/stats.jsonl`：该 shard 的逐条统计
- `run_root_dir/stage1/<shard>/rejects.jsonl`：该 shard 的 reject 样本
- `run_root_dir/stage1/<shard>/summary.json`：该 shard 的 Stage 1 汇总
- `run_root_dir/stage1/<shard>/summary_reject_reasons.json`：该 shard 的 reject 原因明细
- `run_root_dir/stage1/<shard>/plots/`：该 shard 的直方图
- `run_root_dir/stage2/<shard>/output.jsonl`：该 shard 的 Stage 2 输出
- `run_root_dir/stage2/<shard>/api_calls.jsonl`：该 shard 的 API 调用日志
- `run_root_dir/aggregate/stage1_summary.json`：所有 shard 的 Stage 1 聚合摘要
- `run_root_dir/aggregate/stage2_summary.json`：所有 shard 的 Stage 2 聚合摘要

旧模式兼容说明：

- 如果没有设置 `run_root_dir`，程序会退回单文件输出模式
- 这时会使用 `preprocessed_path`、`output_path`、`stats_log_path`、`summary_log_path` 等旧字段

常用调试方式：

```bash
# 先只跑前 5 条，确认过滤和分布图输出正常
python -m html_rewrite.main --config html_rewrite/configs/default_local.yaml --stage preprocess --limit 5

# 查看单条预处理结果和统计
python -m html_rewrite.demo --config html_rewrite/configs/default_local.yaml --stage preprocess --index 0
```

当使用 `input_paths` 或模板展开模式时，`--limit` 和 `demo --index N` 都是基于“拼接后的总顺序”计数的。

## 配置参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `url` | — | API endpoint（必填） |
| `api_key` | — | API key（必填） |
| `model` | — | 模型名（必填） |
| `input_dir` | `""` | 输入目录（模板展开模式） |
| `input_filename_template` | `""` | 输入文件名模板，例如 `part-{index:05d}.jsonl` |
| `input_start_index` | `null` | 输入起始 index，包含 |
| `input_end_index_exclusive` | `null` | 输入结束 index，不包含 |
| `run_root_dir` | `""` | 分片输出根目录；非空时启用 run 级 shard 模式 |
| `output_shard_name_template` | `part-{index:05d}` | 输出 shard 目录名模板 |
| `input_path` | `""` | 兼容旧模式：单个原始 JSONL 路径 |
| `input_paths` | `[]` | 兼容旧模式：多个原始 JSONL 路径，按列表顺序拼接 |
| `preprocessed_path` | `preprocessed.jsonl` | 旧模式 Stage 1 输出路径 |
| `output_path` | `html_rewrite_output.jsonl` | 旧模式 Stage 2 输出路径 |
| `call_log_path` | `logs/html_rewrite_api_calls.jsonl` | 旧模式 API 原始调用日志 |
| `stats_log_path` | `logs/html_rewrite_stats.jsonl` | 旧模式预处理统计日志 |
| `reject_log_path` | `logs/html_rewrite_rejects.jsonl` | 旧模式被 Stage 1 过滤掉的样本日志 |
| `summary_log_path` | `logs/html_rewrite_summary.json` | 旧模式 Stage 1 汇总统计 |
| `stats_plot_dir` | `logs/html_rewrite_plots` | 旧模式 Stage 1 分布图输出目录 |
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

英文主语言过滤参数：

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `enable_language_filter` | `true` | 是否启用“只保留英文主语言页面” |
| `allowed_languages` | `["en"]` | 允许保留的主语言 |
| `language_detector` | `langid` | 轻量语言识别器 |
| `language_min_visible_text_chars` | `200` | 可见文本太短时不做可信语言判断 |
| `language_min_letter_chars` | `100` | 字母字符数下限，避免数字/符号页误判 |
| `language_sample_max_chars` | `12000` | 送入 detector 的最大文本长度 |
| `language_min_latin_ratio` | `0.6` | 拉丁字母在全部字母中的最低占比 |
| `language_min_detector_margin` | `3.0` | `langid` top1/top2 分数差最小值 |

## 输出格式

### Stage 1 输出（preprocessed JSONL）

在默认 shard 模式下，每个输入 shard 对应：

- `run_root_dir/stage1/<shard>/preprocessed.jsonl`

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
    "formatter": { "parse_ok": true, "node_count_before": 1504, "node_count_after": 1596 },
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

`preprocess_stats` 中重点建议关注：

- `cleaned_chars`：最终进入 gate 的 HTML 长度
- `visible_text_chars`：页面可见文本长度，后续如果要加双 gate 会很有用
- `compression_ratio`：判断当前预处理是否真的压缩了上下文
- `scripts/json_payloads/styles/hidden_inputs/comments`：各类局部截断的次数和长度分布
- `language`：页面主语言检测结果和英文过滤决策

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
- `language_detection_insufficient_text`
- `language_detection_insufficient_letters`
- `language_not_mainly_latin_script`
- `non_english_page`
- `language_detection_low_margin`
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
  "response": "...模型 message.content 原文...",
  "reasoning": "...模型 reasoning / reasoning_content（如果后端提供）...",
  "output_html": "<!DOCTYPE html>...",
  "usage": { "prompt_tokens": 12000, "completion_tokens": 8000 },
  "model": "your-model",
  "prompt_tokens": 12000,
  "completion_tokens": 8000,
  "finish_reason": "stop"
}
```

说明：

- `response`：模型最终 `message.content` 文本，不等于 `output_html`
- `reasoning`：若后端是 reasoning 模型并单独返回 thinking，这里保存该文本
- `output_html`：从 `response` 中抽取出来的最终 HTML
- `usage.prompt_tokens / completion_tokens`：后端 API 返回的 usage 统计

对于当前默认本地后端（`http://localhost:8000/v1/chat/completions`，通常是 vLLM OpenAI-compatible server）：

- Qwen3 系列 reasoning 默认开启，除非显式传 `chat_template_kwargs.enable_thinking=false`
- reasoning 模式下，`completion_tokens` 应视为后端 API 口径的总输出 token；若当前记录存在 `reasoning`，通常表示它包含 `thinking + 最终正文`
- 若后端未来返回 `completion_tokens_details.reasoning_tokens` 等细分字段，建议优先使用这些字段做更精确拆分

## Stage 1 统计与可视化

Stage 1 会额外生成：

- `stats_log_path`：逐条统计 JSONL，包含 `status=kept/rejected` 与 `reject_reason`
- `summary_log_path`：聚合统计 JSON，包含 keep/reject 数量、原因分布和长度分位数
- `stats_plot_dir`：若环境已安装 `matplotlib`，自动输出多张分布图 PNG

`summary_log_path` 中会包含：

- `total_input / kept / rejected`
- `reject_reasons`
- `thresholds`
- `cleaned_chars / original_chars / visible_text_chars / compression_ratio` 的汇总统计
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

## 英文主语言过滤

目标是筛出“**页面主要可见文本为英文**”的页面，而不是简单判断这个域名是不是英文站。

### `langid.py` 是规则还是模型

`langid.py` 不是纯规则方法。它自带一个轻量的**预训练语言识别模型**，安装包里已经包含模型参数：

- 不需要联网下载模型
- 不需要 GPU
- 不属于大模型推理
- 适合放在 Stage 1 批量过滤

它比只看 `html lang`、URL、域名后缀、ASCII 比例这类规则更鲁棒；但也不建议单独盲信一次检测结果，所以推荐和规则信号结合使用。

### 为什么不建议只用规则

只用规则会遇到这些问题：

- 很多网页没有 `html lang`，或者模板把 `lang` 写死
- 多语言站点同一域名下可能同时有英文页和非英文页
- ASCII / 拉丁字符比例高，并不等于英文，法语/德语/西语也可能满足
- URL 中的 `/en/`、域名后缀、title 文本都只能做弱信号

### 当前实现

1. 先抽取 `visible_text`，不要直接拿原始 HTML 做检测。
2. 如果 `visible_text_chars < 200`，直接 reject：`language_detection_insufficient_text`。
3. 如果字母字符数 `< 100`，直接 reject：`language_detection_insufficient_letters`。
4. 计算拉丁字母占全部字母的比例；若 `< 0.6`，直接 reject：`language_not_mainly_latin_script`。
5. 收集声明性信号：`<html lang>`、`content-language`、`og:locale`。
6. 对可见文本做采样后送入 `langid`：
   - 建议取前/中/后文本拼接
   - 总长度最多 `12000 chars`
7. 用 `langid.rank()` 看 top1 / top2：
   - 若 `top1 != "en"`，reject：`non_english_page`
   - 若 `top1 == "en"` 但 top1-top2 差值 `< 3.0`，reject：`language_detection_low_margin`
8. 最终只保留主语言判断稳定为英文的页面。

### 为什么这套更稳

- 先用可见文本，避免 HTML 标签和脚本污染
- 先用便宜规则过滤掉明显不适合检测的页面
- 再用轻量模型做主判据
- 用 top1/top2 差值，而不是迷信单个 score
- 用前中后采样，降低导航栏或页头对整页判断的干扰

### 建议记录的调试字段

当前会在 stats 或 reject 明细中记录这些字段：

- `declared_lang`
- `detected_lang`
- `detected_lang_top2`
- `detector_margin`
- `visible_text_chars`
- `letter_chars`
- `latin_ratio`
- `language_filter_decision`
- `language_filter_reason`

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
├── run_layout.py             # shard 展开、run 目录布局与 manifest 管理
├── preprocess/
│   ├── media.py              # 媒体路径替换
│   ├── scripts.py            # inline script / JSON payload 截断
│   ├── styles.py             # inline style 截断
│   ├── forms.py              # hidden input 截断
│   ├── comments.py           # HTML comment 截断
│   ├── formatter.py          # HTML 格式标准化
│   ├── filtering.py          # Stage 1 keep/reject gate
│   ├── analysis.py           # Stage 1 汇总统计 + 直方图输出
│   ├── parser.py             # lxml 依赖检查与统一解析入口
│   ├── text.py               # 可见文本 / 声明语言抽取
│   ├── language.py           # 英文主语言检测与过滤
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
