# html_rewrite_cot 使用说明

## 前置依赖

```bash
# Python 3.11 环境（已在 requirements.txt 中）
pip install beautifulsoup4 lxml pyyaml loguru requests

# Playwright（需额外安装浏览器）
pip install playwright
playwright install chromium
```

---

## 快速开始

### 1. 创建配置文件

复制模板并编辑：

```bash
cp html_rewrite_cot/configs/default.yaml my_config.yaml
```

最小配置（必填字段）：

```yaml
input:
  image_root: /home/yanghaitao/Projects/Data/FineWebEdu/rewrite_samples/images
  jsonl_files:
    - /home/yanghaitao/Projects/Data/FineWebEdu/rewrite_samples/jsonl/data_000000.jsonl

output:
  output_dir: ./outputs/cot
  debug_dir: ./outputs/cot_debug

vlm:
  api_base: http://localhost:8000/v1/chat/completions
  api_key: your-api-key
  model: your-model-name
```

### 2. 运行流水线

```bash
# 从项目根目录运行，完整流程
python -m html_rewrite_cot.main --config my_config.yaml

# 仅运行 Phase 1（outline 提取）
python -m html_rewrite_cot.main --config my_config.yaml --phase phase1

# 仅运行 Phase 2（VLM 生成）
python -m html_rewrite_cot.main --config my_config.yaml --phase phase2

# 调试：仅处理前 3 条，禁用 resume
python -m html_rewrite_cot.main --config my_config.yaml --limit 3 --no-resume
```

---

## 配置文件完整说明

```yaml
# ── 输入 ──────────────────────────────────────────────────────────────────────
input:
  image_root: /path/to/images          # 图片根目录（与 panguml 的 relative_path 拼接）
  jsonl_files:                          # 支持多个输入文件（共用同一个 image_root）
    - /path/to/data_000000.jsonl
    - /path/to/data_000001.jsonl

# ── 输出 ──────────────────────────────────────────────────────────────────────
output:
  output_dir: ./outputs/cot             # panguml 格式输出目录，文件名与输入同名
  debug_dir: ./outputs/cot_debug        # debug 格式输出目录（null 表示不输出）

# ── VLM API ───────────────────────────────────────────────────────────────────
vlm:
  api_base: http://localhost:8000/v1/chat/completions
  api_key: your-token
  model: your-model-name
  temperature: 0.7
  top_p: 0.9
  max_tokens: 2048
  max_retries: 3                        # 失败重试次数（含 timeout 和 5xx）
  timeout: 120.0                        # 单次请求超时（秒）
  ssl_verify: false                     # 本地 vLLM 通常为 false
  stop:                                 # stop tokens，降低 VLM 输出 HTML 的概率
    - "```html"
    - "<!DOCTYPE html>"
    - "<html"

# ── 运行时 ────────────────────────────────────────────────────────────────────
runtime:
  num_workers: 4                        # Phase 2 VLM 并发数
  playwright_concurrency: 2             # Phase 1 Playwright page 并发数
  resume: true                          # 是否断点续跑
  run_dir: ./run                        # checkpoint 和日志目录

# ── 渲染 ──────────────────────────────────────────────────────────────────────
render:
  viewport_strategy: match_image        # match_image | user_config | fallback
  viewport_width: null                  # user_config 模式下的宽度
  viewport_height: null                 # user_config 模式下的高度
  fallback_width: 1280                  # fallback 宽度
  fallback_height: 900                  # fallback 高度
  content_timeout_ms: 10000             # HTML 加载超时（毫秒）
  js_timeout_ms: 5000                   # JS 执行超时（毫秒）
  block_external: true                  # 拦截外部 HTTP/HTTPS 请求

# ── 全局 ──────────────────────────────────────────────────────────────────────
max_input_chars: 262144                 # raw_html 字符数上限（~64K tokens），超出跳过
```

---

## 输出文件说明

### panguml 输出（训练用）

路径：`{output_dir}/{input_stem}.jsonl`

格式与输入完全一致，仅 `assistant.content[0].text.string` 被替换为：

```
{reasoning_text}

```html
{raw_html}
```
```

### debug 输出（质检用）

路径：`{debug_dir}/{input_stem}.jsonl`

每条包含：

```json
{
  "sample_id": "data_000000:0",
  "jsonl_file": "/path/to/data_000000.jsonl",
  "line_no": 0,
  "image_rel_path": "stage2_xxx/image.png",
  "raw_html_len": 7412,
  "html_outline_json": { ... },
  "outline_text": "HTML outline\n\nTitle: ...",
  "vlm": { "model": "...", "api_base": "..." },
  "prompt_version": "image_to_html_cot_v8",
  "vlm_reasoning_raw": "...",
  "reasoning_text": "...",
  "final_answer": "reasoning...\n\n```html\n...\n```",
  "status": { "extraction": "ok", "generation": "ok" },
  "quality_metadata": {
    "reasoning_word_count": 320,
    "contains_html_in_reasoning": false,
    "has_layout_analysis_section": true,
    "has_colors_observed_section": true,
    "has_structure_implementation_plan_section": true,
    "region_section_count": 4,
    "alignment_warnings": []
  }
}
```

### run 目录（checkpoint 和日志）

```
{run_dir}/
├── outlines.jsonl      # Phase 1 outline cache（resume 用）
├── phase2_done.txt     # Phase 2 完成记录（resume 用）
├── vlm_calls.jsonl     # VLM API 调用日志
└── pipeline.log        # 完整运行日志
```

---

## 常见操作

### 断点续跑

中途中断后，直接重新运行相同命令：

```bash
python -m html_rewrite_cot.main --config my_config.yaml
```

- Phase 1 会读取 `run/outlines.jsonl`，跳过已完成的样本
- Phase 2 会读取 `run/phase2_done.txt`，跳过已完成的样本

### 仅重跑 Phase 2

如果 Phase 1 已完成（outline cache 完整），只想重跑 VLM 生成：

```bash
python -m html_rewrite_cot.main --config my_config.yaml --phase phase2
```

如果要用不同 VLM 重跑 Phase 2（清空 done 记录）：

```bash
rm run/phase2_done.txt outputs/cot/*.jsonl outputs/cot_debug/*.jsonl
python -m html_rewrite_cot.main --config my_config.yaml --phase phase2
```

### 多文件批处理

在配置中列出多个文件，每个文件输出到同名输出文件：

```yaml
input:
  image_root: /path/to/images
  jsonl_files:
    - /path/to/data_000000.jsonl    → outputs/cot/data_000000.jsonl
    - /path/to/data_000001.jsonl    → outputs/cot/data_000001.jsonl
    - /path/to/data_000002.jsonl    → outputs/cot/data_000002.jsonl
```

### 调试单条样本

```bash
python -m html_rewrite_cot.main --config my_config.yaml --limit 1 --no-resume
```

然后查看 debug 输出：

```python
import json
with open('outputs/cot_debug/data_000000.jsonl') as f:
    sample = json.loads(f.readline())
print(sample['outline_text'])          # 查看 outline
print(sample['reasoning_text'][:500])  # 查看 reasoning
print(sample['quality_metadata'])      # 查看质量元信息
```

---

## 质量检查

### 批量统计

```python
import json
from pathlib import Path

records = []
for path in Path('outputs/cot_debug').glob('*.jsonl'):
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))

total = len(records)
ok = sum(1 for r in records if r['status']['generation'] == 'ok')
has_html = sum(1 for r in records if r['quality_metadata']['contains_html_in_reasoning'])
has_layout = sum(1 for r in records if r['quality_metadata']['has_layout_analysis_section'])
avg_words = sum(r['quality_metadata']['reasoning_word_count'] for r in records) / total

print(f"Total: {total}")
print(f"Generation ok: {ok} ({ok/total:.1%})")
print(f"HTML in reasoning: {has_html} ({has_html/total:.1%})")
print(f"Has layout analysis: {has_layout} ({has_layout/total:.1%})")
print(f"Avg reasoning words: {avg_words:.0f}")
```

### Playwright 超时统计

```bash
grep "playwright timeout" run/pipeline.log | wc -l
```

---

## 常见问题

**Q: Playwright 报错 "Browser not found"**
```bash
playwright install chromium
```

**Q: VLM 返回包含 HTML 的 reasoning**

会在 `quality_metadata.contains_html_in_reasoning=true` 中标记，generation_status 为 `warning`。样本仍会输出，但建议后续过滤或人工审核。stop tokens 配置可以降低此概率。

**Q: Phase 1 全部 timeout**

- 检查 Playwright 是否正常安装：`python -c "from playwright.sync_api import sync_playwright; print('ok')"`
- 尝试增大 `render.content_timeout_ms`（如 20000）
- 检查 HTML 是否包含阻塞性 JS 调用（这批数据应该没有）

**Q: 输出文件与输入顺序不一致**

Phase 2 使用并发写入，输出顺序与输入顺序无关。如果需要保序，使用 sample_id 重新排序 debug 输出中的记录。panguml 输出顺序不影响训练。

**Q: 如何验证输出格式正确**

```python
import json
from pathlib import Path

with open('outputs/cot/data_000000.jsonl') as f:
    sample = json.loads(f.readline())

# 验证格式
assert sample['meta_prompt'] == ['']
assert len(sample['data']) == 2
assert sample['data'][0]['role'] == 'user'
assert sample['data'][1]['role'] == 'assistant'
answer = sample['data'][1]['content'][0]['text']['string']
assert '```html' in answer
assert 'Layout Analysis:' in answer
print("Format OK")
print(f"Answer length: {len(answer)} chars")
```
