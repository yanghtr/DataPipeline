# html_rewrite_cot

Image-to-HTML CoT 数据构造流水线。

把现有的 image→HTML 训练样本（panguml 格式）中 assistant 的 raw HTML 回复，替换为带 reasoning 的格式：

```
reasoning_text

```html
raw_html
```
```

`reasoning_text` 由 VLM 生成（视觉分析 + 实现计划），`raw_html` 原文不变。

---

## 快速开始

```bash
# 1. 安装 Playwright（首次）
pip install playwright && playwright install chromium

# 2. 配置
cp html_rewrite_cot/configs/default.yaml my_config.yaml
# 编辑 my_config.yaml：填写 input.image_root、input.jsonl_files、vlm.url、vlm.api_key、vlm.model

# 3. 运行（从项目根目录）
python -m html_rewrite_cot.main --config my_config.yaml

# 调试：只跑 3 条
python -m html_rewrite_cot.main --config my_config.yaml --limit 3

# 分阶段运行
python -m html_rewrite_cot.main --config my_config.yaml --phase phase1   # 只跑 outline 提取
python -m html_rewrite_cot.main --config my_config.yaml --phase phase2   # 只跑 VLM 生成

# 忽略已有输出，重新处理
python -m html_rewrite_cot.main --config my_config.yaml --no-resume
```

---

## 流水线两阶段

**Phase 1 — HTML Outline 提取**（asyncio + Playwright）

- BeautifulSoup 静态解析：title、major_structure、dom_outline、structural text
- Playwright 渲染：computed styles、layout hints、colors、visual blocks
- 合并为 `html_outline_json` + `outline_text`
- 结果写入 `{run_dir}/outlines.jsonl`（作为 resume checkpoint）

**Phase 2 — VLM Reasoning 生成**（ThreadPoolExecutor）

- 输入：截图（base64）+ outline_text + raw_html
- 输出：reasoning_text（后处理）→ final_answer
- 结果写入 panguml 输出文件 + debug 文件

**Reasoning 结构（固定 6 段）**

1. 任务引导句
2. Layout Analysis：整体布局从上到下
3. 各区域段落（Header / Navigation / Main Content / Sidebar / Footer 等）— 包含图片/媒体区域描述
4. Colors Observed：主色调 + 色值
5. Structure and Implementation Plan：HTML/CSS 实现方案
6. 过渡句（自然引出 HTML 代码块）

---

## 配置说明

```yaml
input:
  image_root: /path/to/images        # 与 panguml relative_path 拼接的根目录
  jsonl_files:                        # 支持多个输入文件
    - /path/to/data_000000.jsonl

output:
  output_dir: ./outputs/cot          # panguml 格式输出（训练用）
  debug_dir:  ./outputs/cot_debug    # debug 格式输出（含中间字段，null 则跳过）

vlm:                                 # 与 html_rewrite 流水线字段对齐
  url: http://host/v1/chat/completions
  api_key: token
  model: model-name
  generation_params:
    temperature: 0.7
    max_tokens: 2048
    stop: ["```html"]               # 只用这一个；不要加 <html / <!DOCTYPE html>

runtime:
  num_workers: 4                     # Phase 2 VLM 并发
  playwright_concurrency: 2          # Phase 1 Playwright 并发
  resume: true
  run_dir: ./run

render:
  viewport_strategy: match_image     # 优先用截图尺寸作为 viewport
```

### stop tokens 说明

只使用 ` ```html ` 作为 stop token。

不使用 `<html` 或 `<!DOCTYPE html>`：这类 stop 会在模型 reasoning 中分析 HTML 标签时（如 "The `<html>` element has lang='en'"）误截断，对后续构造慢思考或 HTML 分析数据有破坏性。

### 图片/媒体区域描述策略（泛化设计）

训练数据中所有图片均为 LLM 改写后的 placeholder（单色方块 + 虚线边框），而未来推理时模型将面对真实网站截图（含真实照片、banner、icon）。

为使模型能从 placeholder 训练数据泛化到真实图片，prompt 要求模型用**布局语言**描述媒体区域，而非描述 placeholder 的外观特征：

| 不应出现（placeholder 特有，不可泛化） | 应使用（布局语言，可泛化） |
|---|---|
| "a gray box with dashed border" | "a full-width 16:9 hero banner" |
| "light green placeholder background" | "an 80×80px square avatar" |
| "Image Placeholder text overlay" | "a 4:3 product thumbnail in a card" |

无论截图显示的是真实照片还是 placeholder，模型都按同一套描述框架处理：**视觉角色 + 相对尺寸 + 宽高比**。具体实现方式（`<img>`、CSS `background-image` 容器、`<svg>`、placeholder div 等）由模型根据实际 HTML 结构自行判断，prompt 不预设答案。

---

## 输出文件

| 文件 | 格式 | 用途 |
|---|---|---|
| `{output_dir}/{stem}.jsonl` | panguml | 训练数据（assistant 替换为带 CoT 的回复） |
| `{debug_dir}/{stem}.jsonl` | 自定义 | 质检、统计（含 outline、reasoning_raw、quality_metadata） |
| `{run_dir}/outlines.jsonl` | JSONL | Phase 1 cache，resume 用 |
| `{run_dir}/phase2_done.txt` | 文本 | Phase 2 checkpoint，resume 用 |
| `{run_dir}/vlm_calls.jsonl` | JSONL | VLM API 调用日志 |
| `{run_dir}/pipeline.log` | 文本 | 完整运行日志 |

---

## 代码结构

```
html_rewrite_cot/
├── config.py           # 配置 dataclass（VLMConfig 字段与 html_rewrite 对齐）
├── models.py           # SampleRecord + panguml I/O
├── phase1_outline.py   # Phase 1 调度
├── phase2_generate.py  # Phase 2 调度
├── runner.py           # 整体编排 + 统计汇报
├── main.py             # CLI 入口
├── pipeline/
│   ├── outliner.py     # HTML outline 提取（BS4 + Playwright）
│   ├── renderer.py     # outline_json → outline_text
│   ├── vlm.py          # VLM 调用（复用 utils/api_client）
│   ├── postprocess.py  # reasoning 后处理 + 质量检查
│   └── assembler.py    # final_answer 拼接
└── configs/
    └── default.yaml    # 配置模板
```

详细设计见 `docs/task_specs/html_rewrite_cot/`。
