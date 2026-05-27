# html_rewrite_cot 设计报告

## 1. 任务目标

将现有的 image→HTML 训练样本（panguml 格式）转换为带 CoT reasoning 的格式：

**输入**（panguml）：
```
user:      image + "Produce HTML code for this image."
assistant: raw HTML
```

**输出**（panguml，assistant 字段替换）：
```
user:      image + "Produce HTML code for this image."  ← 不变
assistant: reasoning_text\n\n```html\nraw_html\n```
```

`reasoning_text` 由 VLM 生成，描述从截图到 HTML 实现的视觉分析和实现计划。`raw_html` 使用原始目标 HTML，不由 VLM 重新生成。

---

## 2. 总体流程

```
输入 JSONL (panguml)
│
├─ image_rel_path + image_root → 完整图片路径
├─ assistant.string → raw_html
│
▼
Phase 1: HTML Outline 提取（asyncio + Playwright 共享 browser）
│
├─ BeautifulSoup 静态解析
│    title, major_structure, dom_outline, structural_text
├─ Playwright 渲染
│    computed styles, layout hints, colors, visual blocks
└─ 合并 → html_outline_json + outline_text
│
▼
Phase 2: VLM Reasoning 生成（ThreadPoolExecutor）
│
├─ image(base64) + outline_text + raw_html → VLM API
├─ 后处理（strip, fence 剥离, HTML 检测）
├─ 质量元信息检查
└─ 拼接 final_answer
│
▼
写出：
├─ {output_dir}/{stem}.jsonl      panguml 格式（训练用）
├─ {debug_dir}/{stem}.jsonl       debug 格式（含所有中间字段）
└─ {run_dir}/outlines.jsonl       Phase 1 outline cache（resume 用）
```

---

## 3. 架构决策记录

### 3.1 两阶段串行执行

Phase 1（Playwright）和 Phase 2（VLM）分阶段执行，原因：

- Playwright 是重量级操作（浏览器进程），并发数需要独立限制
- VLM API 是网络 I/O，可以高并发
- 两阶段分离使得可以单独重跑某一阶段（`--phase phase1` / `--phase phase2`）

Phase 1 使用 `asyncio` + Playwright async API（共享单个 browser，page-level 并发）。
Phase 2 使用 `ThreadPoolExecutor`，复用现有同步 `utils/api_client`。

### 3.2 sample_id 设计

`sample_id = "{jsonl_stem}:{line_no}"`

- `jsonl_stem`：输入文件名去掉扩展名，如 `data_000000`
- `line_no`：JSONL 文件中的 0-based 行号
- 示例：`data_000000:0`

panguml 格式本身没有 ID 字段，这个 ID 只存在于 debug 输出和内部 checkpoint 中。

### 3.3 Resume 机制

两个 checkpoint 文件：

| 文件 | 用途 |
|---|---|
| `{run_dir}/outlines.jsonl` | Phase 1 outline cache（JSONL，每条含 sample_id） |
| `{run_dir}/phase2_done.txt` | Phase 2 完成记录（每行一个 sample_id） |

启动时：
- Phase 1 读 `outlines.jsonl`，跳过已有 sample_id
- Phase 2 读 `phase2_done.txt`，跳过已有 sample_id

两个文件都是 append-only，写操作用 asyncio.Lock / threading.Lock 保证并发安全。

### 3.4 输出格式分离

| 输出 | 格式 | 用途 |
|---|---|---|
| `{output_dir}/{stem}.jsonl` | panguml（只改 assistant.string） | 训练数据 |
| `{debug_dir}/{stem}.jsonl` | 自定义 debug 格式 | 质检、统计、分析 |

debug 格式包含：`html_outline_json`、`outline_text`、`vlm_reasoning_raw`、`reasoning_text`、`final_answer`、`status`、`quality_metadata`。

### 3.5 Playwright 渲染策略

**外部资源屏蔽**：`block_external=true` 时，拦截所有 `http://` / `https://` 请求，避免等待外网 CDN 资源。样本 HTML 是自包含的（embedded CSS），只需本地渲染。

**viewport 优先级**：
1. `match_image`：使用 panguml 样本中的 `image.width` / `image.height`（首选）
2. `user_config`：使用配置文件中的 `render.viewport_width/height`
3. `fallback`：使用 `fallback_width/height`（默认 1280×900），记录 warning

**超时处理**：`set_content` 和 `evaluate` 各有独立超时。超时时降级为仅静态提取（不丢弃样本），记录 warning 到 `extraction_warnings` 和日志，方便后续统计。

### 3.6 max_input_chars

超过 `max_input_chars`（默认 262144 chars ≈ 64K tokens）的样本在 Phase 1 直接标记为 `failed`，跳过 Phase 2。这批数据是改写后的 self-contained HTML，长度远小于该阈值，此配置作为安全兜底。

---

## 4. 模块说明

```
html_rewrite_cot/
├── config.py           # 配置 dataclass + YAML 加载
├── models.py           # SampleRecord + panguml I/O 辅助函数
├── phase1_outline.py   # Phase 1 主逻辑（asyncio 调度）
├── phase2_generate.py  # Phase 2 主逻辑（ThreadPoolExecutor）
├── runner.py           # 整体编排 + 统计汇报
├── main.py             # CLI 入口
├── pipeline/
│   ├── outliner.py     # HTML outline 提取（BS4 + Playwright）
│   ├── renderer.py     # outline_json → outline_text
│   ├── vlm.py          # VLM 调用（复用 utils/api_client）
│   ├── postprocess.py  # reasoning 后处理 + quality check
│   └── assembler.py    # final_answer 拼接
└── configs/
    └── default.yaml    # 配置模板
```

---

## 5. HTML Outline 提取算法

### 5.1 静态提取（BeautifulSoup）

| 字段 | 算法 |
|---|---|
| `title` | `<title>` > `<h1>` > `og:title` > `meta[name=title]` |
| `major_structure` | 遍历 body 直接子元素，按语义 tag 和 class/id 关键词识别区域，描述子组件 |
| `dom_outline` | 递归保留语义 tag，div/span 按5条规则过滤，深度限 8 层 |
| `headings` | `h1-h6`，跳过 hidden |
| `navigation_links` | `nav/header` 内的 `<a>` + class/id 命中 nav 关键词的容器内的 `<a>` |
| `buttons` | `<button>`, `input[type=button/submit/reset]`, `a[class*=btn/cta]` |
| `forms` | `<form>` + 内部 field 列表 |
| `sidebar_or_card_titles` | sidebar/card 类容器内的 heading/strong |
| `footer_links` | `<footer>` 及 footer-like 容器内的 `<a>` |
| `lists` | `ul/ol` 前 6 个，每个取前 5 items |
| `tables` | headers + row_count + 前 2 行数据 |

div/span 保留条件（满足任一）：
1. class/id 命中结构关键词
2. 自身有 ≤80 字符的直接文本
3. 子树包含 heading/nav/form/button/img/svg/list/table
4. ~~（v0 未使用 computed style 判断，由 Playwright 补充）~~

### 5.2 Playwright 渲染提取

JS 内联脚本在 page context 中执行，提取：

- **采样元素**：所有语义 tag + 结构关键词 div
- **每个元素**：bbox, display, flexDirection, gridTemplateColumns, backgroundColor, color, fontFamily, fontSize, border, borderRadius, backgroundImage
- **视觉块识别**：`area > 400px² ∧ textContent < 15chars ∧ (有背景色 ∨ 有背景图 ∨ 有边框)`
- **pseudo-elements**：`::before`/`::after` 中有背景色的元素

Python 侧后处理：
- **layout_hints**：flex row/column, grid layout
- **style_hints**：full-width bar（nav/header），card-like box
- **colors**：主要元素的 background-color 和 text color，去重，转 hex
- **fonts**：唯一字体族前 5

---

## 6. VLM Prompt

### System Prompt

指示模型作为 frontend engineer + visual annotator，生成 reasoning 文本（不输出代码）。

### User Prompt

包含：
- 任务说明（5条 reasoning 结构要求）
- `<outline>` XML 块（outline_text）
- `<target_html>` XML 块（raw_html）
- 截图（base64 image_url）

### Stop Tokens

`stop: ["```html", "<!DOCTYPE html>", "<html"]`

降低 VLM 提前输出 HTML 的概率。不同后端对 stop 参数支持程度不同，通过配置控制，API 不支持时自动忽略。

---

## 7. 质量元信息

| 字段 | 含义 |
|---|---|
| `reasoning_word_count` | reasoning 词数 |
| `contains_html_in_reasoning` | reasoning 是否包含 HTML 代码 |
| `contains_markdown_fence_in_reasoning` | 是否有 code fence 残留 |
| `has_layout_analysis_section` | 是否有 "Layout Analysis:" 段落 |
| `has_colors_observed_section` | 是否有 "Colors Observed:" 段落 |
| `has_structure_implementation_plan_section` | 是否有 "Structure and Implementation Plan:" 段落 |
| `region_section_count` | region-specific section 数量 |
| `alignment_warnings` | 弱对齐检查警告（如 reasoning 提到 form 但 HTML 无 form） |

---

## 8. 与现有代码的对齐

| 项目 | html_rewrite | html_rewrite_cot |
|---|---|---|
| API 调用 | `utils/api_client.call_chat_completion` | 相同 |
| 并发模型 | `ThreadPoolExecutor` | Phase 1: asyncio；Phase 2: ThreadPoolExecutor |
| 配置 | `HtmlRewriteConfig` dataclass + YAML | 相同模式，嵌套结构 |
| 日志 | loguru | 相同 |
| 输出 | 自定义 JSONL | panguml + debug 双路输出 |
| Resume | output file 去重 | outline cache + done file 双文件 |
