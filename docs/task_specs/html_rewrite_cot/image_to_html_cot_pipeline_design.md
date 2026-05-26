# Image-to-HTML CoT 数据构造管线设计文档

## 1. 目标

已有数据：

```text
Question = webpage screenshot image
Answer   = raw HTML
```

构造新的训练数据：

````text
Question = webpage screenshot image
Answer   = reasoning text + HTML code

{{reasoning_text}}

```html
{{raw_html}}
```
````

其中：

- `reasoning_text` 由 VLM 生成，用于描述从截图到 HTML 实现的视觉分析和实现计划。
- `reasoning_text` 是普通文本，不使用 fenced code block。
- `raw_html` 使用已有目标 HTML，不由 VLM 重新生成。
- HTML 使用 ```html fenced code block。

---

## 2. 总体流程

```text
Input:
  - sample_id
  - image_path
  - raw_html

Step 1. Extract HTML outline
  raw_html + rendered page
  → html_outline_json

Step 2. Render prompt context
  html_outline_json
  → outline_text

Step 3. Generate reasoning
  image + outline_text + raw_html
  → VLM
  → reasoning_text

Step 4. Assemble final answer
  reasoning_text + fenced raw_html
  → final_answer

Output:
  - original sample fields
  - html_outline_json
  - outline_text
  - vlm_reasoning_raw
  - reasoning_text
  - final_answer
  - status / quality_metadata
```

---

## 3. 用户配置

```yaml
# input相关路径
# output相关路径
image_root: /path/to/images

vlm:
  api_base: ...
  api_key: ...
  model: ...
  temperature: ...
  top_p: ...
  max_tokens: ...
  max_retries: ...
  timeout: ...
  stop: ["```html", "<!DOCTYPE html>", "<html"]

runtime:
  num_workers: ...
  resume: true
  debug_dump_dir: /path/to/debug

render:
  viewport_strategy: match_image
  viewport_width: null
  viewport_height: null
```

这里输入的数据是panguml格式 @.claude/skills/panguml-format/SKILL.md ，我们希望可以支持指定多个jsonl文件(它们都是用同样一个image_root)
我们希望输出和输入格式一样，就是把输入的assistant回复变成带cot的格式，文件命名和其它内容和原来一样，只不过是存在了另外一个目录

`stop` 仅在 API 支持时使用，用于降低 VLM 提前输出 HTML 的概率。

`viewport` 只用于 Playwright 渲染 HTML，以提取 computed style 和 bounding box。优先使用截图尺寸作为 viewport；如果截图尺寸不可用，则使用用户配置的 viewport；如果仍未配置，则使用代码 fallback 并记录 warning。

---

## 4. 目标 reasoning 风格

目标 reasoning 借鉴 GLM-5V-Turbo / Kimi-K2.6 的 image-to-HTML 输出风格，但不绑定某个模型的固定模板。

### 4.1 核心风格要求

```text
1. 以一句自然的任务说明开头，说明要把截图转成 self-contained HTML with embedded CSS。
2. 使用结构化视觉分析，而不是单段泛泛描述。
3. 先分析整体布局，再分析页面实际存在的主要区域。
4. 明确列出可见组件和关键文字组，例如 logo、导航、搜索框、标题、卡片、表单、侧栏、footer links。
5. 明确 layout 决策，例如 two-column / three-column、sidebar、card grid、float-like summary box、flex/grid。
6. 单独总结颜色和视觉风格，允许 approximate color words 或 approximate hex。
7. 最后写 HTML/CSS 实现计划，例如 embedded CSS、flexbox/grid、centered container、placeholder、CSS-drawn blocks。
8. 如果布局关系有歧义，可以简短说明判断，并给出最终采用的实现方式。
```

### 4.2 推荐输出结构

```text
[Brief task framing sentence]

Layout Analysis:
...

[Relevant page-region sections based on the screenshot]:
...

Colors Observed:
...

Structure and Implementation Plan:
...
```

固定 section：

```text
Layout Analysis:
Colors Observed:
Structure and Implementation Plan:
```

region-specific sections 由 VLM 根据页面实际情况生成，不限制数量，不要求从固定列表中选择。常见 section heading 可以包括但不限于：

```text
Header Section
Navigation Section
Hero Section
Main Content Area
Sidebar
Summary/Card Section
Cards and Lists
Forms and Tables
Footer Section
Assets and Visual Blocks
```

---

## 5. HTML Outline

### 5.1 内部 JSON

`html_outline_json` 使用 JSON 保存，用于 debug、统计、质检和 prompt 渲染。

```json
{
  "meta": {
    "title": null,
    "viewport": {
      "width": null,
      "height": null,
      "source": "match_image|user_config|fallback"
    },
    "warnings": []
  },
  "structure": {
    "major_structure": [],
    "dom_outline": []
  },
  "text": {
    "headings": [],
    "navigation_links": [],
    "buttons": [],
    "forms": [],
    "sidebar_or_card_titles": [],
    "footer_links": [],
    "lists": [],
    "tables": []
  },
  "layout_style": {
    "layout_hints": [],
    "style_hints": [],
    "colors": [],
    "fonts": []
  },
  "assets": {
    "images": [],
    "svg_elements": [],
    "css_backgrounds": [],
    "visual_blocks": [],
    "pseudo_visual_blocks": []
  }
}
```

所有字段必须存在。没有提取到内容时使用空数组或 null。

### 5.2 Outline Text 渲染模板

程序将 `html_outline_json` 渲染为紧凑文本并放入 VLM prompt：

```text
HTML outline

Title:
{{title_or_none}}

Major structure:
{{major_structure_or_none}}

Structural text:
Headings:
{{headings_or_none}}
Navigation links:
{{navigation_links_or_none}}
Buttons:
{{buttons_or_none}}
Forms:
{{forms_or_none}}
Sidebar/card titles:
{{sidebar_or_card_titles_or_none}}
Footer links:
{{footer_links_or_none}}
Lists:
{{lists_or_none}}
Tables:
{{tables_or_none}}

Layout and style:
{{layout_style_hints_or_none}}

Assets and visual blocks:
{{assets_or_visual_blocks_or_none}}

Extraction warnings:
{{warnings_or_none}}
```

空字段渲染为：

```text
None extracted
```

---

## 6. Outline 提取算法

### 6.1 HTML 解析

实现方式：

```text
Python + BeautifulSoup or lxml
```

处理步骤：

```text
1. 解析 raw_html。
2. 保留 raw_html 原文，不覆盖、不格式化、不重新序列化。
3. DOM outline 中跳过 script / noscript / template / meta / link。
4. style 不进入 DOM outline，但保留在 raw_html 中供浏览器渲染。
5. 跳过明显 hidden 节点：
   - hidden attribute
   - aria-hidden="true"
   - inline style contains display:none
   - inline style contains visibility:hidden
```

### 6.2 Title

提取优先级：

```text
1. <title>
2. first h1
3. meta[property="og:title"]
4. meta[name="title"]
5. null
```

### 6.3 Major Structure

`major_structure` 是页面一级结构的短描述，由规则生成。

提取依据：

```text
semantic tags
class/id keywords
computed layout hints
structural text in the region
visible assets / visual blocks in the region
```

示例：

```json
[
  "header with logo visual block and navigation links",
  "main area with article content and right sidebar form",
  "footer with policy links and logo visual blocks"
]
```

### 6.4 DOM Outline

保留标签：

```text
header, nav, main, article, aside, section, footer
form, table, thead, tbody, tr, th, td
ul, ol, li
h1, h2, h3, h4, h5, h6
button, input, textarea, select, label
img, picture, source, svg
a
```

普通 `div/span` 满足任一条件则保留：

```text
1. class/id 命中结构关键词。
2. 自身有短 visible text。
3. 子树包含 heading / nav / form / button / img / svg / list / table。
4. computed style 显示为重要 layout container。
5. 是可见 visual block。
```

DOM node 格式：

```json
{
  "tag": "div",
  "selector": "div.main-container",
  "role_hint": "main_container",
  "text": "short visible text or null",
  "children": []
}
```

### 6.5 Structural Text

```text
headings:
  h1-h6

navigation_links:
  nav a
  header a
  class/id 命中 nav/menu/navbar/topbar/main-nav/header-nav 的容器内 a

buttons:
  button
  input[type=button]
  input[type=submit]
  input[type=reset]
  a with class/id containing btn/button/cta/action

forms:
  form/input/textarea/select/label/form 内部 button

sidebar_or_card_titles:
  sidebar/card/panel/summary/help-like containers 内的 heading/strong/button/短文本

footer_links:
  footer/footer-like containers 内的 a

lists:
  ul/ol 的代表性 items

tables:
  table headers, row_count, sample_rows
```

### 6.6 Layout / Style

实现方式：

```text
Python 调用 Playwright 渲染 raw_html。
浏览器内执行 JS 获取 getComputedStyle 和 getBoundingClientRect。
```

采样节点：

```text
body
header, nav, main, article, aside, section, footer
form, table
button, input, textarea, select
img, picture, svg
DOM outline 中保留的 div/span
可见 visual blocks
```

抽取字段：

```text
bbox
display
position
flex-direction
grid-template-columns
background-color
text color
font family/size/weight
border
border-radius
padding
margin
gap
```

生成 layout/style hints：

```text
horizontal flex row
vertical flex column
grid layout
two-column layout
three-column layout
wider main column with narrower sidebar
centered max-width container
card-like box with border/background
full-width navigation bar
```

### 6.7 Colors

outline 中保留 computed color：

```json
{
  "selector": ".nav-bar",
  "property": "background-color",
  "value": "rgb(92, 184, 92)",
  "hex": "#5cb85c"
}
```

outline text 中写：

```text
.nav-bar: background-color #5cb85c / rgb(92, 184, 92)
```

VLM reasoning 中自然写：

```text
green navigation bar
yellow call-to-action button
dark text on white background
```

### 6.8 Assets and Visual Blocks

覆盖：

```text
1. img / picture
2. svg
3. CSS backgrounds: background-image, gradient, visible background-color blocks
4. visual_blocks: CSS 画出来的 logo/icon/色块/占位图/装饰块
5. pseudo_visual_blocks: ::before / ::after 可见内容或视觉块
```

visual block 识别条件：

```text
元素可见，bbox 有有效面积；
文本为空或很短；
有 background-color / background-image / border / border-radius / box-shadow / explicit width-height；
class/id 命中 media 或 placeholder 关键词时提高优先级。
```

---

## 7. 结构关键词

关键词用于 DOM 保留、role_hint、major_structure、visual block 识别。

### layout / structure

```text
container, wrapper, inner, outer, layout, grid, row, rows, column, columns, col, flex, stack, section, block, region, area, content, main, article, body, page
```

### header / navigation

```text
header, head, top, topbar, masthead, brand, branding, logo, nav, navbar, navigation, menu, menubar, primary-nav, secondary-nav, main-nav, subnav, breadcrumb, breadcrumbs, tabs, tabbar
```

### hero / banner

```text
hero, banner, jumbotron, cover, splash, headline, intro, landing, promo, promotion, showcase, lead
```

### sidebar / aside

```text
sidebar, side-bar, aside, rail, right-rail, left-rail, drawer, widget, panel, summary, info-box, infobox, help-box, related, recommend, filter, filters
```

### cards / panels / sections

```text
card, cards, box, tile, panel, module, widget, feature, features, item, list-item, summary, callout, notice, alert, profile, stat, stats, pricing, plan, testimonial, review
```

### forms / actions

```text
form, field, input, textarea, select, dropdown, search, searchbox, query, contact, newsletter, subscribe, signup, sign-up, login, cta, action, button, btn, submit, request, help, call
```

### media / placeholders

```text
image, img, photo, picture, media, thumbnail, thumb, avatar, icon, icons, logo, brand, placeholder, figure, gallery, carousel, slider, video, map, background, bg
```

### footer / legal / social

```text
footer, foot, bottom, copyright, legal, privacy, terms, policy, disclaimer, social, share, follow, sponsor, partners, credits, attribution
```

### tables / data

```text
table, thead, tbody, row, cell, data, chart, graph, stats, metrics, comparison, schedule, calendar
```

---

## 8. VLM Prompt

### 8.1 System Prompt

```text
You are an expert frontend engineer and visual layout annotator.

Given a webpage screenshot, a rule-extracted HTML outline, and the original target HTML, write the reasoning text that should appear before the final HTML code in an image-to-HTML training answer.

Use a structured visual implementation analysis style. Explain the page layout, visible regions, important components, colors, visual style, assets, and CSS implementation choices needed to recreate the page as a self-contained HTML file with embedded CSS.

Do not output HTML or CSS code.
```

### 8.2 User Prompt

```text
You are given:
1. A webpage screenshot.
2. A rule-extracted HTML outline.
3. The original target HTML.

Use the screenshot for visual appearance.
Use the HTML outline for structure, visible text, layout, style, assets, and visual blocks.
Use the original target HTML for exact implemented elements.

Write only the reasoning text. A post-processing script will append the final HTML code block after your reasoning.

Required reasoning structure:
1. Start with one brief task framing sentence, such as:
   "The user wants me to convert the webpage screenshot into a single self-contained HTML file with embedded CSS. Let me analyze the image carefully."

2. Include a "Layout Analysis:" section.
   Describe the overall page layout from top to bottom, including major regions and column/grid relationships.

3. Add relevant region-specific sections according to the actual screenshot.
   Use descriptive headings such as Header Section, Navigation Section, Hero Section, Main Content Area, Sidebar, Summary/Card Section, Forms and Tables, Footer Section, or Assets and Visual Blocks when they fit the page.
   Do not force a section that is not present. Do not restrict yourself to this list if another heading better matches the page.

4. Include a "Colors Observed:" section.
   Summarize dominant colors and approximate color roles. Use approximate hex values only when helpful.

5. End with a "Structure and Implementation Plan:" section.
   Describe the HTML/CSS implementation plan. Mention embedded CSS, flexbox/grid, multi-column layout, cards, forms, tables, image placeholders, CSS-drawn blocks, and centered containers when relevant.
   If a layout relationship is ambiguous, briefly state the final implementation choice.

Content requirements:
- Focus on concrete visual and coding-relevant details.
- Mention important visible components and text groups without copying long paragraph text.
- Do not include HTML or CSS code.
- Do not include markdown code fences.

HTML outline:
<outline>
{{OUTLINE_TEXT}}
</outline>

Original target HTML:
<target_html>
{{RAW_HTML}}
</target_html>
```

---

## 9. VLM 输出与后处理

### 9.1 VLM 输出

VLM 输出 plain text：

```text
{{reasoning_text}}
```

### 9.2 后处理

```text
1. 读取完整 VLM response 作为 vlm_reasoning_raw。
2. strip 首尾空白。
3. 如果输出被包在 ```text 或其他非 HTML code fence 中，则剥离外层 fence。
4. 如果输出包含 HTML/CSS 代码、```html、<!DOCTYPE html>、<html 等明显代码内容，则标记 warning 或重试。
5. 得到 reasoning_text。
6. 拼接 final_answer。
```

---

## 10. final_answer 拼接

````text
{{reasoning_text}}

```html
{{raw_html}}
```
````

---

## 11. 质量元信息

```json
{
  "reasoning_word_count": 0,
  "contains_html_in_reasoning": false,
  "contains_markdown_fence_in_reasoning": false,
  "has_layout_analysis_section": true,
  "has_colors_observed_section": true,
  "has_structure_implementation_plan_section": true,
  "region_section_count": 0,
  "extraction_warnings": [],
  "generation_warnings": [],
  "alignment_warnings": []
}
```

基础检查：

```text
reasoning 是否为空
是否包含 Layout Analysis
是否包含 Colors Observed
是否包含 Structure and Implementation Plan
是否意外包含 HTML/CSS 代码
final_answer 是否可拼接
```

弱对齐检查：

```text
reasoning 提到 form    → outline/raw_html 中应有 form/input/textarea/select
reasoning 提到 sidebar → outline/raw_html 中应有 aside/sidebar/card/help/summary evidence
reasoning 提到 footer  → outline/raw_html 中应有 footer/footer-like evidence
reasoning 提到 table   → outline/raw_html 中应有 table evidence
reasoning 提到 image/placeholder/visual block → outline 中应有 assets 或 visual_blocks
```

---

## 12. 输出 JSONL 格式

```json
{
  "sample_id": "...",
  "image": "...",
  "raw_html": "...",
  "html_outline_json": {},
  "outline_text": "...",
  "vlm": {
    "model": "...",
    "api_base": "...",
    "temperature": null,
    "top_p": null,
    "max_tokens": null
  },
  "prompt_version": "image_to_html_cot_v8",
  "vlm_reasoning_raw": "...",
  "reasoning_text": "...",
  "final_answer": "...",
  "status": {
    "extraction": "ok|warning|failed",
    "generation": "ok|warning|failed"
  },
  "quality_metadata": {
    "reasoning_word_count": 0,
    "contains_html_in_reasoning": false,
    "contains_markdown_fence_in_reasoning": false,
    "has_layout_analysis_section": true,
    "has_colors_observed_section": true,
    "has_structure_implementation_plan_section": true,
    "region_section_count": 0,
    "extraction_warnings": [],
    "generation_warnings": [],
    "alignment_warnings": []
  }
}
```

训练时使用：

```text
Question = image
Answer   = final_answer
```

---

## 13. v0 实现范围

```text
1. 读取 image + raw_html。
2. 解析 HTML，提取 title、major structure、DOM outline、structural text。
3. 使用 Playwright 按截图尺寸或用户配置 viewport 渲染 HTML。
4. 提取 computed style、bbox、layout/style hints。
5. 提取 img/svg/background/visual blocks。
6. 生成 html_outline_json。
7. 渲染 outline_text。
8. 调 VLM 生成 reasoning_text。
9. 拼接 final_answer。
10. 输出 JSONL。
```

v0 重点：

```text
字段完整
中间结果可 debug
raw_html 不被破坏
reasoning 不包含 HTML
reasoning 具有 GLM/Kimi-like 结构化视觉分析风格
final_answer 格式稳定
```
