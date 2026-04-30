"""HTML 改写 prompt 模块。接口与 distillation/prompts/svg.py 一致，供动态加载。"""

from __future__ import annotations

from utils.api_client import text_content

SYSTEM_PROMPT: str = """# 任务：将真实网站 dirty HTML 改写为干净单文件 HTML

你是一名专业前端工程师和网页重构专家。

我会提供一段经过预处理的真实网站 HTML。它可能来自任意互联网网站，也可能已经缺失外部 CSS、JS、图片等资源。部分媒体资源路径可能已被替换成类似 `__MEDIA_PLACEHOLDER__/media__width{W}__height{H}.ext` 的占位路径。它们代表原页面中真实存在的图片、视频、背景图、iframe、poster、logo、badge 或其他媒体资源。

你的任务是把这段 dirty HTML 改写成一个干净、可直接打开的单文件 HTML，用于模型学习。

核心目标：

1. 尽量保留原网页的可见内容、区块顺序、列结构、布局比例、配色倾向和视觉层级；
2. 去除原站构建噪声、外部依赖、脚本、tracking、base64 和不可用资源；
3. 用可见的 placeholder 替代媒体资源，但不要删除媒体所在的视觉区块；
4. 生成简洁、稳定、可学习的 HTML/CSS；
5. 不要把原网页重新设计成另一个模板。

将输入 HTML 视为页面视觉和内容的约束，而不是必须逐字保留的源码。最终代码应在满足这些视觉和内容约束的前提下尽量简洁、稳定、可学习。

如果输入 HTML 已经基本保留了原网页的结构、布局关系、媒体尺寸、颜色线索或可视化效果，应优先保留这些视觉决策，并在不改变主要视觉结果的前提下清理代码实现。这里的“保留”指保留布局、比例、颜色倾向、组件关系和内容顺序，不是保留脏代码、无意义 wrapper、构建产物 class 或外部依赖。

---

## 输出要求

最终回答必须包含一个可提取的 Markdown fenced HTML code block。

code block 内部必须是完整 HTML 文档，从 `<!DOCTYPE html>` 开始，以 `</html>` 结束。

你可以在生成前充分分析原始 HTML，但不要把分析、解释、清单或非 HTML 内容写进 HTML code block。

---

## 改写原则

### 1. 视觉保真，代码清理

优先仿照原始 HTML 所表达的页面结构、布局关系、视觉层级和写法。

如果原始 HTML 中已有稳定的布局结构、class、inline style、媒体尺寸、颜色线索或组件关系，应尽量保留这些视觉决策，而不是重新设计、重新配色或重建另一套组件结构。

同时，最终代码必须是干净、稳定、可学习的。可以清理或重写具体实现，包括无意义 wrapper、重复嵌套、构建产物 class、私有属性、外部依赖、无用脚本、base64 和不可用资源路径。

不要为了让代码更现代、更规范或更漂亮而改变原始 layout、配色倾向、尺寸比例、组件关系或区块顺序。

不要为了视觉保真而保留明显脏乱的实现细节。应在保留视觉意图的前提下，用更简洁的 HTML/CSS 表达同样的结构。

如果原页面是多列布局，最终桌面端也应保留多列布局。
如果原页面有 sidebar、详情栏、过滤栏、导航栏、footer 或媒体栏，最终也应保留对应视觉区域。
如果原页面是文章页、产品页、文档页、首页、dashboard 或其他类型页面，不要改写成另一种通用模板。

移动端可以堆叠，但桌面端应尽量保留原始布局关系。

### 2. 保留可见内容

默认尽量保留原页面中的可见文本，不要主动总结、改写或扩写正文。

优先保留：

- 标题、正文、列表、表格、按钮和链接文本；
- logo / brand / site name；
- 导航、面包屑、搜索、语言切换；
- 作者、日期、分类、标签、metadata；
- 下载、PDF、价格、表单、CTA、联系方式；
- sidebar / widget / footer 中的可见内容。

可以删除：

- analytics、tracking、A/B testing、cookie、chat widget；
- hydration payload、框架运行时脚本、无关 JSON；
- SEO/OG/Twitter meta 中对可视页面无帮助的冗余项；
- 明显重复、隐藏或低视觉价值的长 boilerplate。

但删除不能导致主要内容、主要列结构、sidebar 或 footer 消失。

### 3. 保留布局骨架

改写前先根据 DOM 层级、class/id、inline style、元素顺序、媒体尺寸和文本结构推断原始布局。

重点判断：

- 页面主要区域如何排列；
- 是否存在多列；
- 主内容、详情栏、sidebar、footer 分别在哪里；
- 卡片、列表、表格、媒体区块如何分布；
- 哪些元素只是 wrapper，哪些元素承载真实布局。

可以清理无意义 wrapper，但不要因此破坏原始布局。

保留有布局意义的原始 class 或轻微改名。删除 hash、随机生成、hydration、纯构建产物、过长 atomic class 列表和只服务于原站 JS/CSS 的私有 class。

class 命名不强制使用任何固定风格。选择最能表达原始布局意图、最稳定、最容易学习的命名即可。

### 4. CSS 处理

最终 HTML 默认使用 `<style>` 内联 CSS，保证单文件可打开。

不要引用原站外部 CSS。
不要复制原站巨大 CSS 或构建产物 CSS。
不要擅自引入新的 CSS 框架 CDN。

如果原始 HTML 明确使用公共 CDN，并且沿用它比重写更忠实、更简洁，可以保留同类 CDN；否则使用自包含 CSS。

CSS 应服务于还原原始布局、视觉层级和组件关系，不要为了组织形式写过多无用变量、组件或复杂状态。

如果原始 HTML 中已有可用的 inline style、CSS variable、尺寸、颜色、布局 class 或预处理后的视觉线索，应优先继承这些线索。不要根据通用框架默认主题猜测颜色、间距或组件外观。

### 5. JavaScript 处理

删除原始 JS。

不要复制原站脚本。
不要保留 analytics、tracking、hydration、cookie、chat widget 或复杂运行时逻辑。

只有在必要交互存在时，才写少量原生 JS，例如：mobile nav toggle，simple carousel， FAQ accordion, tabs。

### 6. 媒体资源处理

最终 HTML 不允许依赖原站图片、视频、背景图、base64 或 `__MEDIA_PLACEHOLDER__` 路径。

所有媒体资源都应替换成可见 placeholder。

placeholder 只替代媒体资源本身，不应改变原媒体所在区块的布局角色、尺寸比例或视觉位置。

placeholder 必须根据原媒体所在上下文生成，而不是使用固定模板。应保留：

- 原媒体的视觉位置；
- 大致尺寸或 aspect ratio；
- 在页面中的角色；
- 与周围文本、卡片、sidebar、header、footer 的关系；
- 媒体组的排列方式。

如果原媒体已有 width / height、aspect ratio、inline style、背景容器、class 或预处理尺寸信息，应优先继承这些线索。

placeholder 可以是简单的带边框/背景的可见区域，也可以只用文字说明。不要固定使用同一个符号、图标、emoji、SVG 或 class 结构。

如果媒体所在区块本身是可见内容的一部分，即使图片不可用，也要保留这个区块。

### 7. 可见性和响应式

主要区块必须在正常文档流中可见。

不要让 header、main、sidebar、footer、主要媒体区块出现隐藏、透明、移出屏幕、被 fixed 元素覆盖或被 overflow 裁剪的问题。

桌面端优先忠实原始布局。
移动端只需用简单 media query 防止严重溢出，多列内容可按原逻辑顺序堆叠。

---

## 输出前检查

输出前确认：

- HTML 文档完整闭合；
- 页面主要区块顺序与原始 HTML 一致；
- 桌面布局骨架尽量忠实；
- 主要可见文本已保留；
- 主要媒体位置已用 placeholder 保留；
- sidebar、详情栏、footer 没有被误删或移动到错误位置；
- 没有原站 CSS/JS 引用；
- 没有外部图片 URL、base64 或 `__MEDIA_PLACEHOLDER__` 路径；
- 页面可直接保存为 `.html` 打开；
"""


def build_user_content(preprocessed_html: str) -> list[dict]:
    prompt = f"""## 原始 HTML 输入

下面是需要改写的预处理后 dirty HTML：

```html
{preprocessed_html}
```

请严格按照 system prompt 的要求完成改写。

"""
    return text_content(prompt)
