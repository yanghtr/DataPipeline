"""HTML 改写 prompt 模块。接口与 distillation/prompts/svg.py 一致，供动态加载。"""

from __future__ import annotations

from utils.api_client import text_content

# 故意留空：这一版主要依赖完整的 user prompt 来约束模型行为。
SYSTEM_PROMPT: str = ""


def build_user_content(preprocessed_html: str) -> list[dict]:
    prompt = f"""# 任务: 将真实网站 dirty HTML 改写为干净单文件 HTML

## 任务背景

你是一名专业前端工程师和网页重构专家。

我会提供一段从真实网站保存下来的原始 HTML。它可能来自任意互联网网站，例如企业官网、产品页、博客/新闻文章、学术期刊页、文档站、电商页、门户页、CMS 页面或前端框架构建产物。

原始 HTML 通常包含外部 CSS/JS、CMS wrapper、构建产物 class、hydration payload、analytics、tracking、base64 图片、广告脚本、复杂嵌套结构等噪声。

注意：你收到的并不是完全原始的网页源码，而是经过预处理的 dirty HTML。
其中原始媒体资源路径通常已经被替换成占位路径，格式类似：

`__MEDIA_PLACEHOLDER__/media__width{{W}}__height{{H}}.ext`

例如：

- `__MEDIA_PLACEHOLDER__/media__width640__height480.jpg`
- `__MEDIA_PLACEHOLDER__/media__widthunknown__heightunknown.png`

这些占位路径代表原页面中真实存在的图片、视频、音频、iframe、poster、背景图或其它媒体资源。你必须根据它们所在的上下文，在最终 HTML 中把它们改写成“可见的媒体 placeholder 区块”，而不是保留这些路径本身，也不是删除对应区块。

你的任务是将这段 dirty HTML 改写成一个干净的单文件 HTML，用于模型学习。改写结果应当：

- 可直接保存为 `.html` 并在浏览器打开；
- 不依赖原站外部 CSS / JS / 图片资源；
- 尽量保留原始网站的主要内容、区块顺序、列结构和桌面端布局骨架；
- 用 placeholder 替代图片、视频、背景图、logo、badge、counter 等媒体资源；
- 代码简洁、结构清晰，像人工前端工程师手写的 clean HTML/CSS；
- 不要把原网站重新设计成另一个更现代但布局不同的页面。

你的目标不是像素级复刻，而是在缺少外部 CSS、JS、图片和截图的情况下，基于原始 HTML 推断并重建一个结构完整、布局忠实、代码可学习的页面。

---

## 输出要求

你可以在生成前充分分析原始 HTML。

最终答案必须包含一个 Markdown fenced HTML code block，格式如下：

第一行是：

```html

code block 内部必须是完整 HTML 文档，从：

<!DOCTYPE html>

开始，并以：

</html>

结束。

最后一行是：

```

后处理程序会只提取这个 HTML code block。

不要在 HTML code block 内写分析、说明、清单或非 HTML 内容。

---

## 核心原则

### 1. 不要重新设计页面

优先仿照原始 HTML 所表达的页面结构和布局意图。

不要把三列页面改成两列或一列。
不要把 sidebar 移到 footer。
不要把文章详情栏移动到正文底部。
不要把 product grid 改成普通 feature section。
不要把学术文章页改成普通博客页。
不要删除可见的图片型 sidebar block。
不要为了让代码“更现代”而重排页面。

移动端可以堆叠，但桌面端必须尽量保留原始布局关系。

### 2. 保留原始布局骨架

改写前先推断原始页面的桌面端布局骨架：

- 页面有几列；
- 每列的视觉角色是什么；
- 哪些内容属于主内容；
- 哪些内容属于详情栏；
- 哪些内容属于 sidebar / widget；
- 哪些内容属于 footer；
- 图片、badge、卡片、列表、表格分别位于哪里。

常见布局信号包括但不限于：

- `.row`
- `.col-*`
- `.container`
- `.main`
- `.content`
- `.main_entry`
- `.entry_details`
- `.sidebar`
- `.aside`
- `.left`
- `.right`
- `.pkp_structure_main`
- `.pkp_structure_sidebar`
- `.has_sidebar`
- `.complementary`
- `.widget`
- `.block`
- `.region-sidebar`
- `.layout-sidebar`
- `role="main"`
- `role="complementary"`
- `<aside>`

如果原始页面同时包含文章主内容、文章详情栏和站点 sidebar/widget 栏，最终 HTML 必须保留这三个独立区域。

### 3. 保留主要可见区块

内部建立页面区块清单，并确保最终 HTML 覆盖主要可见区块。

常见区块包括但不限于：

- topbar / announcement bar
- header / logo / navigation
- language switcher
- search
- breadcrumbs
- hero / banner / carousel
- main content
- article metadata
- article details
- sidebar / widget
- product grid / feature grid
- filters / facets
- table / chart / card panel
- CTA / contact / newsletter
- related content
- footer

不要因为原始 HTML 很长、嵌套很深、区块在文档靠后、class 很脏、依赖外部资源、图片是 base64、sidebar 看起来像 badge、footer 看起来不重要，就省略主要区块。

如果原站有 footer，最终 HTML 必须有可见 footer。
如果原站有 sidebar，最终 HTML 必须有可见 sidebar 或移动端堆叠后的对应区域。
如果原站有多列桌面布局，最终 HTML 必须保留多列桌面布局。

---

## 内容保留要求

默认尽量保留原始页面中的可见文本，不要主动总结正文内容。

必须优先保留：

1. 页面 title 和 description；
2. logo / brand name / site name；
3. 顶部公告、联系信息、语言切换；
4. 主导航一级菜单；
5. breadcrumbs；
6. h1 / h2 / h3 标题文本；
7. 主要段落、列表、表格内容；
8. 作者、日期、发布信息、分类、tags；
9. DOI、PDF、下载、issue、license、metrics 等文章详情信息；
10. 产品 / 服务 / feature 卡片标题；
11. 主要 CTA 按钮文字和链接；
12. 表单的主要字段和按钮；
13. sidebar widget 标题与主要内容；
14. 联系方式；
15. footer 中的公司信息、地址、email、电话、法律链接、隐私链接等。

可以删除或简化：

- SEO / OG / Twitter meta；
- structured data；
- cookie banner；
- chat widget；
- analytics；
- A/B testing；
- tracking script；
- scroll animation；
- parallax；
- 复杂 carousel 动画；
- 过深的 dropdown submenu；
- 明显重复内容；
- 极长且低视觉价值的法律说明、投稿说明或 boilerplate 文本。

但简化不能导致主要页面区块、主要列结构、正文内容或可见 sidebar 消失。

---

## CSS 与技术来源处理

默认情况下，最终 HTML 使用一个 `<style>` 标签写自包含 CSS，不依赖原站 CSS。

只有当原始 HTML 明确、直接使用公共 CSS 框架 CDN 时，才可以沿用相同 CDN，例如：

- `https://cdn.tailwindcss.com`
- `https://cdn.jsdelivr.net/npm/bootstrap`
- `https://unpkg.com/bootstrap`
- `https://cdnjs.cloudflare.com/.../bootstrap`
- `https://cdn.jsdelivr.net/npm/bulma`

如果原始 HTML 引用的是以下类型之一，则视为站点主题或构建产物，不能直接沿用，也不要擅自引入 Bootstrap/Tailwind CDN：

- `/assets/*.css`
- `/static/*.css`
- `/_next/static/*.css`
- `/typo3temp/*.css`
- `/typo3conf/*.css`
- `/wp-content/themes/*.css`
- `/wp-content/plugins/*.css`
- `/public/journals/*.css`
- `stylesheet?name=...`
- 文件名带 hash 的 CSS
- CMS theme CSS
- sitepackage CSS
- Bootstrap Package 编译 CSS
- Tailwind 编译 CSS
- CSS-in-JS runtime 样式
- 大量 `@property`、`@layer`、utility class 或 reset 产物
- 其它明显依赖原站构建系统的 CSS

这种情况下，应根据原始 DOM、class 名、inline style、图片尺寸和文本结构，重新写简洁 CSS。

---

## HTML 改写要求

最终 HTML 必须是完整单文件：

- 包含 `<!DOCTYPE html>`
- 包含 `<html lang="...">`
- 包含 `<head>`
- 包含 `<meta charset="UTF-8">`
- 包含 `<meta name="viewport" content="width=device-width, initial-scale=1.0">`
- 包含 `<title>`
- 如果原站有 description，保留 `<meta name="description">`
- 包含 `<body>`
- 如果原站有主要内容区，使用 `<main>`
- 如果原站有 footer，使用 `<footer>`

使用合适的语义化 HTML 标签，例如：

- `<header>`
- `<nav>`
- `<main>`
- `<section>`
- `<article>`
- `<aside>`
- `<footer>`
- `<address>`
- `<figure>`
- `<figcaption>`

删除无意义的 CMS / framework wrapper 和私有属性。

保留有语义或功能价值的属性，例如：

- `href`
- `alt`
- `title`
- `aria-*`
- `role`
- `type`
- 必要的 `id`

---

## class 和 CSS 写法

不要强行套用某一种 class 命名风格，例如 BEM、Tailwind utility 或 Bootstrap 风格。

根据原始页面的写法和布局意图，选择最自然、稳定、容易学习的 class 命名。

允许：

- 沿用原始 HTML 中有语义、有布局信息的 class；
- 对有意义的原始 class 做轻微清理或重命名；
- 使用简洁语义 class，例如 `.site-header`、`.main-nav`、`.article-main`、`.article-details`、`.sidebar`、`.sidebar-block`、`.product-card`；
- 使用少量通用 class，例如 `.container`、`.button`、`.media-placeholder`；
- 使用 `.row`、`.column`、`.main-column`、`.side-column` 等简单布局 class，只要它们有助于保留原始布局。

应该删除或改写：

- hash class；
- 随机生成 class；
- hydration class；
- 纯构建产物 class；
- 过长 atomic utility class 列表；
- 无意义 CMS wrapper class；
- 只服务于原站 JS/CSS 的私有 class。

核心原则：

- class 应服务于复刻原始布局，而不是服务于某种命名教条；
- 能清晰表达结构和布局即可；
- 不要为了组织形式增加过多无用 CSS；
- 不要把原始页面中有布局意义的 class 全部删除后重新发明另一套布局。

---

## JavaScript 处理

删除所有原始 JS。

禁止复制原站 JS。
禁止引用原站外部 JS。
禁止保留 analytics、tracking、chat widget、cookie、A/B testing、hydration 脚本。

只有在必要交互存在时，才写少量原生 JS，例如：

- mobile nav toggle
- simple carousel
- FAQ accordion
- tabs

如果交互复杂，可以简化为静态可见内容。
不要为了保留复杂交互而牺牲页面完整性。

---

## 图片、视频、媒体和 badge 处理

最终 HTML 不允许依赖原站图片、视频或背景图文件。

必须将所有媒体资源替换为可见 placeholder，包括：

- `<img>`
- `<picture>`
- `<source>`
- `<video>`
- `poster`
- inline style 中的 `background-image`
- `<style>` 中的 `background-image`
- carousel background image
- hero background image
- product image
- logo image
- article cover image
- chart image
- icon image
- badge image
- base64 image
- visibly rendered counter / banner / widget image
- 预处理后变成 `__MEDIA_PLACEHOLDER__/media__width{{W}}__height{{H}}.ext` 的任何媒体路径

placeholder 必须保留原资源的视觉角色、位置和比例。

如果原始元素有 width / height，使用相同或相近 aspect ratio。
如果没有尺寸，根据上下文估计合理比例。

不要使用 base64。
不要使用外部图片 URL。
不要使用原站图片 URL。
不要在最终 HTML 中保留 `__MEDIA_PLACEHOLDER__` 路径字符串作为实际资源引用。
可以保留简洁内联 SVG 图标，但不要复制复杂大图。

示例：

<div class="media-placeholder media-placeholder--hero">
  <span class="media-placeholder__icon">◇</span>
  <span>Hero image</span>
</div>

---

## 图片型区块保留规则

替换图片不等于删除图片所在区块。

如果原始页面中某个可见区块主要由图片、badge、logo、counter 或 banner 组成，必须保留该区块，并用 placeholder 表示。

例如：

- journal logo
- partner / collaboration logo
- license badge
- article template image
- indexing badges
- recommended tool badges
- flag counter
- sponsor logo
- certification badge
- app store badge
- payment badge
- sidebar banner
- chart / infographic

对于这类区块，最终 HTML 应保留：

- block title
- approximate number of images or badges
- visual position
- approximate aspect ratio
- stacked / grid / inline arrangement

不要因为图片是 base64 就删除整个 block。
不要因为图片来自外部链接就删除整个 block。
不要因为图片看起来像 badge 或 counter 就当作 tracking 噪声删除。

---

## 可见性和响应式

所有主要区块必须在正常文档流中可见。

禁止对主要区块使用：

- `display: none`
- `visibility: hidden`
- `opacity: 0`
- `height: 0`
- `max-height: 0`
- `overflow: hidden` 导致整个区块被裁剪
- `position: absolute` 并移出屏幕
- 负 margin 导致区块不可见
- fixed header 覆盖 main 或 footer

如果使用 fixed header，必须为页面内容设置足够 spacing，避免内容被遮挡。

桌面端优先保留原始布局。
移动端只需用简单 media query 防止严重溢出；多列内容可以按原逻辑顺序堆叠。

如果原网站有 footer，则改写后 footer 必须可见、不被覆盖，并保留主要 footer links 或联系信息。

如果原网站有 sidebar，则改写后 sidebar 必须在桌面端作为独立视觉区域出现，不被移动到 footer，不被隐藏，并保留主要 widget blocks。

---

## 长页面处理

默认尽量保留原始可见内容，不主动总结正文或重写为摘要。

只有在输出长度明显不足、内容高度重复或属于低视觉价值 boilerplate 时，才可以压缩：

- 过深 dropdown submenu；
- 重复产品说明；
- 重复 card；
- 极长 license 文本；
- 投稿指南中的超长步骤；
- SEO meta；
- 非核心小组件。

但不要删除：

- header
- main
- footer
- h1
- breadcrumbs
- 主要正文
- 主要 CTA
- 文章 metadata
- 主要 sidebar
- 主要图片 / badge / logo / counter block placeholder
- 联系方式

---

## 输出前检查

输出前确认：

- HTML 从 `<!DOCTYPE html>` 开始；
- HTML 以 `</html>` 结束；
- `<head>` 和 `<body>` 存在；
- 如果原站有 main content，最终 HTML 有 `<main>`；
- 如果原站有 header，最终 HTML 有 `<header>`；
- 如果原站有 footer，最终 HTML 有 `<footer>`；
- 如果原站有 sidebar，最终 HTML 有 sidebar 或移动端对应区域；
- 如果原站有多列桌面布局，最终桌面端保留多列布局；
- 如果原站有文章详情栏和独立 sidebar，最终保留二者为不同区域；
- 主要 section 顺序与原始页面一致；
- 主要 image / badge / logo / counter block 已用 placeholder 表示；
- 没有引用原站 CSS；
- 没有引用原站 JS；
- 没有外部图片 URL；
- 没有 base64 图片；
- 没有复制原始巨大 CSS；
- 没有复制原始 JS；
- 主要区块不隐藏、不透明度为 0、不被移出屏幕；
- 移动端不会严重溢出；
- 代码缩进清晰；
- 页面可以直接保存为 `.html` 并用浏览器打开。

---

## 原始 HTML 输入

下面是原始 dirty HTML：

{preprocessed_html}
"""
    return text_content(prompt)
