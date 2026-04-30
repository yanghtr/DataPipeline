# Minimal HTML Preprocess Spec for Dirty-HTML Rewriting

## 目标

将真实网站原始 dirty HTML 做最小预处理，使其更适合后续大模型改写为 clean HTML。

预处理只处理确定会造成上下文膨胀、资源复制风险或格式混乱的问题。不要在预处理阶段重写结构、清洗 class、改 layout、压缩导航、合并 sidebar，或替模型做语义化重构。

---

## 核心原则

1. 默认保留原始 DOM 结构。
2. 默认保留原始 class、id、inline style、外部 CSS link、外部 script src、nav、sidebar、footer、form 结构。
3. 只替换明显过长或不可学习的内容。
4. 图片、视频、音频、iframe 等媒体资源仍保持原生 HTML 标签，只替换资源路径。
5. 预处理后的 HTML 应尽量像原始 HTML，而不是自定义中间格式。
6. 清洗过程必须记录统计信息，方便后续根据真实分布调整阈值。

---

## 必须处理的内容

### 1. 媒体资源路径替换

所有图片、视频、音频、iframe、embed、object、CSS `url(...)`、base64 资源路径都统一替换为 placeholder 路径，避免后续模型直接复制原站资源 URL 或 base64。

适用位置：

- `<img src="...">`
- `<source src="...">`
- `<source srcset="...">`
- `<video src="...">`
- `<video poster="...">`
- `<audio src="...">`
- `<iframe src="...">`
- `<embed src="...">`
- `<object data="...">`
- inline style 中的 `url(...)`
- CSS 中的 `url(...)`
- `data:image/...;base64,...`
- `data:video/...;base64,...`
- `data:audio/...;base64,...`

替换要求：

- 保留原标签。
- 保留原有 `alt`、`class`、`id`、`style`、`width`、`height`、`controls`、`loading` 等属性。
- 只替换资源路径，不改写标签结构。
- 如果原标签已有 `width` / `height`，使用原始宽高。
- 如果是 base64 图片，并且可以解析出宽高，使用解析出的宽高。
- 如果无法得到宽高，使用 `unknown`。
- 不要在 placeholder 路径中加入原始文件名、语义标签、bytes、mime 等复杂信息。
- 普通资源和 base64 资源使用同一种 placeholder 命名格式。

统一路径格式：

```html
__MEDIA_PLACEHOLDER__/media__width640__height480.jpg
```

未知宽高：

```html
__MEDIA_PLACEHOLDER__/media__widthunknown__heightunknown.jpg
```

如果原始扩展名可识别，保留扩展名，例如 `.jpg`、`.png`、`.webp`、`.gif`、`.svg`、`.mp4`、`.webm`、`.mp3`、`.pdf`。  
如果无法识别扩展名，使用 `.media`。

对于 `srcset`，不需要保留全部候选 URL；可以替换为单个 placeholder，并保留原标签其它属性。

---

### 2. 是否下载媒体资源以获取尺寸

默认不下载任何媒体资源。

原因：

- 网络请求会显著增加复杂度和不稳定性。
- 很多资源链接会失效、限流、需要 cookie、跨域受限或访问很慢。
- 预处理主线应保持离线、稳定、可复现。
- 后续模型主要需要知道资源位置和大致角色，不一定需要精确像素尺寸。

可选增强：

如果显式开启 `fetch_media_size=true`，且图片缺少 `width` / `height`，可以尝试下载图片头部或小范围内容解析尺寸。

要求：

- 只对图片启用，不对视频、音频、iframe 默认启用。
- 必须设置短 timeout。
- 必须限制最大下载字节数。
- 请求失败时不得影响主流程。
- 不保存资源文件。
- 只用于补充 `width` / `height`。
- 默认关闭。

---

### 3. 超长 inline script

外部 script 标签保留，不删除：

```html
<script src="..."></script>
```

inline script 如果内容长度超过 4096 characters，则替换 script 内容。

替换后保留 `<script>` 标签本身，并添加信息属性：

```html
<script data-inline-script-truncated="true" data-original-chars="48231"></script>
```

如果 inline script 不超过 4096 characters，保留原样。

---

### 4. 超长 JSON / hydration payload

常见的大型 hydration 或数据 payload 必须截断，例如：

- `__NEXT_DATA__`
- Nuxt payload
- Remix hydration data
- serialized state
- large `application/json` script

如果内容超过 4096 characters，替换内容并保留标签：

```html
<script id="__NEXT_DATA__" type="application/json" data-json-payload-truncated="true" data-original-chars="98213"></script>
```

---

### 5. 超长 inline style

`<style>` 默认保留。

只有当单个 `<style>` 内容超过 32768 characters 时，才替换内容：

```html
<style data-inline-style-truncated="true" data-original-chars="83542">
/* original inline CSS truncated */
</style>
```

外部 CSS link 必须保留：

```html
<link rel="stylesheet" href="...">
```

不要解析 CSS。
不要做 CSS 属性白名单。
不要删除 inline style 属性。

---

### 6. 超长 hidden input value

保留 hidden input 标签。

仅当 `input[type="hidden"]` 的 `value` 超过 4096 characters 时，替换 value：

```html
<input type="hidden" name="__VIEWSTATE" value="__LONG_HIDDEN_VALUE_TRUNCATED_CHARS_58231__">
```

短 hidden input 保留原样。

可见表单字段不得删除。

---

### 7. 超长 HTML comments

HTML comments 默认保留。

仅当单条 comment 超过 1024 characters 时，替换为短 comment：

```html
<!-- original comment truncated, chars=12000 -->
```

不要做 comment 白名单。
不要按内容类型删除 comment。

---

### 8. HTML 格式标准化

需要对 HTML 做基础格式标准化，降低原始代码格式混乱带来的噪声。

要求：

- 使用 HTML parser 解析并重新序列化。
- 修复明显未闭合标签。
- 统一缩进。
- 移除连续多余空行。
- 去掉标签间无意义的大段空白。
- 不改变 DOM 结构。
- 不重排节点顺序。
- 不改写标签语义。
- 不删除原始 class、id、style、data-* 属性。
- 不把 dirty HTML 改写成 semantic clean HTML。

格式标准化只解决可读性问题，不做结构清洗。

---

## 不要做的处理

不要执行以下操作：

- 不要删除所有 `<script>`。
- 不要删除外部 CSS link。
- 不要删除 class。
- 不要做 class 白名单或黑名单过滤。
- 不要删除 inline style。
- 不要做 style 属性白名单。
- 不要删除 nav / menu。
- 不要压缩 nav / menu。
- 不要删除 sidebar / aside / widget。
- 不要删除 footer。
- 不要删除 form。
- 不要删除 hidden input 标签本身。
- 不要把 `<img>`、`<video>`、`<audio>`、`<iframe>` 改成自定义 `<media-placeholder>` 标签。
- 不要重排 DOM。
- 不要把 dirty HTML 改写成 semantic clean HTML。
- 不要合并 main content、article details、sidebar。
- 不要根据页面类型做复杂规则分支。
- 不要默认下载媒体资源。

---

## 固定阈值

| 内容类型 | 处理阈值 |
|---|---:|
| inline script | > 4096 chars |
| JSON / hydration payload | > 4096 chars |
| hidden input value | > 4096 chars |
| HTML comment | > 1024 chars |
| inline `<style>` | > 32768 chars |

这些阈值的目标是只处理明显会造成上下文风险的大块内容，避免过度清洗。

---

## 统计记录要求

每次清洗必须记录统计信息，方便后续分析真实网页分布并调整阈值。

至少记录以下统计：

### 基础长度

- 原始 HTML 字符数
- 清洗后 HTML 字符数
- 压缩比例

### 媒体资源统计

- 媒体资源总数
- 图片数量
- 视频数量
- 音频数量
- iframe / embed / object 数量
- base64 媒体资源数量
- 普通媒体资源数量
- 有 width / height 的媒体资源数量
- 缺失 width / height 的媒体资源数量
- media placeholder 替换数量
- 如果启用媒体尺寸下载，记录：
  - 尝试下载数量
  - 下载成功数量
  - 下载失败数量
  - 超时数量

### script 统计

- 外部 script 数量
- inline script 数量
- inline script 长度分布
- 超过 4096 chars 的 inline script 数量和比例
- 被截断的 inline script 数量

### JSON / hydration payload 统计

- JSON / hydration payload 数量
- 长度分布
- 超过 4096 chars 的数量和比例
- 被截断数量

### style 统计

- 外部 CSS link 数量
- inline `<style>` 数量
- inline `<style>` 长度分布
- 超过 32768 chars 的 style 数量和比例
- 被截断的 style 数量

### hidden input 统计

- hidden input 数量
- hidden input value 长度分布
- 超过 4096 chars 的数量和比例
- 被截断数量

### comment 统计

- comment 数量
- comment 长度分布
- 超过 4096 chars 的数量和比例
- 被截断数量

### 格式标准化统计

- 是否成功解析 HTML
- 是否发生 parser 修复
- 清洗前后节点数量变化
- 清洗前后主要标签数量变化，例如 `header`、`main`、`footer`、`nav`、`aside`、`img`、`video`、`audio`、`iframe`、`script`、`style`

长度分布可以记录为直方图。

---

## 最终要求

预处理后的 HTML 应满足：

1. 原始 DOM 结构尽可能保留。
2. 原始 layout 线索尽可能保留。
3. base64 内容不再出现。
4. 媒体资源路径统一替换为 `__MEDIA_PLACEHOLDER__/media__width{W}__height{H}.ext`。
5. 超长 inline script / payload / style / hidden value / comment 被替换。
6. 媒体标签仍保持原生 HTML 形态。
7. HTML 格式被标准化，但结构不被重写。
8. 清洗过程有完整统计记录。
9. 后续大模型仍能从 HTML 中判断页面类型、布局、媒体位置、sidebar、footer 和主要内容。
