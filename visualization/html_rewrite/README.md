# HTML 改写结果对比可视化工具

多列并排对比 HTML 改写 pipeline 各阶段的渲染效果：预处理 HTML、不同配置的模型改写 HTML，以及原始网站内容。

## 快速开始

```bash
# 安装依赖
pip install flask loguru

# 单文件：自动生成 preprocessed + output 两列
python visualization/html_rewrite/viewer.py \
    --jsonl outputs/html_rewrite/output.jsonl \
    --sample-n 20

# 在浏览器中打开
# http://localhost:7862
```

WSL 下从 Windows 浏览器访问，加 `--host 0.0.0.0`，然后访问 `http://<WSL_IP>:7862`。

## 列结构

| 列类型 | 来源 | 说明 |
|--------|------|------|
| 预处理列 (`preprocessed_html`) | 第一个 JSONL 文件 | 媒体 placeholder 替换 + 内容截断后的 HTML |
| 改写列 (`output_html`) | 每个 JSONL 文件各一列 | 模型输出的改写 HTML |

**第一个 JSONL** 同时生成预处理列 + 改写列；**后续 JSONL** 只生成改写列（预处理相同，避免重复）。

界面左侧有固定信息栏，显示每行的 URL、页面类型、预处理统计摘要（字符数、压缩比、媒体数等）。

默认界面还会额外插入一列 **原站代理预览**：

| 列类型 | 来源 | 说明 |
|--------|------|------|
| 原站代理预览 | 第一列 URL / final_url | 由本地 Flask 代理抓取原始网站并重写资源链接，尽量以内嵌方式展示真实页面 |

每个可视化卡片支持两种视图：

- `预览`：按桌面浏览器视口（默认 `1440×900`）渲染，再整体缩放到当前列宽
- `HTML`：直接查看对应页面的 HTML 源码，带行号和基础语法高亮，方便排查 DOM / 资源差异
- `完整回复`：查看模型原始回复全文（如果 JSONL 中保存了 `response` / `raw_response` 等字段）

代码视图默认保持单行并显示横向滚动条；点击 `Break Line` 可切换为自动换行。

卡片底部统计口径：

- `↑ 输入 xxx tok`：API `usage.prompt_tokens`，包含 system prompt、user 包装文本和 `preprocessed_html`
- `↓ API输出 xxx tok`：API `usage.completion_tokens`
- `thinking xxx chars`：单独保存下来的 reasoning 文本长度
- `正文 xxx chars`：`response`，即 `message.content` 的字符长度
- `HTML xxx chars`：最终 `output_html` 的字符长度

对于当前默认本地后端（通常是 vLLM OpenAI-compatible server）：

- 如果当前记录检测到 `thinking`，则 `↓ API输出` 通常应理解为 `thinking + 正文` 的总输出 token
- 如果后端未来返回 `completion_tokens_details.reasoning_tokens` 等细分字段，界面会优先补充显示这些更细的 token 统计

顶部工具栏支持：

- 切换每页条数
- 切换桌面预览屏幕尺寸
- `Hide / Show 列` 按钮：隐藏 / 恢复信息列、原站列和任意结果列

## 外部资源加载机制

改写后的 HTML 中包含指向原始网站的相对路径 CSS/图片/JS。`/render/<col>/<row>` 端点在 `<head>` 中自动注入：

```html
<base href="https://example.com/">
```

这样 `<img src="/logo.png">` 等相对路径会解析到原始服务器，外部资源能正常加载。页面在 `<iframe src="/render/...">` 中以完整页面形式加载，**不加 sandbox 限制**，等效于在新标签打开该 HTML 文件。

对于原站内容，工具不再直接把远端 URL 塞进 iframe，而是通过：

- `/live/<row>`：抓取这一行的原始网页 HTML
- `/proxy/<scheme>/<host>/...`：代理 HTML 中的资源和站内跳转

这样可以绕过很多站点的 `X-Frame-Options` / `frame-ancestors` 限制，比直接 iframe 外链更鲁棒。

> **注意**：登录态强依赖、Service Worker、复杂 SPA 或强风控站点仍可能和真实标签页有差异；这时点击卡片右上角或信息栏中的「打开原站」在新标签查看即可。

## CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--jsonl FILE [FILE ...]` | — | 一个或多个 JSONL 文件路径 |
| `--sample-n N` | `20` | 每个文件最多加载 N 条，`-1` 加载全部 |
| `--random-sample` | 关 | 随机采样（否则取前 N 条） |
| `--port PORT` | `7862` | HTTP 端口 |
| `--host HOST` | `0.0.0.0` | 监听地址 |

## 对比多个模型配置

```bash
python visualization/html_rewrite/viewer.py \
    --jsonl outputs/output_gpt4o.jsonl outputs/output_claude.jsonl \
    --sample-n 30
```

生成列：`output_gpt4o [preprocessed]` · `output_gpt4o [output]` · `output_claude [output]`

后续 JSONL 按第一个文件的行 ID 对齐，相同 URL 排在同一行。

## 输入 JSONL 格式

Stage 2 (`stage2_rewrite.py`) 的输出格式，每行一条 JSON：

```json
{
  "id": "https://example.com/page",
  "_meta": {
    "url": "https://example.com/page",
    "final_url": "https://example.com/page",
    "page_type": ["article"],
    "part": "train"
  },
  "preprocessed_html": "<!DOCTYPE html>...",
  "output_html": "<!DOCTYPE html>...",
  "preprocess_stats": { "original_chars": 45000, "cleaned_chars": 12000, ... },
  "model": "gpt-4o",
  "prompt_tokens": 3500,
  "completion_tokens": 2100,
  "finish_reason": "stop"
}
```

Stage 1 (`stage1_preprocess.py`) 的输出只有 `preprocessed_html`，同样支持加载（自动检测可用字段）。

## 动态添加列

启动后在界面右上角「+ 添加列」面板中：

1. 填入 JSONL 文件路径（绝对路径或相对于工作目录）
2. 选择字段（`preprocessed_html` / `output_html`）
3. 设置加载条数和是否随机采样
4. 填入列标签（可选），点击「添加」

新列按第一列的 ID 顺序对齐。点击列标题右侧的 ✕ 可移除列。

## 交互说明

1. `Hide / Show 列`：控制显示哪些列，适合只保留“原站 + 某一个模型输出”做聚焦对比
2. 卡片顶部 `预览 / HTML / 完整回复`：在渲染结果、HTML 源码和模型原始回复之间切换
3. `预览屏幕`：修改虚拟桌面视口尺寸，默认 `1440×900`
4. 代码视图里的 `Break Line`：在“横向滚动”和“自动换行完整显示”之间切换

## 键盘快捷键

| 按键 | 操作 |
|------|------|
| `←` / `→` | 翻页 |
