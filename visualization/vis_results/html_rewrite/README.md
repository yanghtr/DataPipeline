# HTML 改写结果对比可视化工具

多列并排对比 HTML 改写 pipeline 各阶段的渲染效果：预处理 HTML、不同配置的模型改写 HTML。

## 快速开始

```bash
# 安装依赖
pip install flask loguru

# 单文件：自动生成 preprocessed + output 两列
python visualization/vis_results/html_rewrite/viewer.py \
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

## 外部资源加载机制

改写后的 HTML 中包含指向原始网站的相对路径 CSS/图片/JS。`/render/<col>/<row>` 端点在 `<head>` 中自动注入：

```html
<base href="https://example.com/">
```

这样 `<img src="/logo.png">` 等相对路径会解析到原始服务器，外部资源能正常加载。页面在 `<iframe src="/render/...">` 中以完整页面形式加载，**不加 sandbox 限制**，等效于在新标签打开该 HTML 文件。

> **注意**：直播网站当前页面无法嵌入（绝大多数现代站点设置了 `X-Frame-Options` 拒绝 iframe 嵌入），点击信息栏的「打开原网站」链接在新标签查看。

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
python visualization/vis_results/html_rewrite/viewer.py \
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

## 键盘快捷键

| 按键 | 操作 |
|------|------|
| `←` / `→` | 翻页 |
