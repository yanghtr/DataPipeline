# visualization/distillation — SVG 蒸馏结果对比工具

多列对比可视化：动态加载多个蒸馏结果 JSONL，按 `id` 字段对齐，逐行对比 SVG 渲染效果与原始代码。

---

## 文件说明

```
visualization/distillation/
  viewer.py     Flask 后端：加载/对齐多列 JSONL，提供 REST API
  viewer.html   前端页面：多列表格、SVG 渲染、代码展示、动态增减列
  README.md     本文件
```

---

## 依赖安装

```bash
pip install flask loguru
```

---

## 快速启动

```bash
# 对比两个模型的蒸馏结果（取前 200 条）
python visualization/distillation/viewer.py \
  --jsonl /path/to/model_a.jsonl /path/to/model_b.jsonl \
  --sample-n 200

# 随机采样 500 条
python visualization/distillation/viewer.py \
  --jsonl /path/to/model_a.jsonl /path/to/model_b.jsonl \
  --sample-n 500 --random-sample

# 不带初始文件，在 UI 中动态添加列
python visualization/distillation/viewer.py --port 7861
```

启动后在浏览器打开终端输出的地址（默认 `http://127.0.0.1:7861`）。

---

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--jsonl` | 无 | 初始 JSONL 文件路径（可多个，空格分隔）。也可不填，在 UI 中动态添加 |
| `--sample-n` | 500 | 每列加载条数。`-1` 表示全部加载（谨慎用于大文件） |
| `--random-sample` | 关 | 启用后用水库采样（Algorithm R）随机抽取 N 条；未启用则取文件前 N 条 |
| `--port` | 7861 | HTTP 服务端口 |
| `--host` | 127.0.0.1 | 监听地址。WSL 下若需从 Windows 访问，改为 `0.0.0.0` |

---

## 输入文件格式

工具支持两种 JSONL 格式，可混用（例如第一列加载 query seed 作为原始基线，后续列加载不同模型的蒸馏结果）。

### 1. 蒸馏输出格式（`distillation/` 模块产出）

```json
{
  "id": "stage1_icon/text2svg/data_000000:312",
  "instruction": "Draw an orange circle...",
  "status": "ok",
  "response": "```svg\n<svg xmlns=\"...\" viewBox=\"...\">...</svg>\n```",
  "model": "your-model",
  "prompt_tokens": 42,
  "completion_tokens": 512,
  "finish_reason": "stop",
  "error": null
}
```

- **id**：顶层 `id` 字段
- **SVG**：从 `response` 字段提取，自动剥除 ` ```svg ... ``` ` markdown fence

### 2. Query seed 格式（种子筛选产出，`high_priority_pool.jsonl` 等）

```json
{
  "instruction": "Draw an orange circle...",
  "_meta": {
    "id": "stage1_icon/text2svg/data_000000:312",
    "gt_svg": "```svg\n<svg xmlns=\"...\" viewBox=\"...\">...</svg>\n```"
  }
}
```

- **id**：`_meta.id` 字段
- **SVG**：从 `_meta.gt_svg` 字段提取，自动剥除 markdown fence

---

## 采样与对齐规则

- **第一列是基准列**：只对第一个加载的文件执行采样（前 N 条或随机采样）
- **后续列不再独立采样**：而是扫描整个文件，按第一列采样出的 `id` 精确匹配
- **行顺序固定跟随第一列**：若某列缺少某个 `id`，该单元格显示占位符「—」

---

## 界面操作

### 工具栏

- **‹ / › 按钮**：翻页（也支持键盘 `←`/`→` 键）
- **每页下拉框**：切换每页显示条数（10 / 20 / 50）

### 添加列

在「+ 添加列」栏中：

1. 填写 JSONL 文件的**绝对路径**
2. 设置采样数（默认 500，`-1` = 全部）
3. 勾选「随机采样」或留空（取前 N 条）
4. 点击「添加」或按 `Enter`

每次添加会在表格最右侧追加一列，行对齐自动重新计算。

### 删除列

点击列头右上角的「×」按钮可删除该列，行对齐自动重新计算。

### 每行内容

每条数据独占一行：

- **左侧固定列**（260px）：行号、id（截断显示，悬停查看全文）、instruction 文本
- **每个数据列**（340px）：
  - **上半部分**：SVG 渲染（iframe 沙箱）。无 SVG 时显示状态占位符
  - **下半部分**：状态徽章（ok / error / missing）、模型名称、SVG 原始代码
    - 代码区支持独立滚动
    - 勾选「转义」后将代码切换为 JSON 转义形式（换行显示为 `\n`、双引号显示为 `\"`），方便与原始 JSONL 文件对照

### SVG 渲染说明

- 渲染区域尺寸：240px 高，340px 宽，SVG 自动等比缩放至 `max-width: 100%; max-height: 228px`
- 渲染在 `<iframe sandbox="allow-scripts">` 中进行，与主页面隔离
- 服务端在提取 SVG 时会自动剥除 `<script>` 标签

---

## 常用示例

```bash
# 对比原始 canonical 数据 vs 蒸馏结果
python visualization/distillation/viewer.py \
  --jsonl \
  /data/processed/SAgoge/stage1/icon/generation/text2svg/data_000000.jsonl \
  /data/distillation/SAgoge/svg_responses.jsonl \
  --sample-n 200

# 对比三个模型的蒸馏结果，随机采样 300 条
python visualization/distillation/viewer.py \
  --jsonl \
  /data/distillation/model_a.jsonl \
  /data/distillation/model_b.jsonl \
  /data/distillation/model_c.jsonl \
  --sample-n 300 --random-sample

# 全量加载（小文件）
python visualization/distillation/viewer.py \
  --jsonl /data/distillation/test_run.jsonl \
  --sample-n -1

# WSL 下从 Windows 浏览器访问
python visualization/distillation/viewer.py \
  --jsonl /path/to/file.jsonl \
  --host 0.0.0.0 --port 7861
```

---

## REST API

工具后端提供以下接口（供调试或外部集成）：

| Method | Path | 说明 |
|--------|------|------|
| `GET` | `/` | 前端页面 |
| `GET` | `/api/info` | 列元数据 + 行数 + 对齐模式 |
| `GET` | `/api/rows?page=0&size=20` | 分页行数据（每行含所有列的 cell） |
| `POST` | `/api/columns` | 添加列 `{"path":"...", "sample_n":500, "random":false}` |
| `DELETE` | `/api/columns/<idx>` | 删除第 idx 列（0-based） |

---

## 与 panguml/viewer.py 的差异

| 对比项 | `panguml/viewer.py` | `distillation/viewer.py` |
|-------|---------------------|--------------------------|
| 列数 | 1 列固定 | N 列，UI 动态增减 |
| 主要内容 | 多模态 canonical schema 浏览 | SVG 渲染对比 |
| 行对齐 | 顺序索引 | 按 `id` 字段精确对齐 |
| 图片 | 支持，通过 `/api/image` | 无（仅 SVG 文本） |
| 默认端口 | 7860 | 7861 |
