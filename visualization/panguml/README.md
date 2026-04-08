# visualization

多模态 canonical schema 数据集可视化工具。

读取转换后的 JSONL 文件，在浏览器中逐条浏览样本，支持图像显示、代码高亮、SVG/HTML 实时渲染。

---

## 文件说明

```
visualization/
  viewer.py     Flask 后端：数据加载、采样、API 服务
  viewer.html   前端页面：由 Flask 直接托管，无需单独部署
  README.md     本文件
```

---

## 依赖安装

```bash
pip install flask loguru
```

---

## 启动命令

```bash
python visualization/viewer.py \
  --jsonl <JSONL文件路径> \
  --image-root <图片根目录> \
  [--sample-n 500] \
  [--random-sample] \
  [--port 7860] \
  [--host 127.0.0.1]
```

启动后在浏览器打开终端输出的地址（默认 http://127.0.0.1:7860）。

---

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--jsonl` | 必填 | 输入 JSONL 文件路径 |
| `--image-root` | 无 | 图片根目录。`image.relative_path` 以此目录为基准拼接完整路径 |
| `--sample-n` | 500 | 加载样本数。`-1` 表示全部加载（谨慎用于大文件） |
| `--random-sample` | 关 | 启用后用水库采样随机抽取 N 条；未启用则取文件前 N 条 |
| `--port` | 7860 | HTTP 服务端口 |
| `--host` | 127.0.0.1 | 监听地址。WSL 下若需从 Windows 访问，改为 `0.0.0.0` |

> **注意（大文件）**：程序会完整扫描 JSONL 以统计 schema 信息，SAgoge 2.7M 行约需 1~2 分钟，之后才启动服务。如只想快速预览，配合 `--sample-n 100` 仍需等待扫描完毕。

---

## 常用示例

```bash
# 查看 SAgoge img2svg 前 200 条
python visualization/viewer.py \
  --jsonl /home/yanghaitao/Projects/Data/processed/SAgoge/stage1/icon/generation/img2svg/data_000000.jsonl \
  --image-root /home/yanghaitao/Projects/Data/raw/SAgoge \
  --sample-n 200

# 随机采样 500 条
python visualization/viewer.py \
  --jsonl /home/yanghaitao/Projects/Data/processed/SAgoge/stage1/icon/generation/img2svg/data_000000.jsonl \
  --image-root /home/yanghaitao/Projects/Data/raw/SAgoge \
  --sample-n 500 --random-sample

# text2svg（无图），全量加载（小文件适用）
python visualization/viewer.py \
  --jsonl /home/yanghaitao/Projects/Data/processed/SAgoge/stage1/icon/generation/text2svg/data_000000.jsonl \
  --sample-n -1
```

---

## 界面说明

### 左侧边栏

- 显示已加载的样本列表，每项显示 `index`（0-based）和原始行号 `line`。
- 搜索框支持按 index 或行号过滤。

### 顶部信息栏

显示当前文件的 schema 扫描摘要：

- 文件路径、总行数、已加载数（随机/前N）
- `roles`：出现的角色（user / assistant）
- `types`：content item 类型（text / image / …）
- 图像格式（image/jpeg、image/png）

### 样本面板（三个 Tab）

#### 概览

按 User / Assistant 顺序展示对话内容：

- **User 卡片**：先显示图像（含路径/尺寸说明），再显示文本
- **Assistant 卡片**：文本内容，代码块自动语法高亮

#### 原始 JSON

完整的 canonical schema JSON，带语法高亮，方便核查格式细节。

#### 渲染器

上半部分与"概览"完全相同（User + Assistant 卡片）。
分隔线下方为实时渲染区：

- **SVG**：从 assistant 输出中提取 ` ```svg ... ``` ` 代码块（或裸 `<svg>` 标签），在 iframe 中渲染；同时展示语法高亮的 SVG 源码。
- **HTML**：提取 ` ```html ... ``` ` 代码块，在 iframe 中渲染。
- 无可识别内容时显示提示。

### 键盘导航

| 按键 | 操作 |
|------|------|
| `←` / `↑` | 上一条样本 |
| `→` / `↓` | 下一条样本 |

图片加载失败时显示灰色占位符和路径提示，不影响其他内容展示。
