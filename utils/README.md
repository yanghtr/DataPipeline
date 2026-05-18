# utils 工具说明

## renderers — SVG / HTML 渲染器

`utils/renderers/` 提供统一的 SVG → PNG 渲染接口，支持两种后端：

| 后端 | 模块 | 适用场景 |
|------|------|---------|
| CairoSVG | `svg.py` | 轻量、无需浏览器、速度快；对 SVG 1.1 支持完整 |
| Playwright | `svg_playwright.py` | 与浏览器渲染完全一致；支持 `feDropShadow`、`dominant-baseline`、复杂 CSS 等 CairoSVG 不支持的特性 |

两种后端共享同一 `RenderResult` 返回类型，通过 `render_svg()` 的 `backend` 参数切换。

### 安装依赖

```bash
pip install cairosvg Pillow numpy          # CairoSVG 后端
pip install playwright && playwright install chromium  # Playwright 后端
```

### 快速使用

```python
from utils.renderers import get_svg_dimensions, render_svg

svg_code = open("example.svg").read()
w, h = get_svg_dimensions(svg_code)

# 默认使用 CairoSVG
render_svg(svg_code, "output.png", width=w, height=h)

# 切换到 Playwright（与浏览器渲染一致）
render_svg(svg_code, "output.png", width=w, height=h, backend="playwright")
```

也可直接调用 Playwright 渲染函数：

```python
from utils.renderers import render_svg_playwright
render_svg_playwright(svg_code, "output.png", width=800, height=600)
```

### 已知渲染差异（CairoSVG vs 浏览器）

| 问题 | Playwright 是否解决 |
|------|-------------------|
| `feDropShadow` 阴影丢失 | ✅ |
| `width="100%"` 视口解析错误 | ✅ |
| `dominant-baseline` 文字对齐偏移 | ✅ |
| 复杂滤镜链渲染差异 | ✅ |
| 平台专属字体缺失（`-apple-system` 等） | ❌ 需单独在系统安装字体 |

使用 `scripts/detect_svg_fonts.py` 可扫描数据集中所需的全部字体。

---

## hf_downloader.py

从 HuggingFace 下载数据集、模型或 space，支持断点续传、多线程并行、高速下载和完整性校验。

### 安装依赖

```bash
pip install -r ../requirements.txt
```

### 快速使用

```bash
cd utils

# 下载数据集（默认类型）
python hf_downloader.py --repo-id allenai/c4

# 下载模型
python hf_downloader.py --repo-id meta-llama/Llama-2-7b --type model

# 指定线程数和输出目录
python hf_downloader.py --repo-id allenai/c4 --num-workers 8 --output-dir /data/raw

# 下载私有仓库
python hf_downloader.py --repo-id my-org/private-dataset --token hf_xxxxxxxx
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--repo-id` | 必填 | 仓库 ID，格式为 `owner/name` |
| `--type` | `dataset` | 仓库类型：`dataset` / `model` / `space` |
| `--output-dir` | `../Data/raw` | 输出根目录，文件保存至 `<output-dir>/<repo-name>/` |
| `--num-workers` | `16` | 并行下载线程数，网络带宽充足时可调高 |
| `--token` | 无 | HuggingFace Access Token，访问私有仓库时必填 |
| `--no-verify` | 关闭 | 加此参数跳过下载后的完整性校验 |

### 断点续传

中断后直接重新运行相同命令即可，程序会自动检测本地已有文件并跳过。

### 日志

运行日志自动写入 `logs/hf_downloader_<时间>.log`，单文件上限 100MB，保留 7 天。
