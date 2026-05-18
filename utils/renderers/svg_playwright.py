"""
utils.renderers.svg_playwright — 基于 Playwright/Chromium 的 SVG → PNG 渲染器

与 CairoSVG 渲染器相比，Playwright 使用真实浏览器引擎，对以下特性有完整支持：
  - feDropShadow 等 SVG 2.0 滤镜
  - width="100%" 视口解析
  - dominant-baseline 文字对齐
  - 复杂 CSS <style> 块
  - clip-path / <use href> 等高级特性

局限：
  - 依然受限于系统已安装字体（与 CairoSVG 相同）
  - 启动开销比 CairoSVG 大（建议在批量任务中复用 browser 实例）
  - 需要安装 playwright 及 Chromium：
      pip install playwright
      playwright install chromium
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Union

from PIL import Image

from .base import RenderResult


def render_svg_playwright(
    svg_code: str,
    output_path: Union[str, Path],
    width: int = 0,
    height: int = 0,
    background_color: tuple[int, int, int] = (255, 255, 255),
    timeout: int = 60,
) -> RenderResult:
    """用 Playwright（无头 Chromium）将 SVG 渲染为 PNG。

    渲染方式：将 SVG 嵌入最小化 HTML 页面，截图指定尺寸的元素区域，
    合成到纯色背景后保存为 RGB PNG。

    Args:
        svg_code         : SVG 源代码字符串
        output_path      : PNG 输出路径
        width            : 渲染目标宽度（像素）；0 使用 SVG 自身尺寸
        height           : 渲染目标高度（像素）；0 使用 SVG 自身尺寸
        background_color : RGB 背景颜色，默认纯白 (255, 255, 255)
        timeout          : 页面加载超时毫秒数（Playwright 单位为 ms，内部转换）

    Returns:
        RenderResult — 包含 success、实际 width/height、error 信息
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        return RenderResult(success=False, error=f"playwright not installed: {e}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 确定渲染尺寸：优先使用调用方传入的值，否则从 SVG 自身解析
    render_w, render_h = width, height
    if render_w <= 0 or render_h <= 0:
        from .svg import get_svg_dimensions
        parsed_w, parsed_h = get_svg_dimensions(svg_code)
        if render_w <= 0:
            render_w = parsed_w if parsed_w > 0 else 800
        if render_h <= 0:
            render_h = parsed_h if parsed_h > 0 else 600

    r, g, b = background_color
    bg_css = f"rgb({r},{g},{b})"

    # 构造最小化 HTML：固定 viewport = 渲染尺寸，SVG 充满容器
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  html, body {{
    width: {render_w}px;
    height: {render_h}px;
    background: {bg_css};
    overflow: hidden;
  }}
  #svg-container {{
    width: {render_w}px;
    height: {render_h}px;
    display: flex;
    align-items: stretch;
  }}
  #svg-container svg {{
    width: {render_w}px;
    height: {render_h}px;
  }}
</style>
</head>
<body>
<div id="svg-container">{svg_code}</div>
</body>
</html>"""

    try:
        timeout_ms = max(timeout * 1000, 5000) if timeout > 0 else 30000
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                page = browser.new_page(
                    viewport={"width": render_w, "height": render_h}
                )
                page.set_content(html, timeout=timeout_ms, wait_until="networkidle")
                png_bytes = page.screenshot(
                    clip={"x": 0, "y": 0, "width": render_w, "height": render_h},
                    type="png",
                )
            finally:
                browser.close()
    except Exception as e:
        return RenderResult(success=False, error=f"playwright render error: {e}")

    # 合成到纯色背景（处理截图中可能存在的透明通道）
    try:
        with Image.open(io.BytesIO(png_bytes)) as shot:
            bg = Image.new("RGB", shot.size, background_color)
            if shot.mode == "RGBA":
                bg.paste(shot, mask=shot.split()[3])
            else:
                bg.paste(shot.convert("RGB"))
            bg.save(str(output_path), format="PNG")
            actual_w, actual_h = bg.size
            out_buf = io.BytesIO()
            bg.save(out_buf, format="PNG")
            final_bytes = out_buf.getvalue()
    except Exception as e:
        return RenderResult(success=False, error=f"image save error: {e}")

    return RenderResult(success=True, width=actual_w, height=actual_h, png_bytes=final_bytes)
