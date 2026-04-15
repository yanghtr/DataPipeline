"""
utils.renderers.svg — SVG → PNG 渲染器

功能：
  - 从 SVG 代码中解析宽高（优先 width/height 属性，回退到 viewBox）
  - 用 CairoSVG 光栅化 SVG，并将结果合成到白色背景上（解决透明背景问题）
  - 返回 RenderResult（含实际渲染后的像素宽高）

背景处理说明：
  SVG 默认背景为透明。CairoSVG 渲染后得到 RGBA PNG。
  本模块在保存前将其合成到纯白 RGB 图像上，确保最终 PNG 背景始终为白色，
  与后续全白检测逻辑（check_all_white）保持一致。
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Union

from PIL import Image

from .base import RenderResult

# ── SVG 属性解析正则 ──────────────────────────────────────────────────────────
# 匹配数字或带 px 单位的数字；百分比等相对值不匹配（回退到 viewBox）
_W_RE = re.compile(r'<svg\b[^>]*\bwidth=["\'](\d+(?:\.\d+)?)(?:px)?["\']', re.IGNORECASE)
_H_RE = re.compile(r'<svg\b[^>]*\bheight=["\'](\d+(?:\.\d+)?)(?:px)?["\']', re.IGNORECASE)
_VB_RE = re.compile(
    r'<svg\b[^>]*\bviewBox=["\'][\d.]+\s+[\d.]+\s+([\d.]+)\s+([\d.]+)["\']',
    re.IGNORECASE,
)


def get_svg_dimensions(svg_code: str) -> tuple[int, int]:
    """从 SVG 代码中解析渲染尺寸。

    优先级：width/height 属性 → viewBox 的 width/height → (0, 0)。
    百分比宽高（如 width="100%"）视为未知，回退到 viewBox。

    Returns:
        (width, height) 整数像素值；无法解析时返回 (0, 0)。
    """
    w_m = _W_RE.search(svg_code)
    h_m = _H_RE.search(svg_code)
    if w_m and h_m:
        return int(float(w_m.group(1))), int(float(h_m.group(1)))

    vb_m = _VB_RE.search(svg_code)
    if vb_m:
        return int(float(vb_m.group(1))), int(float(vb_m.group(2)))

    return 0, 0


def render_svg(
    svg_code: str,
    output_path: Union[str, Path],
    width: int = 0,
    height: int = 0,
    background_color: tuple[int, int, int] = (255, 255, 255),
) -> RenderResult:
    """将 SVG 代码渲染为 PNG 并保存到 output_path。

    渲染流程：
      1. CairoSVG 将 SVG 光栅化为 RGBA PNG（内存）
      2. 创建纯色背景图（默认白色）
      3. 将 RGBA 图按 alpha 通道合成到背景上
      4. 保存为 RGB PNG 文件

    Args:
        svg_code         : SVG 源代码字符串
        output_path      : PNG 输出路径
        width            : 渲染目标宽度（像素）；0 表示使用 SVG 自身尺寸
        height           : 渲染目标高度（像素）；0 表示使用 SVG 自身尺寸
        background_color : RGB 背景颜色，默认纯白 (255, 255, 255)

    Returns:
        RenderResult — 包含 success、实际 width/height、error 信息
    """
    try:
        import cairosvg  # 延迟导入，便于在 worker 子进程中使用
    except ImportError as e:
        return RenderResult(success=False, error=f"cairosvg not installed: {e}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. 用 CairoSVG 渲染为内存 PNG bytes（background_color 直接由 cairosvg 填充）
    try:
        r, g, b = background_color
        kwargs: dict = {"background_color": f"#{r:02x}{g:02x}{b:02x}"}
        if width > 0:
            kwargs["output_width"] = width
        if height > 0:
            kwargs["output_height"] = height
        png_bytes: bytes = cairosvg.svg2png(
            bytestring=svg_code.encode("utf-8"), **kwargs
        )
    except Exception as e:
        return RenderResult(success=False, error=f"cairosvg render error: {e}")

    # 2. 保存并读取实际像素尺寸
    try:
        with Image.open(io.BytesIO(png_bytes)) as result_img:
            result_img.save(str(output_path), format="PNG")
            actual_w, actual_h = result_img.size
    except Exception as e:
        return RenderResult(success=False, error=f"image save error: {e}")

    return RenderResult(success=True, width=actual_w, height=actual_h, png_bytes=png_bytes)
