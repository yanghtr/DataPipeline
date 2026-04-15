"""
utils.renderers — 可复用的沙箱渲染工具集

当前支持：
  - SVG → PNG  (utils.renderers.svg)

后续可扩展：
  - HTML → PNG  (utils.renderers.html)
  - Matplotlib Figure → PNG  (utils.renderers.matplotlib)
  - ...

公共接口：
  RenderResult    渲染结果数据类
  read_image_size 读取图片宽高
  check_all_white 检测图片是否全白
"""

from .base import RenderResult, check_all_white, read_image_size
from .svg import get_svg_dimensions, render_svg

__all__ = [
    "RenderResult",
    "read_image_size",
    "check_all_white",
    "get_svg_dimensions",
    "render_svg",
]
