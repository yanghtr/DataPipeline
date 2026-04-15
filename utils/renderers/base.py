"""
utils.renderers.base — 渲染器公共基础类型与图像工具函数

所有渲染器共享 RenderResult，以及通用的图像读取/校验工具。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


@dataclass
class RenderResult:
    """渲染操作的统一返回结果。

    Attributes:
        success : 渲染是否成功
        width   : 渲染后图片的实际宽度（像素），失败时为 0
        height  : 渲染后图片的实际高度（像素），失败时为 0
        error   : 失败时的错误描述，成功时为 None
    """

    success: bool
    width: int = 0
    height: int = 0
    error: Optional[str] = None


def read_image_size(image_path: str | Path) -> tuple[int, int]:
    """读取图片的 (width, height)，失败时返回 (0, 0)。"""
    try:
        return Image.open(image_path).size  # PIL returns (width, height)
    except Exception:
        return 0, 0


def check_all_white(image_path: str | Path, threshold: int = 250) -> bool:
    """检查图片是否为全白（所有 RGB 通道均 >= threshold）。

    Args:
        image_path : 图片路径
        threshold  : 白色判断阈值，默认 250（接近纯白）
    Returns:
        True 表示图片为全白或接近全白；False 表示含有非白像素
    """
    try:
        arr = np.array(Image.open(image_path).convert("RGB"))
        return bool(np.all(arr >= threshold))
    except Exception:
        return False
