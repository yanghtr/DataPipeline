#!/usr/bin/env python3
"""
批量将 img2svg JSONL 中的 SVG 渲染为 PNG。

用法示例：
  python scripts/render_svg.py \\
      --jsonl  C:/data/xxx/data_000000.jsonl \\
      --outdir C:/data/xxx/images_rendered \\
      --backend playwright   # 或 cairosvg（默认）
      --timeout 30
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from loguru import logger
from utils.renderers.svg import get_svg_dimensions, render_svg

_SVG_RE = re.compile(r"<svg\b.*?</svg>", re.DOTALL | re.IGNORECASE)


def _extract_svg(item: dict) -> str | None:
    """从 img2svg JSONL 记录中提取 SVG 字符串。"""
    try:
        text: str = item["data"][1]["content"][0]["text"]["string"]
    except (KeyError, IndexError, TypeError):
        return None
    m = _SVG_RE.search(text)
    return m.group(0) if m else None


def main() -> None:
    parser = argparse.ArgumentParser(description="批量渲染 img2svg JSONL → PNG")
    parser.add_argument("--jsonl",   required=True, help="输入 JSONL 文件路径")
    parser.add_argument("--outdir",  required=True, help="PNG 输出目录")
    parser.add_argument(
        "--backend",
        choices=["cairosvg", "playwright"],
        default="cairosvg",
        help="渲染后端（默认 cairosvg）",
    )
    parser.add_argument("--timeout", type=int, default=30, help="单张渲染超时秒数（默认 30）")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.jsonl, encoding="utf-8") as f:
        lines = f.readlines()

    total = len(lines)
    ok = fail = skip = 0

    for idx, line in enumerate(lines):
        try:
            item = json.loads(line)
        except json.JSONDecodeError as e:
            logger.warning(f"[{idx:>6}] JSON 解析失败: {e}")
            fail += 1
            continue

        svg_code = _extract_svg(item)
        if svg_code is None:
            logger.warning(f"[{idx:>6}] 未找到 SVG，跳过")
            skip += 1
            continue

        w, h = get_svg_dimensions(svg_code)
        out_path = os.path.join(args.outdir, f"{idx:09d}.png")
        result = render_svg(
            svg_code,
            out_path,
            width=w,
            height=h,
            timeout=args.timeout,
            backend=args.backend,
        )

        if result.success:
            logger.info(f"[{idx:>6}/{total}] OK  {result.width}x{result.height}  {out_path}")
            ok += 1
        else:
            logger.error(f"[{idx:>6}/{total}] FAIL  {result.error}")
            fail += 1

    logger.info(f"完成: {ok} 成功 / {skip} 无SVG / {fail} 失败  (共 {total} 条)")


if __name__ == "__main__":
    main()
