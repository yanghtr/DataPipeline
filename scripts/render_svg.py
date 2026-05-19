#!/usr/bin/env python3
"""
批量将 img2svg JSONL 中的 SVG 渲染为 PNG。

用法示例：
  python scripts/render_svg.py \\
      --jsonl  C:/data/xxx/data_000000.jsonl \\
      --outdir C:/data/xxx/images_rendered \\
      --backend playwright   # 或 cairosvg（默认）
      --timeout 30
      --workers 4            # 并行线程数，默认 4；1 为串行
"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def _render_one(idx: int, line: str, outdir: str, backend: str, timeout: int, total: int) -> str:
    """处理单条 JSONL 记录，返回状态 'ok' / 'fail' / 'skip'。"""
    try:
        item = json.loads(line)
    except json.JSONDecodeError as e:
        logger.warning(f"[{idx:>6}] JSON 解析失败: {e}")
        return "fail"

    svg_code = _extract_svg(item)
    if svg_code is None:
        logger.warning(f"[{idx:>6}] 未找到 SVG，跳过")
        return "skip"

    try:
        rel_path: str = item["data"][0]["content"][0]["image"]["relative_path"]
    except (KeyError, IndexError, TypeError):
        rel_path = f"{idx:09d}.png"

    w, h = get_svg_dimensions(svg_code)
    out_path = os.path.join(outdir, rel_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    result = render_svg(
        svg_code,
        out_path,
        width=w,
        height=h,
        timeout=timeout,
        backend=backend,
    )

    if result.success:
        logger.info(f"[{idx:>6}/{total}] OK  {result.width}x{result.height}  {out_path}")
        return "ok"
    else:
        logger.error(f"[{idx:>6}/{total}] FAIL  {result.error}")
        return "fail"


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
    parser.add_argument("--workers", type=int, default=4,  help="并行线程数（默认 4；1 为串行）")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.jsonl, encoding="utf-8") as f:
        lines = f.readlines()

    total = len(lines)
    ok = fail = skip = 0

    if args.workers == 1:
        for idx, line in enumerate(lines):
            status = _render_one(idx, line, args.outdir, args.backend, args.timeout, total)
            if status == "ok":    ok += 1
            elif status == "skip": skip += 1
            else:                  fail += 1
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_render_one, idx, line, args.outdir, args.backend, args.timeout, total): idx
                for idx, line in enumerate(lines)
            }
            for future in as_completed(futures):
                status = future.result()
                if status == "ok":    ok += 1
                elif status == "skip": skip += 1
                else:                  fail += 1

    logger.info(f"完成: {ok} 成功 / {skip} 无SVG / {fail} 失败  (共 {total} 条)")


if __name__ == "__main__":
    main()
