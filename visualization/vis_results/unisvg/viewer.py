#!/usr/bin/env python3
"""
UniSVG 实验结果多列对比可视化工具。

每个实验结果目录包含以下文件：
  gt_{index}.svg, gt_{index}.png   ← 所有实验共享同一套 GT（取第一个目录）
  gen_{index}.svg, gen_{index}.png ← 各实验各自的生成结果

最终页面：最左列为 GT，后续列依次展示各实验目录的 gen 结果。
每个单元格从上到下展示：预渲染图片(.png)、SVG 源码、浏览器实时渲染。

用法:
    python visualization/unisvg/viewer.py \\
        --dirs /path/to/exp1 /path/to/exp2 [/path/to/exp3 ...] \\
        [--port 7862] [--host 0.0.0.0]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from flask import Flask, jsonify, request, abort
from loguru import logger

app = Flask(__name__)

# ── 全局状态 ──────────────────────────────────────────────────────────────────

_dirs: list[Path] = []    # 实验结果目录（已 resolve 为绝对路径）
_indices: list[int] = []  # 从 dirs[0] 扫描到的 GT index 列表（排序后）


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _scan_gt_indices(d: Path) -> list[int]:
    """扫描目录中所有 gt_{n}.svg 文件，返回排序后的 index 列表。"""
    indices: list[int] = []
    for p in d.iterdir():
        m = re.match(r"^gt_(\d+)\.svg$", p.name)
        if m:
            indices.append(int(m.group(1)))
    return sorted(indices)


def _make_label(d: Path) -> str:
    """取目录末尾两级作为短标签（路径过短时直接取目录名）。"""
    parts = d.parts
    return "/".join(parts[-2:]) if len(parts) >= 2 else d.name


def _build_row(index: int) -> dict:
    """读取指定 index 的 GT 和各实验 gen 数据，构建行字典。"""
    gt_dir = _dirs[0]
    gt_svg_path = gt_dir / f"gt_{index}.svg"
    gt_png_path = gt_dir / f"gt_{index}.png"

    try:
        gt_svg_code: str = gt_svg_path.read_text(encoding="utf-8")
    except OSError:
        gt_svg_code = ""

    gt: dict = {
        "svg_code": gt_svg_code,
        "has_png": gt_png_path.exists(),
    }

    gens: list[dict] = []
    for d in _dirs:
        gen_svg_path = d / f"gen_{index}.svg"
        gen_png_path = d / f"gen_{index}.png"
        try:
            gen_svg_code: str = gen_svg_path.read_text(encoding="utf-8")
        except OSError:
            gen_svg_code = ""
        gens.append({
            "svg_code": gen_svg_code,
            "has_png": gen_png_path.exists(),
        })

    return {"index": index, "gt": gt, "gens": gens}


# ── Flask routes ──────────────────────────────────────────────────────────────

_HTML_FILE = Path(__file__).parent / "viewer.html"


@app.route("/")
def route_index():
    return _HTML_FILE.read_text(encoding="utf-8"), 200, {
        "Content-Type": "text/html; charset=utf-8"
    }


@app.route("/api/info")
def api_info():
    return jsonify({
        "total_rows": len(_indices),
        "columns": [
            {"label": _make_label(d), "dir": str(d)}
            for d in _dirs
        ],
    })


@app.route("/api/rows")
def api_rows():
    page = max(0, int(request.args.get("page", 0)))
    size = max(1, min(200, int(request.args.get("size", 10))))
    start = page * size
    page_indices = _indices[start: start + size]
    rows = [_build_row(i) for i in page_indices]
    return jsonify({
        "page": page,
        "size": size,
        "total_rows": len(_indices),
        "rows": rows,
    })


@app.route("/api/image")
def api_image():
    """提供图片文件。

    URL 参数:
      index (int)  样本索引
      role  (str)  "gt" 或 "gen"
      exp   (int)  实验目录下标，role=gen 时有效（默认 0）
    """
    index = request.args.get("index", type=int)
    role = request.args.get("role", "")
    exp = request.args.get("exp", 0, type=int)

    if index is None or role not in ("gt", "gen"):
        abort(400)

    if role == "gt":
        p = _dirs[0] / f"gt_{index}.png"
    else:
        if exp < 0 or exp >= len(_dirs):
            abort(400)
        p = _dirs[exp] / f"gen_{index}.png"

    p = p.resolve()
    # 安全检查：只允许访问已注册目录内的文件
    if not any(str(p).startswith(str(d)) for d in _dirs):
        abort(403)

    if not p.exists():
        abort(404)

    suffix = p.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return p.read_bytes(), 200, {
        "Content-Type": mime,
        "Cache-Control": "max-age=3600",
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="UniSVG 实验结果多列对比可视化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  # 对比两个实验目录
  python visualization/unisvg/viewer.py \\
    --dirs /path/to/exp1 /path/to/exp2 \\
    --port 7862

  # 对比三个实验目录，仅本机访问
  python visualization/unisvg/viewer.py \\
    --dirs /path/to/exp1 /path/to/exp2 /path/to/exp3 \\
    --host 127.0.0.1 --port 7862
""",
    )
    parser.add_argument(
        "--dirs", type=str, nargs="+", required=True,
        help="实验结果目录路径（可指定多个，空格分隔）",
    )
    parser.add_argument(
        "--port", type=int, default=7862,
        help="HTTP 端口（默认 7862）",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="监听地址（默认 0.0.0.0；仅本机访问可设为 127.0.0.1）",
    )
    args = parser.parse_args()

    for raw in args.dirs:
        d = Path(raw).resolve()
        if not d.is_dir():
            logger.error(f"目录不存在: {raw}")
            sys.exit(1)
        _dirs.append(d)
        logger.info(f"注册目录: {d}")

    global _indices
    _indices = _scan_gt_indices(_dirs[0])

    if not _indices:
        logger.warning(f"第一个目录中未找到 gt_*.svg 文件: {_dirs[0]}")
    else:
        logger.info(
            f"找到 {len(_indices)} 个样本，index 范围: "
            f"{_indices[0]} ~ {_indices[-1]}"
        )

    url = f"http://{args.host}:{args.port}"
    logger.info(f"启动服务 → {url}")
    print(f"\n  UniSVG 对比可视化已就绪: {url}\n")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
