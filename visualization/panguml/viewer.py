#!/usr/bin/env python3
"""
多模态数据可视化工具

可视化遵循 canonical schema 格式的 JSONL 数据文件。

用法:
    python visualization/viewer.py --jsonl <path> [--image-root <dir>] \\
        [--sample-n 500] [--random-sample] [--port 7860]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional

from flask import Flask, abort, jsonify, request, send_file
from loguru import logger

app = Flask(__name__)

_samples: list[dict] = []
_meta: dict = {}
_image_root: Optional[Path] = None


# ─── Schema scanning ──────────────────────────────────────────────────────────

def _update_schema(schema: dict, obj: dict) -> None:
    mp = obj.get("meta_prompt")
    if isinstance(mp, list):
        schema["meta_prompts"].add(tuple(mp))
    for turn in obj.get("data", []):
        role = turn.get("role", "")
        if role:
            schema["roles"].add(role)
        for item in turn.get("content", []):
            t = item.get("type", "")
            if t:
                schema["content_types"].add(t)
            if t == "image":
                fmt = item.get("image", {}).get("format", "")
                if fmt:
                    schema["image_formats"].add(fmt)
                schema["has_image"] = True
            elif t == "video":
                schema["has_video"] = True


def scan_and_load(
    path: Path,
    sample_n: int,
    random_sample: bool,
) -> tuple[list[dict], dict]:
    """
    单遍扫描完整 JSONL：提取 schema 信息，同时进行采样。

    采样策略：
    - random_sample=False: 取前 sample_n 条
    - random_sample=True:  水库采样（reservoir sampling），内存高效
    - sample_n=-1:         全部加载
    """
    schema: dict = {
        "roles": set(),
        "content_types": set(),
        "image_formats": set(),
        "meta_prompts": set(),
        "has_image": False,
        "has_video": False,
    }
    reservoir: list[tuple[int, dict]] = []
    first_n: list[tuple[int, dict]] = []
    total = 0
    err_count = 0

    logger.info(f"扫描文件: {path}")
    with path.open("r", encoding="utf-8") as f:
        for raw_lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                err_count += 1
                continue

            total += 1
            _update_schema(schema, obj)

            take_all = sample_n < 0
            if take_all or not random_sample:
                if take_all or total <= sample_n:
                    first_n.append((raw_lineno, obj))
            else:
                # Reservoir sampling (Algorithm R)
                if len(reservoir) < sample_n:
                    reservoir.append((raw_lineno, obj))
                else:
                    j = random.randint(0, total - 1)
                    if j < sample_n:
                        reservoir[j] = (raw_lineno, obj)

            if total % 200_000 == 0:
                logger.info(f"  已扫描 {total:,} 行 ...")

    selected = reservoir if (random_sample and sample_n > 0) else first_n
    selected.sort(key=lambda x: x[0])

    samples = [
        {"index": i, "line": ln, "data": obj}
        for i, (ln, obj) in enumerate(selected)
    ]

    meta = {
        "jsonl_path": str(path),
        "total_in_file": total,
        "total_loaded": len(samples),
        "parse_errors": err_count,
        "random_sample": random_sample,
        "schema": {
            "roles": sorted(schema["roles"]),
            "content_types": sorted(schema["content_types"]),
            "image_formats": sorted(schema["image_formats"]),
            "has_image": schema["has_image"],
            "has_video": schema["has_video"],
            "meta_prompts": [list(mp) for mp in sorted(schema["meta_prompts"])],
        },
    }
    logger.info(
        f"完成: 总 {total:,} 行，加载 {len(samples)} 条"
        f"{'（随机采样）' if random_sample else '（前 N 条）'}"
        + (f"，解析错误 {err_count} 条" if err_count else "")
    )
    return samples, meta


# ─── Flask routes ─────────────────────────────────────────────────────────────

_HTML_FILE = Path(__file__).parent / "viewer.html"


@app.route("/")
def index():
    return _HTML_FILE.read_text(encoding="utf-8"), 200, {
        "Content-Type": "text/html; charset=utf-8"
    }


@app.route("/api/info")
def api_info():
    return jsonify(_meta)


@app.route("/api/samples")
def api_samples():
    """轻量列表，供侧边栏渲染。"""
    return jsonify([{"index": s["index"], "line": s["line"]} for s in _samples])


@app.route("/api/sample/<int:idx>")
def api_sample(idx: int):
    if idx < 0 or idx >= len(_samples):
        abort(404)
    return jsonify(_samples[idx])


@app.route("/api/image")
def api_image():
    """按 relative_path 返回图片文件，防止路径穿越。"""
    rel = request.args.get("path", "").lstrip("/")
    if not rel or not _image_root:
        abort(404)

    try:
        full = (_image_root / rel).resolve()
        root_resolved = _image_root.resolve()
        if not str(full).startswith(str(root_resolved) + "/") and full != root_resolved:
            abort(403)
    except Exception:
        abort(400)

    if not full.exists():
        abort(404)

    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".svg": "image/svg+xml",
        ".bmp": "image/bmp",
    }.get(full.suffix.lower(), "application/octet-stream")

    return send_file(str(full), mimetype=mime)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="多模态 canonical schema 数据可视化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 查看前 200 条
  python visualization/viewer.py \\
    --jsonl /data/processed/SAgoge/stage1/icon/generation/img2svg/data_000000.jsonl \\
    --image-root /data/raw/SAgoge \\
    --sample-n 200

  # 随机采样 500 条
  python visualization/viewer.py \\
    --jsonl /data/processed/SAgoge/stage1/icon/generation/img2svg/data_000000.jsonl \\
    --image-root /data/raw/SAgoge \\
    --sample-n 500 --random-sample

  # 全部加载（小文件）
  python visualization/viewer.py --jsonl small.jsonl --sample-n -1
""",
    )
    parser.add_argument("--jsonl", type=Path, required=True, help="JSONL 文件路径")
    parser.add_argument(
        "--image-root", type=Path, default=None,
        help="图片根目录（image.relative_path 基准目录）",
    )
    parser.add_argument(
        "--sample-n", type=int, default=500,
        help="加载样本数，-1 表示全部（默认 500）",
    )
    parser.add_argument(
        "--random-sample", action="store_true",
        help="随机采样；未设置时取前 N 条",
    )
    parser.add_argument("--port", type=int, default=7860, help="端口号（默认 7860）")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="监听地址")
    args = parser.parse_args()

    if not args.jsonl.exists():
        logger.error(f"文件不存在: {args.jsonl}")
        sys.exit(1)

    global _samples, _meta, _image_root
    _image_root = args.image_root

    _samples, _meta = scan_and_load(args.jsonl, args.sample_n, args.random_sample)

    url = f"http://{args.host}:{args.port}"
    logger.info(f"启动服务 → {url}")
    print(f"\n  可视化工具已就绪: {url}\n")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
