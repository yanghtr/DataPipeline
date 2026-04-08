#!/usr/bin/env python3
"""
SVG 蒸馏结果多列对比可视化工具。

支持动态添加/删除 JSONL 列，按 id 字段对齐数据行，
渲染 SVG 并对比不同模型的蒸馏结果。

支持两种 JSONL 格式：
  - Distillation output：id + instruction + response（含 fence）
  - Query seed：        instruction + _meta.id + _meta.gt_svg（含 fence）

用法:
    python visualization/distillation/viewer.py \\
        --jsonl /path/to/col1.jsonl /path/to/col2.jsonl \\
        [--sample-n 500] [--random-sample] [--port 7861]

    # 也可不带 --jsonl 启动，在 UI 里动态添加列
    python visualization/distillation/viewer.py --port 7861
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
from pathlib import Path

from flask import Flask, jsonify, request
from loguru import logger

app = Flask(__name__)

# ── SVG 提取工具 ───────────────────────────────────────────────────────────────

_SVG_FENCE_RE = re.compile(r"```svg\s*([\s\S]*?)```", re.IGNORECASE)
_SCRIPT_RE    = re.compile(r"<script[\s\S]*?</script>", re.IGNORECASE)
_RAW_SVG_RE   = re.compile(r"<svg[\s\S]*?</svg>", re.IGNORECASE)


def _extract_svg(response: str | None) -> str | None:
    """从 response 提取 SVG 字符串，剥除 markdown fence 和 <script> 标签。"""
    if not response:
        return None
    m = _SVG_FENCE_RE.search(response)
    if m:
        svg = m.group(1).strip()
    else:
        m2 = _RAW_SVG_RE.search(response)
        svg = m2.group(0) if m2 else None
    if not svg:
        return None
    svg = _SCRIPT_RE.sub("", svg).strip()
    return svg or None


def _extract_instruction(record: dict) -> str:
    """提取 instruction。两种格式均在顶层 instruction 字段。"""
    return record.get("instruction") or ""


def _extract_id(record: dict) -> str:
    """提取唯一 ID。
    - Distillation output: 顶层 id 字段
    - Query seed:          _meta.id 字段
    """
    return record.get("id") or record.get("_meta", {}).get("id") or ""


def _get_response_text(record: dict) -> str | None:
    """提取 SVG response 文本（含 fence，由 _extract_svg 负责剥除）。
    - Distillation output: record["response"]
    - Query seed:          record["_meta"]["gt_svg"]
    """
    if "response" in record:
        return record["response"]
    return record.get("_meta", {}).get("gt_svg") or None


# ── Column 数据结构 ────────────────────────────────────────────────────────────

class Column:
    def __init__(
        self,
        path: str,
        label: str,
        id_order: list[str],
        cells: dict[str, dict],
        instructions: dict[str, str],
        total_in_file: int,
        loaded: int,
        sample_n: int,
        random_sample: bool,
    ) -> None:
        self.path = path
        self.label = label
        self.id_order = id_order          # 保留加载顺序
        self.cells = cells                # id → cell dict
        self.instructions = instructions  # id → instruction text
        self.total_in_file = total_in_file
        self.loaded = loaded
        self.sample_n = sample_n
        self.random_sample = random_sample


def _make_label(path: str) -> str:
    parts = Path(path).parts
    return "/".join(parts[-2:]) if len(parts) >= 2 else Path(path).name


def _parse_column_records(
    records: list[tuple[int, dict]],
    path: str,
    total: int,
    sample_n: int,
    random_sample: bool,
) -> Column:
    """将已选中的记录转换成列数据。"""
    records.sort(key=lambda x: x[0])

    id_order: list[str] = []
    cells: dict[str, dict] = {}
    instructions: dict[str, str] = {}

    for _, obj in records:
        rec_id = _extract_id(obj)
        if not rec_id or rec_id in cells:
            continue
        id_order.append(rec_id)
        resp_text = _get_response_text(obj)
        svg = _extract_svg(resp_text)
        cells[rec_id] = {
            "svg": svg,
            "raw_response": resp_text,
            "status": obj.get("status", ""),
            "model": obj.get("model", ""),
            "has_svg": svg is not None,
        }
        instructions[rec_id] = _extract_instruction(obj)

    logger.info(f"[load] 完成: {len(id_order):,} 条（文件总行 {total:,}）")
    return Column(
        path=path,
        label=_make_label(path),
        id_order=id_order,
        cells=cells,
        instructions=instructions,
        total_in_file=total,
        loaded=len(id_order),
        sample_n=sample_n,
        random_sample=random_sample,
    )


def _load_column(
    path: str,
    sample_n: int,
    random_sample: bool,
    reference_ids: list[str] | None = None,
) -> Column:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    reference_index = None
    reference_set = None
    if reference_ids is not None:
        reference_index = {rid: idx for idx, rid in enumerate(reference_ids)}
        reference_set = set(reference_ids)

    reservoir: list[tuple[int, dict]] = []
    first_n: list[tuple[int, dict]] = []
    matched_records: list[tuple[int, dict]] = []
    total = 0

    logger.info(f"[load] 加载: {path}")
    with p.open(encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            total += 1
            if reference_set is not None:
                rec_id = _extract_id(obj)
                if rec_id in reference_set:
                    matched_records.append((reference_index[rec_id], obj))
                continue
            take_all = sample_n < 0
            if take_all or not random_sample:
                if take_all or total <= sample_n:
                    first_n.append((lineno, obj))
            else:
                if len(reservoir) < sample_n:
                    reservoir.append((lineno, obj))
                else:
                    j = random.randint(0, total - 1)
                    if j < sample_n:
                        reservoir[j] = (lineno, obj)

    if reference_set is not None:
        logger.info(
            f"[load] 参考首列 ID 匹配: {len(matched_records):,} / {len(reference_ids):,} 条"
        )
        return _parse_column_records(
            matched_records,
            path=path,
            total=total,
            sample_n=sample_n,
            random_sample=random_sample,
        )

    selected = reservoir if (random_sample and sample_n > 0) else first_n
    return _parse_column_records(
        selected,
        path=path,
        total=total,
        sample_n=sample_n,
        random_sample=random_sample,
    )


# ── 全局状态 ──────────────────────────────────────────────────────────────────

_columns: list[Column] = []
_row_ids: list[str] = []
_state_lock = threading.Lock()


def _recompute_row_ids() -> None:
    """以第一列为基准，行列表 = 第一列的 id 列表（其他列按 id 匹配，缺失则显示占位符）。
    调用前必须持有 _state_lock。
    """
    global _row_ids
    if not _columns:
        _row_ids = []
        return
    # 始终以第一列的 id 顺序作为基准
    _row_ids = list(_columns[0].id_order)
    logger.info(f"[rows] 基准行数: {len(_row_ids):,}（以第一列为参考）")


def _build_info() -> dict:
    return {
        "columns": [
            {
                "idx": i,
                "path": c.path,
                "label": c.label,
                "total_in_file": c.total_in_file,
                "loaded": c.loaded,
                "sample_n": c.sample_n,
                "random_sample": c.random_sample,
            }
            for i, c in enumerate(_columns)
        ],
        "row_count": len(_row_ids),
    }


# ── Flask routes ──────────────────────────────────────────────────────────────

_HTML_FILE = Path(__file__).parent / "viewer.html"


@app.route("/")
def index() -> tuple:
    return _HTML_FILE.read_text(encoding="utf-8"), 200, {
        "Content-Type": "text/html; charset=utf-8"
    }


@app.route("/api/info")
def api_info():
    with _state_lock:
        return jsonify(_build_info())


@app.route("/api/rows")
def api_rows():
    page = max(0, int(request.args.get("page", 0)))
    size = max(1, min(200, int(request.args.get("size", 20))))
    with _state_lock:
        start = page * size
        page_ids = _row_ids[start: start + size]
        rows = []
        for rid in page_ids:
            instruction = ""
            for col in _columns:
                if rid in col.instructions:
                    instruction = col.instructions[rid]
                    break
            cells = [
                col.cells.get(rid, {
                    "svg": None,
                    "raw_response": None,
                    "status": "missing",
                    "model": "",
                    "has_svg": False,
                })
                for col in _columns
            ]
            rows.append({"id": rid, "instruction": instruction, "cells": cells})
        return jsonify({
            "page": page,
            "size": size,
            "total_rows": len(_row_ids),
            "rows": rows,
        })


@app.route("/api/columns", methods=["POST"])
def api_add_column():
    body = request.get_json(force=True) or {}
    path = (body.get("path") or "").strip()
    if not path:
        return jsonify({"error": "path is required"}), 400
    sample_n = int(body.get("sample_n", 500))
    random_sample = bool(body.get("random", False))
    try:
        with _state_lock:
            reference_ids = list(_columns[0].id_order) if _columns else None
        col = _load_column(path, sample_n, random_sample, reference_ids=reference_ids)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
    with _state_lock:
        _columns.append(col)
        _recompute_row_ids()
        return jsonify(_build_info())


@app.route("/api/columns/<int:idx>", methods=["DELETE"])
def api_remove_column(idx: int):
    with _state_lock:
        if idx < 0 or idx >= len(_columns):
            return jsonify({"error": "index out of range"}), 400
        _columns.pop(idx)
        _recompute_row_ids()
        return jsonify(_build_info())


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SVG 蒸馏结果多列对比可视化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  # 对比两列，每列取前 200 条
  python visualization/distillation/viewer.py \\
    --jsonl /path/to/model_a.jsonl /path/to/model_b.jsonl \\
    --sample-n 200

  # 随机采样 500 条
  python visualization/distillation/viewer.py \\
    --jsonl /path/to/model_a.jsonl --sample-n 500 --random-sample

  # 不带初始文件启动，在 UI 中动态添加列
  python visualization/distillation/viewer.py --port 7861
""",
    )
    parser.add_argument(
        "--jsonl", type=str, nargs="*", default=[],
        help="初始 JSONL 文件路径（可多个，空格分隔）",
    )
    parser.add_argument(
        "--sample-n", type=int, default=500,
        help="每列加载条数，-1 表示全部（默认 500）",
    )
    parser.add_argument(
        "--random-sample", action="store_true",
        help="随机采样；未设置则取文件前 N 条",
    )
    parser.add_argument("--port", type=int, default=7861, help="HTTP 端口（默认 7861）")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="监听地址")
    args = parser.parse_args()

    for path in (args.jsonl or []):
        try:
            reference_ids = list(_columns[0].id_order) if _columns else None
            col = _load_column(
                path,
                args.sample_n,
                args.random_sample,
                reference_ids=reference_ids,
            )
            _columns.append(col)
        except Exception as exc:
            logger.error(f"加载失败 {path}: {exc}")
            sys.exit(1)

    _recompute_row_ids()

    url = f"http://{args.host}:{args.port}"
    logger.info(f"启动服务 → {url}")
    print(f"\n  SVG 蒸馏对比工具已就绪: {url}\n")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
