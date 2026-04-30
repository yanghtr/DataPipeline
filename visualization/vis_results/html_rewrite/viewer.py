#!/usr/bin/env python3
"""
HTML 改写结果多列对比可视化工具。

列结构（默认）：
  - 第一个 JSONL：同时生成 preprocessed_html 列 + output_html 列
  - 后续 JSONL：只生成 output_html 列（预处理结果相同，避免重复）
  - Live URL 列：尝试在 iframe 中嵌入原始网站，
    多数现代站点会因 X-Frame-Options 拒绝嵌入，提供「新标签打开」兜底

渲染机制（非 sandbox iframe）：
  Flask 的 /render/<col>/<row> 端点将 HTML 以完整页面形式返回，
  并在 <head> 中注入 <base href="原始域名">，使 CSS/图片/JS 等相对路径
  资源能从原始服务器正常加载。<iframe src="/render/..."> 完整页面加载，
  不加 sandbox 限制，等效于在新标签打开该 HTML 文件。

用法:
    python visualization/vis_results/html_rewrite/viewer.py \\
        --jsonl /path/to/output.jsonl \\
        [--sample-n 20] [--random-sample] [--port 7862]

    # 对比两个模型配置的改写结果
    python visualization/vis_results/html_rewrite/viewer.py \\
        --jsonl /path/to/output_a.jsonl /path/to/output_b.jsonl \\
        --sample-n 20
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
from pathlib import Path
from urllib.parse import urlparse

from flask import Flask, jsonify, make_response, request
from loguru import logger

app = Flask(__name__)

# ── 常量 ─────────────────────────────────────────────────────────────────────

FIELD_PREPROCESSED = "preprocessed_html"
FIELD_OUTPUT = "output_html"

# ── HTML 工具 ─────────────────────────────────────────────────────────────────

_HEAD_RE = re.compile(r"(<head[^>]*>)", re.IGNORECASE)


def _origin(url: str) -> str:
    """提取 URL 的 origin（scheme://host/），用作 <base href>。"""
    try:
        p = urlparse(url)
        return f"{p.scheme}://{p.netloc}/"
    except Exception:
        return url


def inject_base_url(html: str, url: str) -> str:
    """
    若 HTML 中没有 <base> 标签，在 <head> 紧后注入 <base href="origin">。
    这让 HTML 中的相对路径资源（CSS、图片、JS）能从原始域名正确加载。
    """
    if not url or "<base" in html.lower():
        return html
    origin = _origin(url)
    base_tag = f'<base href="{origin}">'
    m = _HEAD_RE.search(html)
    if m:
        return html[: m.end()] + "\n  " + base_tag + html[m.end():]
    # 没有 <head> 就插在最前
    return base_tag + "\n" + html


# ── 数据结构 ─────────────────────────────────────────────────────────────────

class Column:
    def __init__(
        self,
        path: str,
        field: str,
        label: str,
        id_order: list[str],
        meta: dict[str, dict],
        html_data: dict[str, str],
        total_in_file: int,
    ) -> None:
        self.path = path
        self.field = field          # FIELD_PREPROCESSED or FIELD_OUTPUT
        self.label = label
        self.id_order = id_order
        self.meta = meta            # id -> metadata dict
        self.html_data = html_data  # id -> html string
        self.total_in_file = total_in_file
        self.loaded = len(id_order)


# ── 数据加载 ─────────────────────────────────────────────────────────────────

def _get_id(record: dict) -> str:
    return (
        record.get("id")
        or record.get("_meta", {}).get("url")
        or record.get("url", "")
    )


def _extract_meta(record: dict) -> dict:
    m = record.get("_meta", {})
    return {
        "url": m.get("url") or record.get("id", ""),
        "final_url": m.get("final_url", ""),
        "page_type": m.get("page_type", []),
        "model": record.get("model", ""),
        "prompt_tokens": record.get("prompt_tokens"),
        "completion_tokens": record.get("completion_tokens"),
        "finish_reason": record.get("finish_reason", ""),
        "preprocess_stats": _compress_stats(record.get("preprocess_stats", {})),
    }


def _compress_stats(stats: dict) -> dict:
    """只保留 stats 里的摘要数字，避免 API 响应过大。"""
    if not stats:
        return {}
    return {
        "original_chars": stats.get("original_chars"),
        "cleaned_chars": stats.get("cleaned_chars"),
        "compression_ratio": stats.get("compression_ratio"),
        "media_total": stats.get("media", {}).get("total"),
        "media_with_size": stats.get("media", {}).get("with_size"),
        "script_truncated": stats.get("scripts", {}).get("inline_truncated"),
        "style_truncated": stats.get("styles", {}).get("inline_truncated"),
    }


def load_column(
    path: str,
    field: str,
    label: str,
    sample_n: int,
    random_sample: bool,
    reference_ids: list[str] | None = None,
) -> Column:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    all_records: list[tuple[int, dict]] = []
    total = 0

    logger.info(f"[load] 读取 {path}  field={field}")
    with p.open(encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            total += 1
            if field in obj:
                all_records.append((lineno, obj))

    # ── 选取记录 ─────────────────────────────────────────────────
    if reference_ids is not None:
        # 非首列：按 reference_ids 顺序精确匹配
        ref_idx = {rid: i for i, rid in enumerate(reference_ids)}
        ref_set = set(reference_ids)
        matched = [
            (ref_idx[_get_id(obj)], obj)
            for _, obj in all_records
            if _get_id(obj) in ref_set
        ]
        matched.sort(key=lambda x: x[0])
        selected = [obj for _, obj in matched]
    elif random_sample and 0 < sample_n < len(all_records):
        selected = [obj for _, obj in random.sample(all_records, sample_n)]
    elif 0 < sample_n:
        selected = [obj for _, obj in all_records[:sample_n]]
    else:
        selected = [obj for _, obj in all_records]

    id_order: list[str] = []
    meta: dict[str, dict] = {}
    html_data: dict[str, str] = {}

    for obj in selected:
        rec_id = _get_id(obj)
        if not rec_id or rec_id in html_data:
            continue
        id_order.append(rec_id)
        html_data[rec_id] = obj[field]
        meta[rec_id] = _extract_meta(obj)

    col = Column(path, field, label, id_order, meta, html_data, total)
    logger.info(f"[load] 完成: {col.loaded:,} 条（文件总行 {total:,}）label={label!r}")
    return col


def detect_fields(path: str) -> list[str]:
    """读取文件前 20 行，检测哪些 HTML 字段存在。"""
    found: set[str] = set()
    try:
        with Path(path).open(encoding="utf-8") as f:
            for i, raw in enumerate(f):
                if i >= 20:
                    break
                try:
                    obj = json.loads(raw)
                    for field in (FIELD_PREPROCESSED, FIELD_OUTPUT):
                        if field in obj:
                            found.add(field)
                except Exception:
                    pass
    except Exception:
        pass
    return [f for f in (FIELD_PREPROCESSED, FIELD_OUTPUT) if f in found]


# ── 全局状态 ─────────────────────────────────────────────────────────────────

_columns: list[Column] = []
_row_ids: list[str] = []
_state_lock = threading.Lock()


def _recompute_row_ids() -> None:
    global _row_ids
    _row_ids = list(_columns[0].id_order) if _columns else []
    logger.info(f"[rows] 基准行数: {len(_row_ids):,}")


# ── Flask routes ──────────────────────────────────────────────────────────────

_HTML_FILE = Path(__file__).parent / "viewer.html"

_PLACEHOLDER_HTML = """<!DOCTYPE html>
<html><body style="
  margin:0;display:flex;align-items:center;justify-content:center;
  height:100vh;font-family:sans-serif;font-size:14px;color:#aaa;">
  — 无数据 —
</body></html>"""


@app.route("/")
def index():
    return _HTML_FILE.read_text(encoding="utf-8"), 200, {
        "Content-Type": "text/html; charset=utf-8"
    }


@app.route("/render/<int:col_idx>/<int:row_idx>")
def render_html(col_idx: int, row_idx: int):
    """
    返回单元格 HTML 内容，注入 <base href> 后作为完整页面响应。
    <iframe src="/render/col/row"> 加载此端点，无 sandbox 限制，
    外部 CSS/图片/JS 资源通过 <base href> 从原始域名正常加载。
    """
    with _state_lock:
        if col_idx < 0 or col_idx >= len(_columns):
            return _PLACEHOLDER_HTML, 200, {"Content-Type": "text/html; charset=utf-8"}
        col = _columns[col_idx]
        if row_idx < 0 or row_idx >= len(_row_ids):
            return _PLACEHOLDER_HTML, 200, {"Content-Type": "text/html; charset=utf-8"}
        rec_id = _row_ids[row_idx]
        html = col.html_data.get(rec_id)
        url = col.meta.get(rec_id, {}).get("url", "")

    if not html:
        html = _PLACEHOLDER_HTML
    else:
        html = inject_base_url(html, url)

    resp = make_response(html, 200)
    resp.headers["Content-Type"] = "text/html; charset=utf-8"
    resp.headers["X-Frame-Options"] = "SAMEORIGIN"
    return resp


@app.route("/api/info")
def api_info():
    with _state_lock:
        return jsonify({
            "columns": [
                {
                    "idx": i,
                    "path": c.path,
                    "field": c.field,
                    "label": c.label,
                    "total_in_file": c.total_in_file,
                    "loaded": c.loaded,
                }
                for i, c in enumerate(_columns)
            ],
            "row_count": len(_row_ids),
        })


@app.route("/api/rows")
def api_rows():
    page = max(0, int(request.args.get("page", 0)))
    size = max(1, min(100, int(request.args.get("size", 10))))
    with _state_lock:
        start = page * size
        page_ids = _row_ids[start: start + size]
        rows = []
        for abs_row_idx, rid in enumerate(page_ids, start=start):
            # 取 meta：从第一个有数据的列
            row_meta: dict = {}
            for col in _columns:
                if rid in col.meta:
                    row_meta = col.meta[rid]
                    break
            cells = []
            for col in _columns:
                if rid in col.html_data:
                    m = col.meta.get(rid, {})
                    cells.append({
                        "has_html": True,
                        "html_len": len(col.html_data[rid]),
                        "model": m.get("model", ""),
                        "prompt_tokens": m.get("prompt_tokens"),
                        "completion_tokens": m.get("completion_tokens"),
                        "finish_reason": m.get("finish_reason", ""),
                    })
                else:
                    cells.append({"has_html": False})
            rows.append({
                "row_idx": abs_row_idx,
                "id": rid,
                "meta": row_meta,
                "cells": cells,
            })
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
        return jsonify({"error": "path required"}), 400
    field = body.get("field", FIELD_OUTPUT)
    if field not in (FIELD_PREPROCESSED, FIELD_OUTPUT):
        return jsonify({"error": "field 必须是 preprocessed_html 或 output_html"}), 400
    sample_n = int(body.get("sample_n", 50))
    random_sample = bool(body.get("random", False))
    label = (body.get("label") or "").strip()
    if not label:
        stem = Path(path).stem
        label = f"{stem} [{'preprocessed' if 'preprocessed' in field else 'output'}]"

    try:
        with _state_lock:
            reference_ids = list(_columns[0].id_order) if _columns else None
        col = load_column(path, field, label, sample_n, random_sample, reference_ids)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    with _state_lock:
        _columns.append(col)
        _recompute_row_ids()
    return jsonify({"ok": True, "col_idx": len(_columns) - 1})


@app.route("/api/columns/<int:idx>", methods=["DELETE"])
def api_remove_column(idx: int):
    with _state_lock:
        if idx < 0 or idx >= len(_columns):
            return jsonify({"error": "index out of range"}), 400
        _columns.pop(idx)
        _recompute_row_ids()
    return jsonify({"ok": True})


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HTML 改写结果多列对比可视化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  # 单文件（生成 preprocessed + output 两列）
  python viewer.py --jsonl output.jsonl --sample-n 20

  # 对比两个模型配置（首文件生成 preprocessed + output，第二文件只生成 output）
  python viewer.py --jsonl output_a.jsonl output_b.jsonl --sample-n 20

  # 只看预处理结果（Stage 1 产出）
  python viewer.py --jsonl preprocessed.jsonl --sample-n 50

  # WSL 下从 Windows 浏览器访问
  python viewer.py --jsonl output.jsonl --host 0.0.0.0 --port 7862
""",
    )
    parser.add_argument("--jsonl", nargs="*", default=[], metavar="FILE")
    parser.add_argument("--sample-n", type=int, default=20, help="加载条数，-1=全部")
    parser.add_argument("--random-sample", action="store_true")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    is_first_file = True
    for path in (args.jsonl or []):
        fields = detect_fields(path)
        if not fields:
            logger.warning(f"未找到可用 HTML 字段 (preprocessed_html/output_html): {path}")
            continue

        # 首文件展示全部字段；后续文件只展示 output（预处理相同，避免重复列）
        show_fields = fields if is_first_file else [f for f in fields if f == FIELD_OUTPUT] or fields

        stem = Path(path).stem
        for field in show_fields:
            field_short = "preprocessed" if "preprocessed" in field else "output"
            label = f"{stem} [{field_short}]"
            try:
                with _state_lock:
                    ref = list(_columns[0].id_order) if _columns else None
                col = load_column(path, field, label, args.sample_n, args.random_sample, ref)
                with _state_lock:
                    _columns.append(col)
                    _recompute_row_ids()
            except Exception as exc:
                logger.error(f"加载失败 {path} field={field}: {exc}")
                sys.exit(1)
        is_first_file = False

    url = f"http://localhost:{args.port}"
    logger.info(f"启动 → {url}")
    print(f"\n  HTML 改写对比工具已就绪: {url}\n")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
