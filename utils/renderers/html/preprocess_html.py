#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预处理 HTML 字符串：将 fonts.googleapis.com/css2? URL 转换为 css? (v1) 格式。

背景：
  css2? API 在某些网络环境下返回 text/html（被拦截），
  导致 Playwright 渲染时出现 CONSOLE_ERROR（MIME 类型不匹配）。
  将 URL 转换为 css? (v1) 格式后，在相同网络条件下往往可以正常返回 CSS，
  从而消除该类 WARN，保留更多可用数据。

转换规则（css2→css1）：
  1. URL 路径 /css2? → /css?
  2. family 参数中 wght@N1;N2;N3 → N1,N2,N3
  3. family 参数中 ital,wght@0,N1;0,N2;1,N3 → N1,N2,N3italic
  4. 多个 &family= 参数 → 合并为 family=A|B 格式
  5. 移除 &display=... 参数（v1 不支持）

用法（独立脚本）:
    python preprocess_html.py input.jsonl -o preprocessed/
    python html_render.py preprocess input.jsonl -o preprocessed/ --html_field output_html
"""

import re
import json
import argparse
import sys
from pathlib import Path

# ── 正则 ──────────────────────────────────────────────────────────────────────

# 匹配含 fonts.googleapis.com 的 <link> 标签（提取 href 用于 URL 转换）
_GFONT_LINK_RE = re.compile(
    r'(<link\b[^>]*href\s*=\s*["\'])([^"\']*fonts\.googleapis\.com[^"\']*?)(["\'][^>]*/?>)',
    re.IGNORECASE,
)

# 匹配 @import 语句中的 Google Fonts URL（url('...') 或 url("...") 或裸引号）
_GFONT_IMPORT_RE = re.compile(
    r'(@import\s+(?:url\s*\(\s*)?["\'])(https?://fonts\.googleapis\.com[^"\']+)(["\'](?:\s*\))?)',
    re.IGNORECASE,
)


# ── URL 转换核心 ───────────────────────────────────────────────────────────────

def _convert_family_spec(spec: str) -> str:
    """
    将单个 family 参数值从 css2 格式转为 css1 格式。
    示例：
      "Roboto:wght@300;400;700"       → "Roboto:300,400,700"
      "Roboto:ital,wght@0,300;1,400"  → "Roboto:300,400italic"
      "Roboto:300,400,700"            → "Roboto:300,400,700"  (不变)
      "Roboto"                        → "Roboto"              (不变)
    """
    if ':' not in spec:
        return spec

    name, variants_str = spec.split(':', 1)

    if '@' not in variants_str:
        return spec  # 已是 v1 格式

    axis_part, values_str = variants_str.split('@', 1)
    axes = [a.strip() for a in axis_part.split(',')]
    tuples = [t.strip() for t in values_str.split(';') if t.strip()]

    if axes == ['wght']:
        # wght@300;400;700 → 300,400,700
        return f"{name}:{','.join(tuples)}"

    if len(axes) == 2 and set(axes) == {'ital', 'wght'}:
        ital_idx = axes.index('ital')
        wght_idx = axes.index('wght')
        v1_parts: list[str] = []
        for tup in tuples:
            vals = tup.split(',')
            if len(vals) <= max(ital_idx, wght_idx):
                continue
            ital = vals[ital_idx].strip()
            wght = vals[wght_idx].strip()
            v1_parts.append(f"{wght}italic" if ital != '0' else wght)
        if v1_parts:
            return f"{name}:{','.join(v1_parts)}"
        return name

    # 未知轴组合，只保留名称
    return name


def convert_gfont_url(url: str) -> str:
    """
    将 fonts.googleapis.com/css2? URL 转换为 css? (v1) 格式。
    非 css2 URL 原样返回。
    """
    if '/css2?' not in url:
        return url

    base, qs = url.split('?', 1)
    base = base.replace('/css2', '/css', 1)

    families: list[str] = []
    other_params: list[str] = []

    for part in qs.split('&'):
        if not part:
            continue
        if part.startswith('family='):
            families.append(_convert_family_spec(part[7:]))
        elif part.startswith('display='):
            pass  # 移除 display= 参数（v1 不支持）
        else:
            other_params.append(part)

    if not families:
        return url

    parts = [f"family={'|'.join(families)}"] + other_params
    return base + '?' + '&'.join(parts)


def preprocess_html_string(html: str) -> str:
    """
    预处理单个 HTML 字符串：将 css2 Google Fonts URL 转换为 css1 格式。
    不含 Google Fonts 引用时直接返回原对象（零拷贝）。
    """
    if 'fonts.googleapis.com' not in html:
        return html

    def _replace_link(m: re.Match) -> str:
        prefix, url, suffix = m.group(1), m.group(2), m.group(3)
        return prefix + convert_gfont_url(url) + suffix

    def _replace_import(m: re.Match) -> str:
        prefix, url, suffix = m.group(1), m.group(2), m.group(3)
        return prefix + convert_gfont_url(url) + suffix

    html = _GFONT_LINK_RE.sub(_replace_link, html)
    html = _GFONT_IMPORT_RE.sub(_replace_import, html)
    return html


# ── 文件处理 ──────────────────────────────────────────────────────────────────

def _process_jsonl(src: Path, out_path: Path, html_field: str) -> tuple[int, int]:
    total = modified = 0
    lines_out: list[str] = []
    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.rstrip("\n\r")
            if not stripped:
                lines_out.append(line)
                continue
            try:
                rec = json.loads(stripped)
            except json.JSONDecodeError:
                lines_out.append(line)
                continue
            total += 1
            html = rec.get(html_field, "")
            new_html = preprocess_html_string(html) if html else html
            if new_html is not html:
                rec = dict(rec)
                rec[html_field] = new_html
                modified += 1
                lines_out.append(json.dumps(rec, ensure_ascii=False) + "\n")
            else:
                lines_out.append(line)
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines_out)
    return total, modified


def _process_json_array(src: Path, out_path: Path, html_field: str) -> tuple[int, int]:
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return 0, 0
    modified = 0
    result = []
    for rec in data:
        html = rec.get(html_field, "")
        new_html = preprocess_html_string(html) if html else html
        if new_html is not html:
            rec = dict(rec)
            rec[html_field] = new_html
            modified += 1
        result.append(rec)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    return len(result), modified


def process_file(src: Path, out_dir: Path | None, html_field: str,
                 in_place: bool) -> tuple[int, int]:
    out_path = src if in_place else (out_dir / src.name)
    if src.suffix.lower() == ".jsonl":
        return _process_jsonl(src, out_path, html_field)
    return _process_json_array(src, out_path, html_field)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="预处理 JSONL/JSON 中的 HTML 字段，将 Google Fonts css2 URL 转换为 css1 格式"
    )
    parser.add_argument("input", nargs="+",
                        help="输入文件（JSON/JSONL）或目录（扫描一级）")
    parser.add_argument("--output", "-o", default=None,
                        help="输出目录（与 --in_place 二选一）")
    parser.add_argument("--in_place", action="store_true",
                        help="直接覆盖原文件（危险）")
    parser.add_argument("--html_field", default="html",
                        help="HTML 内容所在字段名（默认 html）")
    args = parser.parse_args(argv)

    if not args.in_place and not args.output:
        parser.error("必须指定 --output 目录 或 使用 --in_place")

    out_dir = Path(args.output) if args.output else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    files: list[Path] = []
    for inp in args.input:
        p = Path(inp)
        if p.is_dir():
            files.extend(sorted(p.glob("*.json")) + sorted(p.glob("*.jsonl")))
        elif p.is_file():
            files.append(p)
        else:
            print(f"[SKIP] 不存在: {inp}", file=sys.stderr)

    total_records = total_modified = 0
    for src in files:
        t, m = process_file(src, out_dir, args.html_field, args.in_place)
        total_records += t
        total_modified += m
        print(f"  {src.name}: {m}/{t} 条含 Google Fonts css2 引用已转换")

    print(f"\n[DONE] 共 {total_records} 条，转换 {total_modified} 条")


if __name__ == "__main__":
    main()
