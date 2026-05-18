#!/usr/bin/env python3
"""
用 Playwright 对 img2svg 的问题 case 重新渲染，输出到 images_rerender。
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.renderers.svg import get_svg_dimensions
from utils.renderers.svg_playwright import render_svg_playwright

JSONL = r"C:\j00934267\data\SAgoge_stage2\gemini-3.1-pro-preview-thinking_svg2\img2svg\jsonl\data_000000.jsonl"
OUT_DIR = r"C:\j00934267\data\SAgoge_stage2\gemini-3.1-pro-preview-thinking_svg2\img2svg\images_rerender\00000"
os.makedirs(OUT_DIR, exist_ok=True)

with open(JSONL, encoding="utf-8") as f:
    lines = f.readlines()

# (输出文件名, JSONL行号)
CASES = [
    ("000000020.png",  20),
    ("000000142.png", 142),
    ("000000172.png", 172),
    ("000000214.png", 213),
    ("000000224.png", 223),
    ("000000230.png", 229),
    ("000000233.png", 232),
    ("000000247.png", 246),
    ("000000260.png", 259),
    ("000000282.png", 281),
    ("000000290.png", 289),
    ("000000293.png", 292),
    ("000000303.png", 302),
    ("000000329.png", 328),
    ("000000330.png", 329),
    ("000000346.png", 344),
    ("000000347.png", 345),
    ("000000368.png", 366),
    ("000000400.png", 398),
    ("000000467.png", 465),
    ("000000483.png", 481),
]

ok = fail = 0
for fname, lno in CASES:
    item = json.loads(lines[lno])
    text = item["data"][1]["content"][0]["text"]["string"]
    m = re.search(r"<svg\b.*?</svg>", text, re.DOTALL | re.IGNORECASE)
    if not m:
        print(f"[SKIP] {fname}: no SVG found in line {lno}")
        fail += 1
        continue

    svg_code = m.group(0)
    w, h = get_svg_dimensions(svg_code)
    out_path = os.path.join(OUT_DIR, fname)
    result = render_svg_playwright(svg_code, out_path, width=w, height=h, timeout=30)

    if result.success:
        print(f"[OK]   {fname}  {result.width}x{result.height}")
        ok += 1
    else:
        print(f"[FAIL] {fname}: {result.error}")
        fail += 1

print(f"\n完成: {ok} 成功, {fail} 失败")
