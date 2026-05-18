#!/usr/bin/env python3
"""
scripts/detect_svg_fonts.py — 扫描 JSONL 数据集中 SVG 所需的所有字体

从 canonical schema JSONL 文件（或目录）中提取每条记录 assistant 响应里的 SVG 代码，
收集所有 font-family 声明（属性值 + CSS <style> 块），去重排序后写入 txt 文件。

输出格式（每行一个字体名，按出现频次降序排列）：
  Arial
  -apple-system
  BlinkMacSystemFont
  ...

用法：
  # 扫描单个 JSONL 文件
  python scripts/detect_svg_fonts.py --input data.jsonl --output fonts.txt

  # 扫描目录下所有 JSONL 文件
  python scripts/detect_svg_fonts.py --input /path/to/jsonl_dir/ --output fonts.txt

  # 只统计出现次数 >= N 的字体
  python scripts/detect_svg_fonts.py --input data.jsonl --output fonts.txt --min-count 2

  # 打印详细统计（含出现次数）
  python scripts/detect_svg_fonts.py --input data.jsonl --output fonts.txt --verbose
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


# ── 正则 ──────────────────────────────────────────────────────────────────────

# SVG 属性：font-family="..." 或 font-family='...'
_ATTR_RE = re.compile(
    r'\bfont-family\s*=\s*["\']([^"\']+)["\']',
    re.IGNORECASE,
)

# CSS 属性：font-family: ...;  （style 块 / style 属性内）
_CSS_RE = re.compile(
    r'\bfont-family\s*:\s*([^;{}"\'<\n]+)',
    re.IGNORECASE,
)

# SVG 代码块（```svg ... ``` 优先，回退到裸 <svg>）
_SVG_FENCE_RE = re.compile(r"```svg\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_SVG_BARE_RE = re.compile(r"<svg\b.*?</svg>", re.DOTALL | re.IGNORECASE)


def _extract_svg(response_text: str) -> str | None:
    """从 assistant 回复中提取最后一段 SVG 代码。"""
    fenced = _SVG_FENCE_RE.findall(response_text)
    if fenced:
        candidate = fenced[-1].strip()
        if "<svg" in candidate.lower():
            return candidate
    bare = _SVG_BARE_RE.findall(response_text)
    return bare[-1].strip() if bare else None


def _parse_font_list(raw: str) -> list[str]:
    """将 font-family 值字符串拆分为单个字体名列表。

    处理：逗号分隔、去除引号、去除空白、过滤通用族关键字。
    """
    generic = {"serif", "sans-serif", "monospace", "cursive", "fantasy", "system-ui",
               "ui-serif", "ui-sans-serif", "ui-monospace", "ui-rounded", "inherit",
               "initial", "unset", "revert"}
    fonts = []
    for part in raw.split(","):
        name = part.strip().strip("'\"").strip()
        if name and name.lower() not in generic:
            fonts.append(name)
    return fonts


def extract_fonts_from_svg(svg_code: str) -> list[str]:
    """从 SVG 代码中提取所有字体名（去重，保留顺序）。"""
    seen: set[str] = set()
    result: list[str] = []
    for raw in _ATTR_RE.findall(svg_code) + _CSS_RE.findall(svg_code):
        for font in _parse_font_list(raw):
            if font not in seen:
                seen.add(font)
                result.append(font)
    return result


def scan_jsonl(path: Path) -> Counter:
    """扫描单个 JSONL 文件，返回字体出现次数计数器。"""
    counter: Counter = Counter()
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            # 从 assistant turn 提取文本
            data = record.get("data", [])
            response_text = ""
            for turn in data:
                if turn.get("role") == "assistant":
                    for item in turn.get("content", []):
                        if item.get("type") == "text":
                            t = item.get("text", {})
                            response_text += t.get("string", "")

            if not response_text:
                continue

            svg_code = _extract_svg(response_text)
            if not svg_code:
                continue

            for font in extract_fonts_from_svg(svg_code):
                counter[font] += 1

    return counter


def collect_jsonl_files(input_path: Path) -> list[Path]:
    """收集输入路径下所有 JSONL 文件。"""
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.rglob("*.jsonl"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="扫描 JSONL 数据集中 SVG 所需字体",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", "-i", required=True, type=Path,
                        help="JSONL 文件路径或包含 JSONL 的目录")
    parser.add_argument("--output", "-o", required=True, type=Path,
                        help="输出 txt 文件路径")
    parser.add_argument("--min-count", type=int, default=1, metavar="N",
                        help="只输出出现次数 >= N 的字体（默认: 1）")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="同时在终端打印字体名和出现次数")
    args = parser.parse_args()

    files = collect_jsonl_files(args.input)
    if not files:
        print(f"[ERROR] 未找到 JSONL 文件: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"扫描 {len(files)} 个 JSONL 文件...")
    total_counter: Counter = Counter()
    for f in files:
        cnt = scan_jsonl(f)
        total_counter.update(cnt)
        print(f"  {f.name}: {sum(cnt.values())} 处字体引用，{len(cnt)} 种字体")

    # 按出现次数降序，同频次按字母升序
    sorted_fonts = sorted(
        ((font, count) for font, count in total_counter.items() if count >= args.min_count),
        key=lambda x: (-x[1], x[0].lower()),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fout:
        for font, count in sorted_fonts:
            fout.write(f"{font}\n")

    print(f"\n共找到 {len(sorted_fonts)} 种字体（min-count={args.min_count}），已写入: {args.output}")

    if args.verbose:
        print("\n字体列表（出现次数）：")
        for font, count in sorted_fonts:
            print(f"  {count:>6}  {font}")


if __name__ == "__main__":
    main()
