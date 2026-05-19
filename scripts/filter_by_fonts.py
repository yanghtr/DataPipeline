#!/usr/bin/env python3
"""
scripts/filter_by_fonts.py — 按字体黑名单过滤 JSONL 数据集

从 canonical schema JSONL 文件（或目录）中读取记录，提取 assistant 响应里的 SVG，
若 SVG 用到字体黑名单中的任意字体则丢弃该条记录，其余写入输出目录。

用法：
  # 过滤单个文件
  python scripts/filter_by_fonts.py \\
      --input  path/to/data_000000.jsonl \\
      --fonts  path/to/cannot_install_svg_fonts.txt \\
      --output-dir path/to/jsonl_filtered/

  # 过滤整个目录
  python scripts/filter_by_fonts.py \\
      --input  path/to/jsonl/ \\
      --fonts  path/to/cannot_install_svg_fonts.txt \\
      --output-dir path/to/jsonl_filtered/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


# ── SVG 提取正则 ──────────────────────────────────────────────────────────────

_SVG_FENCE_RE = re.compile(r"```svg\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_SVG_BARE_RE = re.compile(r"<svg\b.*?</svg>", re.DOTALL | re.IGNORECASE)

# font-family 属性值 / CSS 属性值
_ATTR_RE = re.compile(r'\bfont-family\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)
_CSS_RE = re.compile(r'\bfont-family\s*:\s*([^;{}"\'<\n]+)', re.IGNORECASE)

_GENERIC_FONTS = {
    "serif", "sans-serif", "monospace", "cursive", "fantasy",
    "system-ui", "ui-serif", "ui-sans-serif", "ui-monospace",
    "ui-rounded", "inherit", "initial", "unset", "revert",
}


def _extract_svg(text: str) -> str | None:
    fenced = _SVG_FENCE_RE.findall(text)
    if fenced:
        cand = fenced[-1].strip()
        if "<svg" in cand.lower():
            return cand
    bare = _SVG_BARE_RE.findall(text)
    return bare[-1].strip() if bare else None


def _split_font_family(raw: str) -> list[str]:
    result = []
    for part in raw.split(","):
        name = part.strip().strip("'\"").strip()
        if name and name.lower() not in _GENERIC_FONTS:
            result.append(name)
    return result


def _svg_fonts(svg_code: str) -> list[str]:
    """返回 SVG 中出现的所有字体名（去重，保留顺序）。"""
    seen: set[str] = set()
    out: list[str] = []
    for raw in _ATTR_RE.findall(svg_code) + _CSS_RE.findall(svg_code):
        for font in _split_font_family(raw):
            if font not in seen:
                seen.add(font)
                out.append(font)
    return out


def _get_response_text(record: dict) -> str:
    text = ""
    for turn in record.get("data", []):
        if turn.get("role") == "assistant":
            for item in turn.get("content", []):
                if item.get("type") == "text":
                    text += item.get("text", {}).get("string", "")
    return text


def _build_blocklist(fonts_file: Path) -> set[str]:
    """读取字体黑名单，返回小写集合（用于大小写不敏感匹配）。"""
    lines = fonts_file.read_text(encoding="utf-8").splitlines()
    return {line.strip().lower() for line in lines if line.strip()}


def filter_jsonl(
    src: Path,
    dst: Path,
    blocklist: set[str],
) -> dict:
    """过滤单个 JSONL 文件，返回统计信息 dict。"""
    total = 0
    kept = 0
    # 每个黑名单字体独立命中次数（一条记录可被多个字体命中）
    font_hits: Counter = Counter()
    # 记录哪些字体命中过（用于最终报告仅显示实际命中的字体）
    blocked_records = 0

    dst.parent.mkdir(parents=True, exist_ok=True)

    with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                kept += 1
                fout.write(line + "\n")
                continue

            text = _get_response_text(record)
            svg = _extract_svg(text) if text else None

            if svg is None:
                # 无 SVG：保留
                kept += 1
                fout.write(line + "\n")
                continue

            fonts_in_svg = _svg_fonts(svg)
            hit_fonts = [f for f in fonts_in_svg if f.lower() in blocklist]

            if hit_fonts:
                blocked_records += 1
                for f in hit_fonts:
                    # 统计时用黑名单中的原始大小写（匹配 blocklist 条目）
                    font_hits[f.lower()] += 1
            else:
                kept += 1
                fout.write(line + "\n")

    return {
        "total": total,
        "kept": kept,
        "removed": blocked_records,
        "font_hits": font_hits,
    }


def _canonical_font_name(font_lower: str, fonts_file: Path) -> str:
    """从原始 txt 文件中找回对应的原始大小写字体名。"""
    for line in fonts_file.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if name.lower() == font_lower:
            return name
    return font_lower


def _print_stats(
    file_stats: list[tuple[str, dict]],
    fonts_file: Path,
    blocklist: set[str],
) -> None:
    sep = "=" * 64

    # 汇总
    total_all = sum(s["total"] for _, s in file_stats)
    kept_all = sum(s["kept"] for _, s in file_stats)
    removed_all = sum(s["removed"] for _, s in file_stats)
    combined_hits: Counter = Counter()
    for _, s in file_stats:
        combined_hits.update(s["font_hits"])

    print(sep)
    print("  字体过滤统计")
    print(sep)

    if len(file_stats) > 1:
        print(f"\n  [汇总] 共处理 {len(file_stats)} 个文件")
        print(f"  {'过滤前总条数':<14}: {total_all:>8,}")
        print(f"  {'过滤后总条数':<14}: {kept_all:>8,}")
        print(f"  {'移除条数':<14}: {removed_all:>8,}  ({removed_all/total_all*100:.1f}%)")

    for fname, s in file_stats:
        print(f"\n  文件: {fname}")
        print(f"  {'过滤前':<12}: {s['total']:>8,} 条")
        print(f"  {'过滤后':<12}: {s['kept']:>8,} 条")
        print(f"  {'已移除':<12}: {s['removed']:>8,} 条", end="")
        if s["total"] > 0:
            print(f"  ({s['removed']/s['total']*100:.1f}%)", end="")
        print()

    # 按字体分类（汇总或单文件均显示）
    hits = combined_hits if len(file_stats) > 1 else file_stats[0][1]["font_hits"]
    if hits:
        print(f"\n  按字体命中明细（一条记录可被多个字体命中，各自独立计数）：")
        # 只显示黑名单中的字体，未命中的显示 0
        all_fonts_ordered = [
            line.strip()
            for line in fonts_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        max_name = max(len(f) for f in all_fonts_ordered)
        for font in all_fonts_ordered:
            count = hits.get(font.lower(), 0)
            marker = "[x]" if count > 0 else "   "
            print(f"  {marker} {font:<{max_name}} : {count:>6,} 条记录命中")
    else:
        print("\n  未命中任何黑名单字体，所有记录均保留。")

    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="按字体黑名单过滤 JSONL 数据集（SVG font-family）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", "-i", required=True, type=Path,
                        help="JSONL 文件路径或包含 JSONL 的目录")
    parser.add_argument("--fonts", "-f", required=True, type=Path,
                        help="字体黑名单 txt 文件（每行一个字体名）")
    parser.add_argument("--output-dir", "-o", required=True, type=Path,
                        help="过滤后 JSONL 的输出目录")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"[ERROR] 输入路径不存在: {args.input}", file=sys.stderr)
        sys.exit(1)
    if not args.fonts.exists():
        print(f"[ERROR] 字体黑名单文件不存在: {args.fonts}", file=sys.stderr)
        sys.exit(1)

    blocklist = _build_blocklist(args.fonts)
    print(f"字体黑名单: {len(blocklist)} 个字体")

    # 收集输入文件
    if args.input.is_file():
        src_files = [args.input]
    else:
        src_files = sorted(args.input.rglob("*.jsonl"))
    if not src_files:
        print(f"[ERROR] 未找到 JSONL 文件: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"处理 {len(src_files)} 个文件 → 输出目录: {args.output_dir}\n")

    file_stats: list[tuple[str, dict]] = []
    for src in src_files:
        dst = args.output_dir / src.name
        stats = filter_jsonl(src, dst, blocklist)
        file_stats.append((src.name, stats))
        print(f"  [ok] {src.name}: {stats['total']} -> {stats['kept']} 条（移除 {stats['removed']}）")

    print()
    _print_stats(file_stats, args.fonts, blocklist)


if __name__ == "__main__":
    main()
