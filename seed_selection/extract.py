"""
Step 1 — extract.py

从 canonical schema JSONL 中提取 instruction 文本及辅助元数据，
输出统一的中间记录格式。

中间记录 schema：
{
  "id":          str,   # "{domain}/{source}/{stem}:{line_no}"
  "instruction": str,   # user 侧 text content（原始，未清洗）
  "svg_len":     int,   # assistant 侧 SVG 文本字符数（含 ``` 包裹）
  "domain":      str,   # "stage1_icon" | "stage2_icon" | "stage2_illustration"
  "source":      str,   # "img2svg" | "text2svg"
}
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from loguru import logger

from .io_utils import infer_domain, infer_source, make_id, read_jsonl, write_jsonl


# ── Canonical schema 解析 ──────────────────────────────────────────────

def _get_user_text(data_list: list[dict]) -> Optional[str]:
    """从 data[0].content 中提取 type=="text" 的 string。"""
    if not data_list:
        return None
    user_turn = data_list[0]
    if user_turn.get("role") != "user":
        return None
    for item in user_turn.get("content", []):
        if item.get("type") == "text":
            text_obj = item.get("text", {})
            val = text_obj.get("string", "")
            if isinstance(val, str):
                return val
    return None


def _get_svg_text(data_list: list[dict]) -> Optional[str]:
    """从 data[1].content 中提取 type=="text" 的 string（包含 ```svg 包裹）。"""
    if len(data_list) < 2:
        return None
    asst_turn = data_list[1]
    if asst_turn.get("role") != "assistant":
        return None
    for item in asst_turn.get("content", []):
        if item.get("type") == "text":
            text_obj = item.get("text", {})
            val = text_obj.get("string", "")
            if isinstance(val, str):
                return val
    return None


# ── 统计 ───────────────────────────────────────────────────────────────

@dataclass
class ExtractStats:
    total: int = 0
    extracted: int = 0
    skip_no_instruction: int = 0
    skip_no_svg: int = 0
    skip_parse_error: int = 0

    def report(self) -> str:
        return (
            f"总扫描行数:        {self.total:,}\n"
            f"成功提取:          {self.extracted:,}\n"
            f"跳过（无 instruction）: {self.skip_no_instruction:,}\n"
            f"跳过（无 SVG）:    {self.skip_no_svg:,}\n"
            f"跳过（解析错误）:  {self.skip_parse_error:,}"
        )


# ── 核心逻辑 ───────────────────────────────────────────────────────────

def iter_records(
    input_path: Path,
    dry_run_limit: Optional[int] = None,
) -> Iterator[dict]:
    """逐行提取一个 JSONL 文件中的所有有效记录（不过滤，统计在外部）。"""
    domain = infer_domain(str(input_path))
    source = infer_source(str(input_path))
    # Use domain+source+stem to avoid id collisions when multiple files share the same filename
    file_key = f"{domain}/{source}/{input_path.stem}"

    for line_no, raw in enumerate(read_jsonl(input_path), start=1):
        if dry_run_limit and line_no > dry_run_limit:
            break

        data_list = raw.get("data", [])

        instruction = _get_user_text(data_list)
        if not instruction:
            yield {"_skip": "no_instruction"}
            continue

        svg_text = _get_svg_text(data_list)
        if svg_text is None:
            yield {"_skip": "no_svg"}
            continue

        yield {
            "id":          make_id(file_key, line_no),
            "instruction": instruction,
            "svg_len":     len(svg_text),
            "domain":      domain,
            "source":      source,
        }


def run_extract(
    input_paths: list[Path],
    output_path: Path,
    dry_run_limit: Optional[int] = None,
) -> ExtractStats:
    """
    遍历全部输入文件，提取有效记录，写入 output_path。

    img2svg 文件必须排在 text2svg 之前（由 config.input_paths 顺序保证），
    以确保后续 exact dedup 中 img2svg 记录先出现，优先作为 representative。
    """
    stats = ExtractStats()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fout:
        for path in input_paths:
            logger.info(f"[extract] 扫描: {path.name}")
            file_total = 0

            for rec in iter_records(path, dry_run_limit=dry_run_limit):
                stats.total += 1
                file_total += 1

                skip = rec.get("_skip")
                if skip == "no_instruction":
                    stats.skip_no_instruction += 1
                elif skip == "no_svg":
                    stats.skip_no_svg += 1
                elif skip == "parse_error":
                    stats.skip_parse_error += 1
                else:
                    import json
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    stats.extracted += 1

            logger.info(f"[extract]   ↳ {path.name}: {file_total:,} 行")

    logger.info(f"[extract] 完成\n{stats.report()}")
    return stats
