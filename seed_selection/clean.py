"""
Step 2 — clean.py

对 instruction 文本进行 Unicode 规范化和基础清洗，过滤无效记录。
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loguru import logger

from .io_utils import read_jsonl, write_jsonl


MIN_INSTRUCTION_LEN = 3   # 清洗后最短有效指令字符数


def clean_instruction(text: str) -> str:
    """Unicode NFC 规范化 + strip。"""
    return unicodedata.normalize("NFC", text).strip()


def is_valid(text: str) -> bool:
    return len(text) >= MIN_INSTRUCTION_LEN


@dataclass
class CleanStats:
    total: int = 0
    kept: int = 0
    skip_empty: int = 0
    skip_too_short: int = 0

    def report(self) -> str:
        return (
            f"总输入:          {self.total:,}\n"
            f"保留:            {self.kept:,}\n"
            f"跳过（空文本）:  {self.skip_empty:,}\n"
            f"跳过（过短）:    {self.skip_too_short:,}"
        )


def run_clean(input_path: Path, output_path: Path) -> CleanStats:
    stats = CleanStats()

    def _process():
        for rec in read_jsonl(input_path):
            stats.total += 1
            cleaned = clean_instruction(rec.get("instruction", ""))
            if not cleaned:
                stats.skip_empty += 1
                continue
            if not is_valid(cleaned):
                stats.skip_too_short += 1
                continue
            rec["instruction"] = cleaned
            stats.kept += 1
            yield rec

    write_jsonl(_process(), output_path)
    logger.info(f"[clean] 完成\n{stats.report()}")
    return stats
