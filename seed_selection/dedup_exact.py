"""
Step 3 — dedup_exact.py

基于 instruction 文本的精确去重。

去重策略：
- 同一 instruction 出现多次时，保留 source 优先级最高的版本
- img2svg（priority=0）优先于 text2svg（priority=1）
- 相同 source 时保留先出现的（文件顺序即优先级）
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from .io_utils import SOURCE_PRIORITY, read_jsonl, write_jsonl


@dataclass
class ExactDedupStats:
    total: int = 0
    kept: int = 0
    replaced: int = 0    # img2svg 替换了 text2svg representative
    removed: int = 0     # 被去重的条数

    def report(self) -> str:
        return (
            f"总输入:        {self.total:,}\n"
            f"保留（唯一）:  {self.kept:,}\n"
            f"去重移除:      {self.removed:,}\n"
            f"优先替换:      {self.replaced:,}（img2svg 替换 text2svg）"
        )


def run_dedup_exact(input_path: Path, output_path: Path) -> ExactDedupStats:
    """
    单遍扫描：dict keyed by instruction，img2svg 优先。
    输出保持 img2svg 记录在前（因为输入中 img2svg 已排在前面）。
    """
    stats = ExactDedupStats()
    # key: instruction → (priority, record)
    seen: dict[str, tuple[int, dict]] = {}

    for rec in read_jsonl(input_path):
        stats.total += 1
        instr = rec.get("instruction", "")
        priority = SOURCE_PRIORITY.get(rec.get("_meta", {}).get("source", "text2svg"), 1)

        if instr not in seen:
            seen[instr] = (priority, rec)
        else:
            existing_priority, _ = seen[instr]
            if priority < existing_priority:
                # img2svg 替换已有的 text2svg
                seen[instr] = (priority, rec)
                stats.replaced += 1
            # 否则保持现有（first-come-wins within same priority）

    # 按 insertion order 输出（Python 3.7+ dict 保序）
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for _priority, rec in seen.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    stats.kept = len(seen)
    stats.removed = stats.total - stats.kept
    logger.info(f"[dedup_exact] 完成\n{stats.report()}")
    return stats
