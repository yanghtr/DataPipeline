"""
Step 5 — svg_filter.py

SVG 复杂度过滤：移除 stage1_icon / stage2_icon 中 svg_len 最小的 bottom_pct。
stage2_illustration 全部保留。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from loguru import logger

from .io_utils import DOMAINS, read_jsonl


@dataclass
class SVGFilterStats:
    total_per_domain: dict[str, int] = field(default_factory=dict)
    kept_per_domain: dict[str, int] = field(default_factory=dict)
    removed_per_domain: dict[str, int] = field(default_factory=dict)
    cutoff_per_domain: dict[str, int] = field(default_factory=dict)

    def report(self) -> str:
        lines = []
        for domain in DOMAINS:
            t = self.total_per_domain.get(domain, 0)
            k = self.kept_per_domain.get(domain, 0)
            r = self.removed_per_domain.get(domain, 0)
            c = self.cutoff_per_domain.get(domain, "-")
            lines.append(
                f"  {domain}: {t:,} → 保留 {k:,}，移除 {r:,}"
                + (f"（svg_len 截断值={c}）" if r else "")
            )
        return "[svg_filter] 结果:\n" + "\n".join(lines)


# 只对这些 domain 做复杂度过滤
_FILTER_DOMAINS = {"stage1_icon", "stage2_icon"}


def run_svg_filter(
    input_path: Path,
    output_path: Path,
    bottom_pct: float = 0.10,
) -> SVGFilterStats:
    """
    两遍算法：
    1. 第一遍：按 domain 收集所有 svg_len，计算各 domain 的 bottom_pct 截断值
    2. 第二遍（重新读文件）：过滤并写出
    """
    stats = SVGFilterStats()
    domain_svg_lens: dict[str, list[int]] = {d: [] for d in DOMAINS}

    # 第一遍：收集 svg_len
    for rec in read_jsonl(input_path):
        meta = rec.get("_meta", {})
        domain = meta.get("domain", "stage1_icon")
        domain_svg_lens.setdefault(domain, []).append(meta.get("svg_len", 0))

    # 计算截断值
    cutoffs: dict[str, int] = {}
    for domain, lens in domain_svg_lens.items():
        stats.total_per_domain[domain] = len(lens)
        if domain in _FILTER_DOMAINS and lens:
            cutoff = int(np.percentile(lens, bottom_pct * 100))
            cutoffs[domain] = cutoff
            stats.cutoff_per_domain[domain] = cutoff
        else:
            cutoffs[domain] = 0  # 不过滤

    # 第二遍：写出
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for rec in read_jsonl(input_path):
            meta = rec.get("_meta", {})
            domain = meta.get("domain", "stage1_icon")
            cutoff = cutoffs.get(domain, 0)
            if domain in _FILTER_DOMAINS and meta.get("svg_len", 0) <= cutoff:
                stats.removed_per_domain[domain] = stats.removed_per_domain.get(domain, 0) + 1
                continue
            stats.kept_per_domain[domain] = stats.kept_per_domain.get(domain, 0) + 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(stats.report())
    return stats
