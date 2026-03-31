"""
Step 4 — dedup_near.py

基于 MinHash LSH 的近似去重，按 domain 分组处理（不同阈值）。

设计：
- 字符级 n-gram（char_ngram=5），适合短文本
- 增量式 LSH：顺序插入，遇到已有 near-dup 则跳过
- 由于输入中 img2svg 记录在前，自然保留 img2svg 版本
- 内存估算：128 perm × 4B × 3M records ≈ 1.5 GB（可接受）
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from datasketch import MinHash, MinHashLSH
from loguru import logger

from .io_utils import DOMAINS, read_jsonl, write_jsonl


def _compute_minhash(text: str, num_perm: int, char_ngram: int) -> MinHash:
    m = MinHash(num_perm=num_perm)
    encoded = text.encode("utf-8")
    for i in range(max(1, len(encoded) - char_ngram + 1)):
        m.update(encoded[i : i + char_ngram])
    return m


@dataclass
class NearDedupStats:
    total_per_domain: dict[str, int] = field(default_factory=dict)
    kept_per_domain: dict[str, int] = field(default_factory=dict)
    removed_per_domain: dict[str, int] = field(default_factory=dict)

    def report(self) -> str:
        lines = []
        for domain in DOMAINS:
            t = self.total_per_domain.get(domain, 0)
            k = self.kept_per_domain.get(domain, 0)
            r = self.removed_per_domain.get(domain, 0)
            lines.append(f"  {domain}: {t:,} → 保留 {k:,}，去重 {r:,}")
        return "[near_dedup] 各 domain 结果:\n" + "\n".join(lines)


def _dedup_domain(
    records: list[dict],
    threshold: float,
    num_perm: int,
    char_ngram: int,
) -> tuple[list[dict], int]:
    """对单个 domain 的记录做 MinHash LSH 去重，返回 (保留列表, 去重数)。"""
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept: list[dict] = []
    removed = 0

    for i, rec in enumerate(records):
        m = _compute_minhash(rec["instruction"], num_perm, char_ngram)
        key = f"r{i}"
        if not lsh.query(m):
            lsh.insert(key, m)
            kept.append(rec)
        else:
            removed += 1

    return kept, removed


def run_dedup_near(
    input_path: Path,
    output_path: Path,
    thresholds: dict[str, float],
    num_perm: int = 128,
    char_ngram: int = 5,
) -> NearDedupStats:
    stats = NearDedupStats()

    # 分 domain 收集记录（顺序保持 img2svg 优先）
    domain_records: dict[str, list[dict]] = {d: [] for d in DOMAINS}
    for rec in read_jsonl(input_path):
        domain = rec.get("domain", "stage1_icon")
        if domain in domain_records:
            domain_records[domain].append(rec)
        else:
            domain_records.setdefault(domain, []).append(rec)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for domain, records in domain_records.items():
            threshold = thresholds.get(domain, 0.8)
            stats.total_per_domain[domain] = len(records)
            logger.info(
                f"[dedup_near] {domain}: {len(records):,} 条，"
                f"阈值={threshold}，num_perm={num_perm}，char_ngram={char_ngram}"
            )
            kept, removed = _dedup_domain(records, threshold, num_perm, char_ngram)
            stats.kept_per_domain[domain] = len(kept)
            stats.removed_per_domain[domain] = removed

            for rec in kept:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(stats.report())
    return stats
