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
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from datasketch import MinHash, MinHashLSH
from loguru import logger

from .io_utils import DOMAINS, read_jsonl


# ── MinHash 计算（无状态，可并行）────────────────────────────────────────────

def _compute_minhash_batch(args: tuple) -> list[bytes]:
    """
    Worker：批量计算 MinHash，返回 hashvalues bytes 列表（uint64 array 序列化）。
    顶层函数，可被 ProcessPoolExecutor 序列化。
    """
    texts, num_perm, char_ngram = args
    results = []
    for text in texts:
        m = MinHash(num_perm=num_perm)
        encoded = text.encode("utf-8")
        for i in range(max(1, len(encoded) - char_ngram + 1)):
            m.update(encoded[i : i + char_ngram])
        results.append(m.hashvalues.tobytes())
    return results


def _minhash_from_bytes(hv_bytes: bytes, num_perm: int) -> MinHash:
    """从序列化的 hashvalues bytes 还原 MinHash 对象。"""
    m = MinHash(num_perm=num_perm)
    m.hashvalues = np.frombuffer(hv_bytes, dtype=np.uint64).copy()
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


# ── 域内去重（两阶段）────────────────────────────────────────────────────────

def _dedup_domain(
    records: list[dict],
    threshold: float,
    num_perm: int,
    char_ngram: int,
    num_workers: int = 1,
) -> tuple[list[dict], int]:
    """
    对单个 domain 做 MinHash LSH 去重。

    两阶段策略（num_workers > 1 时启用）：
      Phase 1（并行）：批量计算全部 MinHash hashvalues
      Phase 2（顺序）：用预计算的 hashvalues 做增量 LSH 去重

    真正的瓶颈是 Phase 1（纯 Python 的 n-gram hash）；
    Phase 2 的 LSH query/insert 有状态依赖，必须顺序执行。
    """
    n = len(records)
    texts = [rec.get("instruction", "") for rec in records]

    # ── Phase 1：MinHash 计算 ──────────────────────────────────────────
    if num_workers > 1 and n >= num_workers * 2:
        chunk_size = max(1, (n + num_workers - 1) // num_workers)
        chunks = [
            (texts[i : i + chunk_size], num_perm, char_ngram)
            for i in range(0, n, chunk_size)
        ]
        all_hv_bytes: list[bytes] = []
        with ProcessPoolExecutor(max_workers=num_workers) as exe:
            for batch in exe.map(_compute_minhash_batch, chunks):
                all_hv_bytes.extend(batch)
    else:
        # 单进程：直接顺序计算
        all_hv_bytes = _compute_minhash_batch((texts, num_perm, char_ngram))

    # ── Phase 2：LSH 增量去重（必须顺序，有状态）─────────────────────────
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept: list[dict] = []
    removed = 0

    for i, (rec, hv_bytes) in enumerate(zip(records, all_hv_bytes)):
        m = _minhash_from_bytes(hv_bytes, num_perm)
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
    num_workers: int = 1,
) -> NearDedupStats:
    """
    各 domain 顺序处理，每个 domain 内部 MinHash 计算并行（num_workers 进程）。

    为什么不做域间并行：_dedup_domain 内部已经用了 ProcessPoolExecutor，
    嵌套进程池在 Python 中不可靠；且 stage1_icon 是绝对瓶颈，
    域间并行对它没有帮助，反而增加复杂度。
    """
    stats = NearDedupStats()

    # 分 domain 收集记录（顺序保持 img2svg 优先）
    domain_records: dict[str, list[dict]] = {d: [] for d in DOMAINS}
    for rec in read_jsonl(input_path):
        domain = rec.get("_meta", {}).get("domain", "stage1_icon")
        domain_records.setdefault(domain, []).append(rec)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for domain in DOMAINS:
            records = domain_records.get(domain, [])
            if not records:
                continue
            threshold = thresholds.get(domain, 0.8)
            logger.info(
                f"[dedup_near] {domain}: {len(records):,} 条，"
                f"阈值={threshold}，num_perm={num_perm}，char_ngram={char_ngram}，"
                f"num_workers={num_workers}"
            )
            kept, removed = _dedup_domain(
                records, threshold, num_perm, char_ngram, num_workers
            )
            stats.total_per_domain[domain] = len(records)
            stats.kept_per_domain[domain] = len(kept)
            stats.removed_per_domain[domain] = removed

            for rec in kept:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(stats.report())
    return stats
