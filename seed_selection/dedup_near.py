"""
Step 4 — dedup_near.py

基于 MinHash LSH 的近似去重，按 domain 分组处理（不同阈值）。

设计：
- 字符级 n-gram（char_ngram=5），适合短文本
- 增量式 LSH：顺序插入，遇到已有 near-dup 则跳过
- 由于输入中 img2svg 记录在前，自然保留 img2svg 版本
- 内存估算：128 perm × 8B × 3M records ≈ 3 GB（可接受）

两阶段并行设计：
  Phase 1（并行）：ProcessPoolExecutor 批量计算 MinHash hashvalues，
                   序列化为 uint64 bytes，传回主进程。
  Phase 2（顺序）：将全部 hashvalues 拼成 numpy 矩阵，直接在矩阵上
                   计算 band hash（byteswap + tobytes），完全绕过
                   MinHash 对象构造（每次构造 ~335μs，138× 瓶颈）。
                   使用 Python dict 做 band hash 桶去重，~400K rec/s。
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from datasketch import MinHash, MinHashLSH  # MinHash used in Phase 1 worker; MinHashLSH for band params
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

    Phase 1（并行）：ProcessPoolExecutor 批量计算 MinHash hashvalues，
                     每个 worker 返回 uint64 bytes 列表。
    Phase 2（顺序）：将全部 hashvalues 拼成 (n, num_perm) numpy 矩阵，
                     直接计算 band hash（byteswap + tobytes），完全绕过
                     MinHash 对象构造。band hash 参数通过创建一个 dummy
                     MinHashLSH 获取，保证与 datasketch 语义一致。

    性能对比（2.8M 条，stage1_icon）：
      旧 Phase 2：MinHash() 构造 ×2.8M ≈ 22 分钟
      新 Phase 2：直接 numpy band hash  ≈  0.1 分钟（138× 加速）
    """
    n = len(records)
    if n == 0:
        return [], 0

    texts = [rec.get("instruction", "") for rec in records]

    # ── Phase 1：并行计算 MinHash hashvalues ─────────────────────────
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
        all_hv_bytes = _compute_minhash_batch((texts, num_perm, char_ngram))

    # 将全部 hashvalues 拼成矩阵，避免 Phase 2 中重复构造 numpy 数组
    hv_matrix = np.frombuffer(
        b"".join(all_hv_bytes), dtype=np.uint64
    ).reshape(n, num_perm).copy()
    del all_hv_bytes  # 释放原始 bytes 列表

    # ── Phase 2：直接 band hash 去重（无 MinHash 对象）───────────────
    # 用 dummy LSH 获取 band 分割参数，保证与 datasketch 内部一致
    _dummy = MinHashLSH(threshold=threshold, num_perm=num_perm)
    hashranges: list[tuple[int, int]] = _dummy.hashranges
    del _dummy

    n_bands = len(hashranges)
    band_tables: list[dict] = [{} for _ in range(n_bands)]
    kept: list[dict] = []
    removed = 0

    for i in range(n):
        hv = hv_matrix[i]
        # 一次性计算该记录所有 band hash key（与 datasketch._H 语义相同）
        keys = [hv[s:e].byteswap().tobytes() for s, e in hashranges]
        if any(keys[b] in band_tables[b] for b in range(n_bands)):
            removed += 1
        else:
            for b, key in enumerate(keys):
                band_tables[b][key] = True
            kept.append(records[i])

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
