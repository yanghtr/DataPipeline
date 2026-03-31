"""
Step 8 — sample.py

从聚类结果中采样，产出三个文件：
  - pool_1000k.jsonl         总采样池（anneal + high-priority 合并）
  - anneal_pool.jsonl        900k anneal 池（从 1000k 中去掉 high-priority 部分）
  - high_priority_pool.jsonl 100k high-priority 池（各 cluster 最中心样本）

采样策略：
  1. 各 bucket 按比例分配 quota（prop ∝ bucket_size）
  2. bucket 内各 cluster 按 sqrt(cluster_size) 分配 budget，最少 1
  3. cluster 内按 distance_to_centroid 升序（最近=最中心）选 top-budget
  4. 100k high-priority = 各 cluster 中 distance 最小的 1 条（不足时按 distance 补）
  5. 900k anneal = pool_1000k - high_priority_pool
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger

from .io_utils import DOMAINS, read_jsonl, write_jsonl


@dataclass
class SampleStats:
    total_input: int = 0
    pool_1000k: int = 0
    high_priority: int = 0
    anneal: int = 0
    budget_per_bucket: dict[str, int] = field(default_factory=dict)

    def report(self) -> str:
        return (
            f"总输入:           {self.total_input:,}\n"
            f"1000k 池:         {self.pool_1000k:,}\n"
            f"high-priority 池: {self.high_priority:,}\n"
            f"anneal 池:        {self.anneal:,}"
        )


def _allocate_quota(bucket_sizes: dict[str, int], total: int) -> dict[str, int]:
    """按 bucket_size 比例分配 quota，整数，确保总和 == total。"""
    total_size = sum(bucket_sizes.values())
    if total_size == 0:
        return {k: 0 for k in bucket_sizes}

    raw = {k: total * v / total_size for k, v in bucket_sizes.items()}
    floored = {k: int(v) for k, v in raw.items()}
    remainder = total - sum(floored.values())
    # 按小数部分降序分配剩余
    keys_by_frac = sorted(raw, key=lambda k: raw[k] - floored[k], reverse=True)
    for k in keys_by_frac[:remainder]:
        floored[k] += 1
    return floored


def _allocate_cluster_budget(cluster_sizes: dict[int, int], total_budget: int) -> dict[int, int]:
    """
    按 sqrt(cluster_size) 分配 budget，总和恰好等于 total_budget。

    - 当 cluster 数 <= total_budget 时：每个 cluster 至少分到 1
    - 当 cluster 数 > total_budget 时：只给 total_budget 个最大 cluster 各分 1，
      其余 cluster 分配 0（budget 不足以覆盖所有 cluster）
    """
    if not cluster_sizes or total_budget <= 0:
        return {cid: 0 for cid in cluster_sizes}

    n_clusters = len(cluster_sizes)

    # cluster 数超过 budget：按 size 降序选前 total_budget 个，各给 1
    if n_clusters >= total_budget:
        sorted_cids = sorted(cluster_sizes, key=lambda c: cluster_sizes[c], reverse=True)
        result = {cid: 0 for cid in cluster_sizes}
        for cid in sorted_cids[:total_budget]:
            result[cid] = 1
        return result

    # 正常情况：按 sqrt(size) 比例分配，每个至少 1
    sqrt_sizes = {cid: math.sqrt(sz) for cid, sz in cluster_sizes.items()}
    total_sqrt = sum(sqrt_sizes.values())

    raw = {cid: total_budget * s / total_sqrt for cid, s in sqrt_sizes.items()}
    floored = {cid: max(1, int(v)) for cid, v in raw.items()}
    current_total = sum(floored.values())

    diff = total_budget - current_total
    if diff > 0:
        keys_by_frac = sorted(raw, key=lambda k: raw[k] - floored[k], reverse=True)
        for k in keys_by_frac:
            if diff <= 0:
                break
            floored[k] += 1
            diff -= 1
    elif diff < 0:
        keys_by_size = sorted(floored, key=lambda k: floored[k], reverse=True)
        for k in keys_by_size:
            if diff >= 0:
                break
            if floored[k] > 1:
                floored[k] -= 1
                diff += 1

    return floored


def run_sample(
    input_path: Path,
    output_dir: Path,
    total_pool_size: int = 1_000_000,
    high_priority_size: int = 100_000,
    random_seed: int = 42,
) -> SampleStats:
    random.seed(random_seed)
    stats = SampleStats()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 读取全部记录，按 bucket 分组
    bucket_records: dict[str, list[dict]] = defaultdict(list)
    for rec in read_jsonl(input_path):
        stats.total_input += 1
        bucket_records[rec.get("bucket_key", rec.get("domain", "stage1_icon"))].append(rec)

    # 2. 按 bucket 比例分配 1000k quota
    bucket_sizes = {k: len(v) for k, v in bucket_records.items()}
    bucket_quotas = _allocate_quota(bucket_sizes, min(total_pool_size, stats.total_input))
    for k, q in bucket_quotas.items():
        stats.budget_per_bucket[k] = q

    # 3. 采样 1000k
    pool_records: list[dict] = []

    for bucket, records in bucket_records.items():
        quota = bucket_quotas.get(bucket, 0)
        if quota == 0:
            continue

        # 按 cluster 分组
        cluster_groups: dict[int, list[dict]] = defaultdict(list)
        for rec in records:
            cluster_groups[rec.get("cluster_id", 0)].append(rec)

        cluster_sizes = {cid: len(recs) for cid, recs in cluster_groups.items()}
        cluster_budgets = _allocate_cluster_budget(cluster_sizes, quota)

        for cid, recs in cluster_groups.items():
            budget = min(cluster_budgets.get(cid, 1), len(recs))
            # 按 distance_to_centroid 升序，取前 budget 条
            sorted_recs = sorted(recs, key=lambda r: r.get("distance_to_centroid", 0))
            pool_records.extend(sorted_recs[:budget])

    # 稳定打乱（不影响 downstream reproducibility）
    random.shuffle(pool_records)

    # 4. 写 pool_1000k
    pool_path = output_dir / "pool_1000k.jsonl"
    with pool_path.open("w", encoding="utf-8") as f:
        for rec in pool_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    stats.pool_1000k = len(pool_records)

    # 5. 选 high-priority：各 cluster 中 distance 最小的 1 条
    seen_cluster: set[tuple[str, int]] = set()
    hp_records: list[dict] = []

    # 先按 distance 排序所有 pool 记录
    sorted_pool = sorted(pool_records, key=lambda r: r.get("distance_to_centroid", 0))
    for rec in sorted_pool:
        key = (rec.get("bucket_key", ""), rec.get("cluster_id", 0))
        if key not in seen_cluster:
            seen_cluster.add(key)
            hp_records.append(rec)
        if len(hp_records) >= high_priority_size:
            break

    # 如果 cluster 数不足 high_priority_size，按 distance 补充（不重复）
    if len(hp_records) < high_priority_size:
        hp_ids = {r["id"] for r in hp_records}
        for rec in sorted_pool:
            if rec["id"] not in hp_ids:
                hp_records.append(rec)
                hp_ids.add(rec["id"])
            if len(hp_records) >= high_priority_size:
                break

    hp_path = output_dir / "high_priority_pool.jsonl"
    with hp_path.open("w", encoding="utf-8") as f:
        for rec in hp_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    stats.high_priority = len(hp_records)

    # 6. anneal = pool_1000k - high_priority
    hp_ids_set = {r["id"] for r in hp_records}
    anneal_path = output_dir / "anneal_pool.jsonl"
    with anneal_path.open("w", encoding="utf-8") as f:
        for rec in pool_records:
            if rec["id"] not in hp_ids_set:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats.anneal += 1

    logger.info(f"[sample] 完成\n{stats.report()}")
    return stats
