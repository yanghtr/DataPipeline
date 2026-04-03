"""
Step 8 — sample.py

从聚类结果中采样，产出三个文件：
  - pool_1000k.jsonl         总采样池（anneal + high-priority 合并）
  - anneal_pool.jsonl        800k anneal 池（从 1000k 中去掉 high-priority 部分）
  - high_priority_pool.jsonl 200k high-priority 池（各 cluster 最中心样本）

采样策略：
  1. 各 bucket 按比例分配 quota（支持 bucket_quota_overrides 显式覆盖）
  2. bucket 内各 cluster 按 sqrt(cluster_size) 分配 budget，最少 1
  3. cluster 内按 distance_to_centroid 升序（最近=最中心）选 top-budget
  4. 200k high-priority 两阶段选取：
     - Phase 1：每 (bucket, cluster) 取 distance 最小的 1 条 → 约 15K 条
     - Phase 2：Round-robin 轮转各 cluster，依次取下一条最近记录，直到 200K
  5. 800k anneal = pool_1000k - high_priority_pool
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


def _allocate_quota(
    bucket_sizes: dict[str, int],
    total: int,
    overrides: dict[str, int] | None = None,
) -> dict[str, int]:
    """
    按 bucket_size 比例分配 quota，整数，确保总和 == total。

    overrides 中显式指定的 bucket 直接使用给定配额，
    剩余 total 按比例分配给其余 bucket。
    """
    if not bucket_sizes:
        return {}

    overrides = overrides or {}
    result: dict[str, int] = {}

    # 先处理 overrides（clip 到实际数据量，不能超过 bucket 大小）
    override_total = 0
    for k, v in overrides.items():
        if k in bucket_sizes:
            capped = min(v, bucket_sizes[k])
            result[k] = capped
            override_total += capped

    remaining_total = max(0, total - override_total)
    free_buckets = {k: sz for k, sz in bucket_sizes.items() if k not in result}

    if not free_buckets:
        return result

    total_free_size = sum(free_buckets.values())
    if total_free_size == 0:
        for k in free_buckets:
            result[k] = 0
        return result

    raw = {k: remaining_total * v / total_free_size for k, v in free_buckets.items()}
    floored = {k: int(v) for k, v in raw.items()}
    remainder = remaining_total - sum(floored.values())
    keys_by_frac = sorted(raw, key=lambda k: raw[k] - floored[k], reverse=True)
    for k in keys_by_frac[:remainder]:
        floored[k] += 1

    result.update(floored)
    return result


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


def _select_high_priority(
    pool_records: list[dict],
    high_priority_size: int,
) -> list[dict]:
    """
    从 pool_1000k 中选取 high_priority_pool。

    两阶段：
      Phase 1：每个 (bucket_key, cluster_id) 取 distance_to_centroid 最小的 1 条
               → 约覆盖全部 cluster（约 15K 条）
      Phase 2：Round-robin 轮转所有 cluster，依次追加每个 cluster 的下一条最近记录，
               直到达到 high_priority_size。
               → 各 cluster 贡献条数均衡，不因 distance 绝对值而倾斜。

    相比全局 distance 排序补充（旧方案），round-robin 避免了 dense cluster
    （距离绝对值小）在 Phase 2 中贡献过多记录，保证语义多样性。
    """
    # 按 (bucket, cluster) 分组，各 cluster 内已排好序
    cluster_queues: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for rec in pool_records:
        meta = rec.get("_meta", {})
        key = (meta.get("bucket_key", ""), meta.get("cluster_id", 0))
        cluster_queues[key].append(rec)

    # 各 cluster 内按 distance 升序排列，建立独立指针
    for key in cluster_queues:
        cluster_queues[key].sort(
            key=lambda r: r.get("_meta", {}).get("distance_to_centroid", 0.0)
        )

    cluster_keys = list(cluster_queues.keys())
    pointers: dict[tuple[str, int], int] = {k: 0 for k in cluster_keys}

    hp_records: list[dict] = []
    hp_ids: set[str] = set()

    # Phase 1：每 cluster 取第一条（distance 最小）
    for key in cluster_keys:
        queue = cluster_queues[key]
        if queue:
            rec = queue[0]
            hp_records.append(rec)
            hp_ids.add(rec["_meta"]["id"])
            pointers[key] = 1
        if len(hp_records) >= high_priority_size:
            break

    # Phase 2：round-robin 补充至 high_priority_size
    if len(hp_records) < high_priority_size:
        # 循环直到无新记录可取或达到目标
        made_progress = True
        while len(hp_records) < high_priority_size and made_progress:
            made_progress = False
            for key in cluster_keys:
                if len(hp_records) >= high_priority_size:
                    break
                ptr = pointers[key]
                queue = cluster_queues[key]
                while ptr < len(queue):
                    rec = queue[ptr]
                    ptr += 1
                    if rec["_meta"]["id"] not in hp_ids:
                        hp_records.append(rec)
                        hp_ids.add(rec["_meta"]["id"])
                        made_progress = True
                        break
                pointers[key] = ptr

    return hp_records


def run_sample(
    input_path: Path,
    output_dir: Path,
    total_pool_size: int = 1_000_000,
    high_priority_size: int = 200_000,
    random_seed: int = 42,
    bucket_quota_overrides: dict[str, int] | None = None,
) -> SampleStats:
    random.seed(random_seed)
    stats = SampleStats()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 读取全部记录，按 bucket 分组
    bucket_records: dict[str, list[dict]] = defaultdict(list)
    for rec in read_jsonl(input_path):
        stats.total_input += 1
        meta = rec.get("_meta", {})
        bucket_records[meta.get("bucket_key", meta.get("domain", "stage1_icon"))].append(rec)

    # 2. 按 bucket 比例分配 1000k quota（支持 overrides）
    bucket_sizes = {k: len(v) for k, v in bucket_records.items()}
    bucket_quotas = _allocate_quota(
        bucket_sizes,
        min(total_pool_size, stats.total_input),
        overrides=bucket_quota_overrides,
    )
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
            cluster_groups[rec.get("_meta", {}).get("cluster_id", 0)].append(rec)

        cluster_sizes = {cid: len(recs) for cid, recs in cluster_groups.items()}
        cluster_budgets = _allocate_cluster_budget(cluster_sizes, quota)

        for cid, recs in cluster_groups.items():
            budget = min(cluster_budgets.get(cid, 1), len(recs))
            # 按 distance_to_centroid 升序，取前 budget 条
            sorted_recs = sorted(recs, key=lambda r: r.get("_meta", {}).get("distance_to_centroid", 0))
            pool_records.extend(sorted_recs[:budget])

    # 稳定打乱（不影响 downstream reproducibility）
    random.shuffle(pool_records)

    # 4. 写 pool_1000k
    pool_path = output_dir / "pool_1000k.jsonl"
    with pool_path.open("w", encoding="utf-8") as f:
        for rec in pool_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    stats.pool_1000k = len(pool_records)

    # 5. 选 high-priority（two-phase round-robin）
    hp_records = _select_high_priority(pool_records, high_priority_size)

    hp_path = output_dir / "high_priority_pool.jsonl"
    with hp_path.open("w", encoding="utf-8") as f:
        for rec in hp_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    stats.high_priority = len(hp_records)

    # 6. anneal = pool_1000k - high_priority
    hp_ids_set = {r["_meta"]["id"] for r in hp_records}
    anneal_path = output_dir / "anneal_pool.jsonl"
    with anneal_path.open("w", encoding="utf-8") as f:
        for rec in pool_records:
            if rec["_meta"]["id"] not in hp_ids_set:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats.anneal += 1

    logger.info(f"[sample] 完成\n{stats.report()}")
    return stats
