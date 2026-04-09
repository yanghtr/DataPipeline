"""
Step 8 — sample.py

从聚类结果中采样，产出七个文件：
  - pool_1000k.jsonl                   总采样池（六层合并）
  - stage1_icon_high.jsonl             stage1_icon 高优（100K）
  - stage1_icon_medium.jsonl           stage1_icon 中优（200K）
  - stage1_icon_low.jsonl              stage1_icon 低优（400K）
  - stage2_illustration_high.jsonl     stage2_illustration 高优（100K）
  - stage2_illustration_medium.jsonl   stage2_illustration 中优（100K）
  - stage2_illustration_low.jsonl      stage2_illustration 低优（100K）

采样策略：
  1. 各 bucket 按比例分配 quota（支持 bucket_quota_overrides 显式覆盖）
  2. bucket 内各 cluster 按 sqrt(cluster_size) 分配 budget，最少 1
  3. cluster 内按 distance_to_centroid 升序（最近=最中心）选 top-budget
  4. 六层分层：对每个 bucket 独立执行三阶段 Round-Robin
     - 各 cluster 内记录已按 distance_to_centroid 升序排列
     - 循环遍历所有 cluster（Round-Robin），依次取下一条最近记录：
       · 前 tier_sizes[bucket][0] 条 → 高优（high）
       · 接下来 tier_sizes[bucket][1] 条 → 中优（medium）
       · 剩余 → 低优（low）
     - 各 cluster 贡献条数均衡，层间质量单调（高优平均距离最小）
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

TIER_NAMES = ("high", "medium", "low")


@dataclass
class SampleStats:
    total_input: int = 0
    pool_1000k: int = 0
    tier_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    budget_per_bucket: dict[str, int] = field(default_factory=dict)

    def report(self) -> str:
        lines = [
            f"总输入:    {self.total_input:,}",
            f"1000k 池:  {self.pool_1000k:,}",
        ]
        for bucket, tiers in self.tier_counts.items():
            h = tiers.get("high", 0)
            m = tiers.get("medium", 0)
            lo = tiers.get("low", 0)
            lines.append(f"  {bucket}: 高优 {h:,} / 中优 {m:,} / 低优 {lo:,}")
        return "\n".join(lines)


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


def _assign_priority_tiers(
    pool_records: list[dict],
    tier_sizes: dict[str, tuple[int, int, int]],
) -> dict[str, dict[str, list[dict]]]:
    """
    从 pool_1000k 中，对每个 bucket 独立执行三阶段 Round-Robin，
    将记录分配到高优（high）/ 中优（medium）/ 低优（low）三层。

    对每个 bucket：
      1. 按 distance_to_centroid 升序对各 cluster 内记录排序
      2. Round-Robin 循环遍历所有 cluster，依次取下一条最近记录
      3. 前 tier_sizes[bucket][0] 条 → high
      4. 接下来 tier_sizes[bucket][1] 条 → medium
      5. 剩余所有记录 → low

    Round-Robin 保证各 cluster 贡献条数均衡（不因 distance 绝对值而倾斜）；
    高优平均 distance 最小，层间质量单调。
    """
    # 1. 按 bucket_key 分组
    bucket_pool: dict[str, list[dict]] = defaultdict(list)
    for rec in pool_records:
        bk = rec.get("_meta", {}).get("bucket_key", "")
        bucket_pool[bk].append(rec)

    result: dict[str, dict[str, list[dict]]] = {}

    for bucket, records in bucket_pool.items():
        sizes = tier_sizes.get(bucket, (0, 0, len(records)))
        high_size, medium_size = sizes[0], sizes[1]

        # 2. 按 cluster_id 分组并按 distance 升序排列
        cluster_queues: dict[int, list[dict]] = defaultdict(list)
        for rec in records:
            cid = rec.get("_meta", {}).get("cluster_id", 0)
            cluster_queues[cid].append(rec)
        for cid in cluster_queues:
            cluster_queues[cid].sort(
                key=lambda r: r.get("_meta", {}).get("distance_to_centroid", 0.0)
            )

        cluster_keys = list(cluster_queues.keys())
        pointers: dict[int, int] = {k: 0 for k in cluster_keys}
        tiers: dict[str, list[dict]] = {"high": [], "medium": [], "low": []}

        def _fill_tier(tier_name: str, target: int) -> None:
            if target <= 0:
                return
            made_progress = True
            while len(tiers[tier_name]) < target and made_progress:
                made_progress = False
                for key in cluster_keys:
                    if len(tiers[tier_name]) >= target:
                        break
                    ptr = pointers[key]
                    if ptr < len(cluster_queues[key]):
                        tiers[tier_name].append(cluster_queues[key][ptr])
                        pointers[key] = ptr + 1
                        made_progress = True

        # 3. 依次填充高优和中优，指针连续前进
        _fill_tier("high", high_size)
        _fill_tier("medium", medium_size)

        # 4. 剩余全部归低优（指针后所有记录）
        for key in cluster_keys:
            tiers["low"].extend(cluster_queues[key][pointers[key]:])

        result[bucket] = tiers

    return result


def run_sample(
    input_path: Path,
    output_dir: Path,
    total_pool_size: int = 1_000_000,
    tier_sizes: dict[str, tuple[int, int, int]] | None = None,
    random_seed: int = 42,
    bucket_quota_overrides: dict[str, int] | None = None,
) -> SampleStats:
    if tier_sizes is None:
        tier_sizes = {
            "stage1_icon":         (100_000, 200_000, 400_000),
            "stage2_illustration": (100_000, 100_000, 100_000),
        }

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

    # 3. 采样 pool_1000k（三层 cluster 配额分配）
    pool_records: list[dict] = []

    for bucket, records in bucket_records.items():
        quota = bucket_quotas.get(bucket, 0)
        if quota == 0:
            continue

        cluster_groups: dict[int, list[dict]] = defaultdict(list)
        for rec in records:
            cluster_groups[rec.get("_meta", {}).get("cluster_id", 0)].append(rec)

        cluster_sizes = {cid: len(recs) for cid, recs in cluster_groups.items()}
        cluster_budgets = _allocate_cluster_budget(cluster_sizes, quota)

        for cid, recs in cluster_groups.items():
            budget = min(cluster_budgets.get(cid, 1), len(recs))
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

    # 5. 分域三阶段 Round-Robin 分层
    tier_result = _assign_priority_tiers(pool_records, tier_sizes)

    # 6. 写六个分层文件
    tier_label_map = {"high": "高优", "medium": "中优", "low": "低优"}
    for bucket, tiers in tier_result.items():
        stats.tier_counts[bucket] = {}
        for tier_name in TIER_NAMES:
            recs = tiers.get(tier_name, [])
            fname = f"{bucket}_{tier_name}.jsonl"
            fpath = output_dir / fname
            with fpath.open("w", encoding="utf-8") as f:
                for rec in recs:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats.tier_counts[bucket][tier_name] = len(recs)
            logger.info(
                f"[sample] {bucket} {tier_label_map[tier_name]} → {fname} ({len(recs):,} 条)"
            )

    logger.info(f"[sample] 完成\n{stats.report()}")
    return stats
