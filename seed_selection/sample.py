"""
Step 8 — sample.py

从聚类/FPS 结果中采样，产出七个文件：
  - pool_1000k.jsonl                   总采样池（六层合并）
  - stage1_icon_high.jsonl             stage1_icon 高优（100K）
  - stage1_icon_medium.jsonl           stage1_icon 中优（200K）
  - stage1_icon_low.jsonl              stage1_icon 低优（400K）
  - stage2_illustration_high.jsonl     stage2_illustration 高优（50K）
  - stage2_illustration_medium.jsonl   stage2_illustration 中优（100K）
  - stage2_illustration_low.jsonl      stage2_illustration 低优（150K）

采样策略：

stage1_icon（KMeans K=100K + 类内 FPS）：
  1. 按 bucket 比例分配 700K quota（支持 bucket_quota_overrides）
  2. 各 cluster 按 sqrt(cluster_size) 分配 budget（容量感知约束再分配，确保 sum == quota）
  3. 每个 cluster 内按 fps_round 升序取前 budget 条（FPS 最早选出的 = 最有代表性）
  4. Round-Robin 三阶段分层：
     - 各 cluster 内记录按 fps_round 升序排列
     - 循环 Round-Robin 遍历所有 cluster，依次取下一条：
       · 前 tier_sizes[bucket][0] 条 → 高优（high）
       · 接下来 tier_sizes[bucket][1] 条 → 中优（medium）
       · 剩余 → 低优（low）
     理论上（K=100K, avg n_c=7）：round 1 → 100K high，round 2-3 → 200K medium，
     round 4-7 → 400K low

stage2_illustration（直接全局 FPS）：
  1. 按 fps_rank 升序取前 300K（已在 cluster.py 中由 FPS 选出）
  2. 直接按 fps_rank 顺序切分：
     - 前 tier_sizes[bucket][0] → 高优（fps 最早选出，多样性最大）
     - 接下来 tier_sizes[bucket][1] → 中优
     - 剩余 → 低优
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


def _allocate_cluster_budget_constrained(
    cluster_sizes: dict[int, int],
    total_budget: int,
) -> dict[int, int]:
    """
    按 sqrt(cluster_size) 分配 budget，总和恰好等于 total_budget。

    容量感知约束再分配（Capacity-Aware Constrained Redistribution）：
      1. 按 sqrt(size) 比例初始分配（每个 cluster 至少 1）
      2. 将超过 cluster_size 的 budget 截断（cap at cluster_size）
      3. 将超出部分（excess）按 sqrt(size) 重新分配给还有容量的 cluster
      4. 迭代至收敛（最多 20 轮），确保 sum(budgets) == total_budget

    当 cluster 数 > total_budget 时：按 size 降序选前 total_budget 个，各给 1。
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

    # 初始分配：按 sqrt(size) 比例，每个至少 1
    import math
    sqrt_sizes = {cid: math.sqrt(sz) for cid, sz in cluster_sizes.items()}
    total_sqrt = sum(sqrt_sizes.values())

    budget: dict[int, int] = {}
    for cid, s in sqrt_sizes.items():
        budget[cid] = max(1, int(total_budget * s / total_sqrt))

    # 迭代约束再分配
    for _ in range(20):
        # 截断超出 cluster_size 的部分
        excess = 0
        for cid in list(budget):
            cap = cluster_sizes[cid]
            if budget[cid] > cap:
                excess += budget[cid] - cap
                budget[cid] = cap

        if excess == 0:
            break

        # 将 excess 按 sqrt(remaining_capacity) 分配给还有容量的 cluster
        remaining_capacity = {
            cid: cluster_sizes[cid] - budget[cid]
            for cid in budget
            if cluster_sizes[cid] > budget[cid]
        }
        if not remaining_capacity:
            break

        total_rem_sqrt = sum(math.sqrt(v) for v in remaining_capacity.values())
        if total_rem_sqrt == 0:
            break

        for cid, cap in remaining_capacity.items():
            delta = int(excess * math.sqrt(cap) / total_rem_sqrt)
            budget[cid] = min(budget[cid] + delta, cluster_sizes[cid])

    # 最终微调：确保 sum == total_budget（可能因取整有 ±几 的误差）
    current_total = sum(budget.values())
    diff = total_budget - current_total

    if diff > 0:
        # 还有余量可增加的 cluster（按 remaining capacity 降序）
        with_capacity = sorted(
            [cid for cid in budget if budget[cid] < cluster_sizes[cid]],
            key=lambda c: cluster_sizes[c] - budget[c],
            reverse=True,
        )
        for cid in with_capacity:
            if diff <= 0:
                break
            add = min(diff, cluster_sizes[cid] - budget[cid])
            budget[cid] += add
            diff -= add
    elif diff < 0:
        # 需要减少（优先从 budget 最大的 cluster 减）
        by_budget = sorted(budget, key=lambda c: budget[c], reverse=True)
        for cid in by_budget:
            if diff >= 0:
                break
            reduce = min(-diff, budget[cid] - 1)
            if reduce > 0:
                budget[cid] -= reduce
                diff += reduce

    return budget


def _assign_priority_tiers(
    pool_records: list[dict],
    tier_sizes: dict[str, tuple[int, int, int]],
) -> dict[str, dict[str, list[dict]]]:
    """
    从 pool 中，对每个 bucket 独立分配到高优/中优/低优三层。

    stage1_icon：Round-Robin 三阶段分层
      - 各 cluster 内记录按 fps_round 升序排列（若无 fps_round 则按 distance_to_centroid）
      - Round-Robin 循环遍历所有 cluster，依次取下一条：
        · 前 tier_sizes[bucket][0] 条 → high
        · 接下来 tier_sizes[bucket][1] 条 → medium
        · 剩余 → low
      Round-Robin 保证各 cluster 贡献条数均衡；
      fps_round 排序保证高优包含每个 cluster 最具代表性的样本。

    stage2_illustration：直接按 fps_rank 顺序切分
      - 记录已按 fps_rank 升序排列（pool 建立时已排序）
      - 直接切分：前 high_size → high，接下来 medium_size → medium，其余 → low
      - fps_rank 越小 = FPS 越早选出 = 对全局覆盖贡献越大 = 越高优
    """
    # 按 bucket_key 分组
    bucket_pool: dict[str, list[dict]] = defaultdict(list)
    for rec in pool_records:
        bk = rec.get("_meta", {}).get("bucket_key", "")
        bucket_pool[bk].append(rec)

    result: dict[str, dict[str, list[dict]]] = {}

    for bucket, records in bucket_pool.items():
        sizes = tier_sizes.get(bucket, (0, 0, len(records)))
        high_size, medium_size = sizes[0], sizes[1]

        # stage2_illustration：直接按 fps_rank 切分
        if bucket == "stage2_illustration":
            sorted_recs = sorted(
                records,
                key=lambda r: r.get("_meta", {}).get("fps_rank", 0),
            )
            result[bucket] = {
                "high":   sorted_recs[:high_size],
                "medium": sorted_recs[high_size:high_size + medium_size],
                "low":    sorted_recs[high_size + medium_size:],
            }
            continue

        # stage1_icon 及其他 bucket：Round-Robin + fps_round 排序
        cluster_queues: dict[int, list[dict]] = defaultdict(list)
        for rec in records:
            cid = rec.get("_meta", {}).get("cluster_id", 0)
            cluster_queues[cid].append(rec)

        def _sort_key(r: dict) -> float:
            meta = r.get("_meta", {})
            fps_r = meta.get("fps_round")
            if fps_r is not None:
                return float(fps_r)
            return meta.get("distance_to_centroid", 0.0)

        for cid in cluster_queues:
            cluster_queues[cid].sort(key=_sort_key)

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

        _fill_tier("high", high_size)
        _fill_tier("medium", medium_size)

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
            "stage2_illustration": (50_000, 100_000, 150_000),
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

    # 3. 采样 pool_1000k
    pool_records: list[dict] = []

    for bucket, records in bucket_records.items():
        quota = bucket_quotas.get(bucket, 0)
        if quota == 0:
            continue

        # stage2_illustration：按 fps_rank 升序取前 quota 条
        if bucket == "stage2_illustration":
            selected = [r for r in records if r.get("_meta", {}).get("fps_rank", 0) > 0]
            selected.sort(key=lambda r: r.get("_meta", {}).get("fps_rank", 0))
            pool_records.extend(selected[:quota])
            logger.info(
                f"[sample] {bucket}: fps_rank 采样 {min(len(selected), quota):,}/{len(records):,}"
            )
            continue

        # stage1_icon 及其他 bucket：容量感知约束 cluster 配额分配
        cluster_groups: dict[int, list[dict]] = defaultdict(list)
        for rec in records:
            cluster_groups[rec.get("_meta", {}).get("cluster_id", 0)].append(rec)

        cluster_sizes = {cid: len(recs) for cid, recs in cluster_groups.items()}
        cluster_budgets = _allocate_cluster_budget_constrained(cluster_sizes, quota)

        for cid, recs in cluster_groups.items():
            budget = cluster_budgets.get(cid, 0)
            if budget == 0:
                continue
            # 按 fps_round 升序取前 budget 条（FPS 最早选出的 = 最有代表性）
            sorted_recs = sorted(
                recs,
                key=lambda r: r.get("_meta", {}).get(
                    "fps_round",
                    r.get("_meta", {}).get("distance_to_centroid", 0),
                ),
            )
            pool_records.extend(sorted_recs[:budget])

    # 稳定打乱（不影响 downstream reproducibility）
    random.shuffle(pool_records)

    # 4. 写 pool_1000k
    pool_path = output_dir / "pool_1000k.jsonl"
    with pool_path.open("w", encoding="utf-8") as f:
        for rec in pool_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    stats.pool_1000k = len(pool_records)

    # 5. 分 bucket 三阶段分层
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
