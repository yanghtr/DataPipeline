"""sample.py 测试：验证 quota 分配、三层分级选取、去重正确性。"""

import json
from collections import Counter
from pathlib import Path

import pytest

from seed_selection.sample import (
    _allocate_cluster_budget,
    _allocate_quota,
    _assign_priority_tiers,
    run_sample,
)


# ── 单元测试：辅助函数 ────────────────────────────────────────────────────────

def test_allocate_quota_sum():
    sizes = {"a": 300, "b": 700}
    q = _allocate_quota(sizes, 100)
    assert sum(q.values()) == 100
    assert q["a"] == 30
    assert q["b"] == 70


def test_allocate_quota_empty():
    q = _allocate_quota({}, 100)
    assert q == {}


def test_allocate_cluster_budget_sum():
    sizes = {0: 100, 1: 400, 2: 25}
    budget = _allocate_cluster_budget(sizes, 50)
    assert sum(budget.values()) == 50
    for v in budget.values():
        assert v >= 1


def test_allocate_cluster_budget_more_clusters_than_budget():
    """cluster 数 > budget 时，只覆盖最大的 budget 个 cluster，总和 == budget。"""
    sizes = {i: i + 1 for i in range(100)}
    budget = _allocate_cluster_budget(sizes, 10)
    assert sum(budget.values()) == 10
    assert all(v in (0, 1) for v in budget.values())
    selected = {cid for cid, v in budget.items() if v == 1}
    assert selected == set(range(90, 100))


# ── 集成测试：run_sample ──────────────────────────────────────────────────────

def _make_cluster_assignments(tmp_path: Path, n_per_cluster: int = 5,
                               n_clusters: int = 4, domain: str = "stage1_icon") -> Path:
    p = tmp_path / "cluster.jsonl"
    records = []
    for cid in range(n_clusters):
        for j in range(n_per_cluster):
            idx = cid * n_per_cluster + j
            records.append({
                "instruction": f"instruction {idx}",
                "_meta": {
                    "id":                   f"r:{idx}",
                    "domain":               domain,
                    "source":               "img2svg",
                    "svg_len":              100 + idx,
                    "bucket_key":           domain,
                    "cluster_id":           cid,
                    "cluster_size":         n_per_cluster,
                    "distance_to_centroid": float(j) / 10,
                },
            })
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")
    return p


def test_run_sample_output_files(tmp_path):
    """run_sample 应输出 pool_1000k 和六层分级文件。"""
    cluster_path = _make_cluster_assignments(tmp_path)
    tier_sizes = {"stage1_icon": (2, 4, 4)}
    run_sample(cluster_path, tmp_path, total_pool_size=10,
               tier_sizes=tier_sizes, random_seed=42)

    assert (tmp_path / "pool_1000k.jsonl").exists()
    assert (tmp_path / "stage1_icon_high.jsonl").exists()
    assert (tmp_path / "stage1_icon_medium.jsonl").exists()
    assert (tmp_path / "stage1_icon_low.jsonl").exists()


def test_run_sample_tier_counts_sum(tmp_path):
    """三层条数之和应等于 pool_1000k 条数。"""
    cluster_path = _make_cluster_assignments(tmp_path, n_per_cluster=10, n_clusters=4)
    tier_sizes = {"stage1_icon": (4, 8, 8)}
    stats = run_sample(cluster_path, tmp_path,
                       total_pool_size=20, tier_sizes=tier_sizes, random_seed=42)

    tc = stats.tier_counts.get("stage1_icon", {})
    total_tiers = tc.get("high", 0) + tc.get("medium", 0) + tc.get("low", 0)
    assert total_tiers == stats.pool_1000k


def test_run_sample_high_tier_most_central(tmp_path):
    """高优层应包含各 cluster distance_to_centroid 最小的样本。"""
    cluster_path = _make_cluster_assignments(tmp_path, n_per_cluster=5, n_clusters=2)
    # 高优只取 2 条（每个 cluster 1 条），应为 distance=0.0 的样本
    tier_sizes = {"stage1_icon": (2, 2, 6)}
    run_sample(cluster_path, tmp_path,
               total_pool_size=10, tier_sizes=tier_sizes, random_seed=42)

    high = [json.loads(l) for l in
            (tmp_path / "stage1_icon_high.jsonl").read_text().splitlines() if l.strip()]
    distances = [r["_meta"]["distance_to_centroid"] for r in high]
    assert all(d == 0.0 for d in distances)


def test_run_sample_no_overlap(tmp_path):
    """六层之间不应有 id 重叠，且并集等于 pool_1000k。"""
    cluster_path = _make_cluster_assignments(tmp_path, n_per_cluster=10, n_clusters=4)
    tier_sizes = {"stage1_icon": (4, 8, 8)}
    run_sample(cluster_path, tmp_path,
               total_pool_size=20, tier_sizes=tier_sizes, random_seed=42)

    def ids(f):
        return {json.loads(l)["_meta"]["id"]
                for l in (tmp_path / f).read_text().splitlines() if l.strip()}

    high_ids = ids("stage1_icon_high.jsonl")
    medium_ids = ids("stage1_icon_medium.jsonl")
    low_ids = ids("stage1_icon_low.jsonl")
    pool_ids = ids("pool_1000k.jsonl")

    assert high_ids.isdisjoint(medium_ids)
    assert high_ids.isdisjoint(low_ids)
    assert medium_ids.isdisjoint(low_ids)
    assert high_ids | medium_ids | low_ids == pool_ids


def test_allocate_quota_with_overrides():
    """overrides 覆盖指定 bucket，剩余按比例分配。"""
    sizes = {"stage1_icon": 2250, "stage2_illustration": 450}
    q = _allocate_quota(sizes, 1000, overrides={"stage2_illustration": 300})
    assert q["stage2_illustration"] == 300
    assert q["stage1_icon"] == 700
    assert sum(q.values()) == 1000


def test_allocate_quota_overrides_capped():
    """overrides 超过 bucket 实际大小时，截断到 bucket 大小。"""
    sizes = {"a": 100, "b": 900}
    q = _allocate_quota(sizes, 1000, overrides={"a": 500})  # a 只有 100 条
    assert q["a"] == 100   # 截断
    assert q["b"] == 900
    assert sum(q.values()) == 1000


def test_assign_priority_tiers_round_robin():
    """Round-Robin：各 cluster 在高优层贡献条数应大体均衡。"""
    # 4 个 cluster，各 10 条；高优取 8 条（每 cluster 2 条），中优取 8 条
    n_clusters, n_per = 4, 10
    pool = []
    for cid in range(n_clusters):
        for j in range(n_per):
            pool.append({
                "instruction": f"inst {cid}-{j}",
                "_meta": {
                    "id": f"r:{cid}:{j}",
                    "bucket_key": "stage1_icon",
                    "cluster_id": cid,
                    "distance_to_centroid": float(j) / n_per,
                },
            })

    tier_sizes = {"stage1_icon": (8, 8, 24)}
    result = _assign_priority_tiers(pool, tier_sizes)
    tiers = result["stage1_icon"]

    assert len(tiers["high"]) == 8
    assert len(tiers["medium"]) == 8
    assert len(tiers["low"]) == 24

    # Round-Robin 应使各 cluster 在高优层各贡献 2 条（8/4=2）
    high_cluster_counts = Counter(r["_meta"]["cluster_id"] for r in tiers["high"])
    for cid in range(n_clusters):
        assert high_cluster_counts[cid] == 2, \
            f"cluster {cid} contributed {high_cluster_counts[cid]} to high tier"


def test_assign_priority_tiers_quality_monotone():
    """高优层平均 distance 应小于中优层，中优层应小于低优层。"""
    n_clusters, n_per = 3, 9
    pool = []
    for cid in range(n_clusters):
        for j in range(n_per):
            pool.append({
                "instruction": f"inst {cid}-{j}",
                "_meta": {
                    "id": f"r:{cid}:{j}",
                    "bucket_key": "stage1_icon",
                    "cluster_id": cid,
                    "distance_to_centroid": float(j),
                },
            })

    tier_sizes = {"stage1_icon": (3, 6, 18)}
    result = _assign_priority_tiers(pool, tier_sizes)
    tiers = result["stage1_icon"]

    def avg_dist(recs):
        return sum(r["_meta"]["distance_to_centroid"] for r in recs) / len(recs)

    assert avg_dist(tiers["high"]) < avg_dist(tiers["medium"])
    assert avg_dist(tiers["medium"]) < avg_dist(tiers["low"])


def test_run_sample_tier_counts_match_tier_sizes(tmp_path):
    """高优和中优的实际条数应等于 tier_sizes 配置值（在数据充足时）。"""
    cluster_path = _make_cluster_assignments(tmp_path, n_per_cluster=10, n_clusters=4)
    tier_sizes = {"stage1_icon": (4, 8, 8)}
    stats = run_sample(cluster_path, tmp_path,
                       total_pool_size=40, tier_sizes=tier_sizes, random_seed=42)
    tc = stats.tier_counts.get("stage1_icon", {})
    assert tc.get("high", 0) <= 4
    assert tc.get("medium", 0) <= 8
