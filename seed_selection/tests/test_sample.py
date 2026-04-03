"""sample.py 测试：验证 quota 分配、高优先级选取、去重正确性。"""

import json
from pathlib import Path

import pytest

from seed_selection.sample import (
    _allocate_cluster_budget,
    _allocate_quota,
    _select_high_priority,
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
    cluster_path = _make_cluster_assignments(tmp_path)
    run_sample(cluster_path, tmp_path, total_pool_size=10,
               high_priority_size=4, random_seed=42)

    assert (tmp_path / "pool_1000k.jsonl").exists()
    assert (tmp_path / "anneal_pool.jsonl").exists()
    assert (tmp_path / "high_priority_pool.jsonl").exists()


def test_run_sample_counts(tmp_path):
    """anneal + high_priority == pool_1000k."""
    cluster_path = _make_cluster_assignments(tmp_path, n_per_cluster=10, n_clusters=4)
    stats = run_sample(cluster_path, tmp_path,
                       total_pool_size=20, high_priority_size=4, random_seed=42)

    assert stats.pool_1000k == stats.anneal + stats.high_priority


def test_run_sample_high_priority_most_central(tmp_path):
    """high-priority 应包含 distance_to_centroid 最小的样本（每 cluster 代表）。"""
    cluster_path = _make_cluster_assignments(tmp_path, n_per_cluster=5, n_clusters=2)
    run_sample(cluster_path, tmp_path,
               total_pool_size=10, high_priority_size=2, random_seed=42)

    hp = [json.loads(l) for l in (tmp_path / "high_priority_pool.jsonl").read_text().splitlines() if l.strip()]
    distances = [r["_meta"]["distance_to_centroid"] for r in hp]
    assert all(d == 0.0 for d in distances)


def test_run_sample_no_overlap(tmp_path):
    """anneal 和 high_priority 不应有 id 重叠。"""
    cluster_path = _make_cluster_assignments(tmp_path, n_per_cluster=10, n_clusters=4)
    run_sample(cluster_path, tmp_path,
               total_pool_size=20, high_priority_size=4, random_seed=42)

    def ids(f):
        return {json.loads(l)["_meta"]["id"]
                for l in (tmp_path / f).read_text().splitlines() if l.strip()}

    hp_ids = ids("high_priority_pool.jsonl")
    anneal_ids = ids("anneal_pool.jsonl")
    assert hp_ids.isdisjoint(anneal_ids)


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


def test_select_high_priority_round_robin():
    """Phase 2 round-robin：各 cluster 贡献条数应大体均衡。"""
    # 4 个 cluster，各 10 条；要求选 16 条（Phase 1: 4 条 + Phase 2: 12 条）
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

    hp = _select_high_priority(pool, high_priority_size=16)
    assert len(hp) == 16

    from collections import Counter
    cluster_counts = Counter(r["_meta"]["cluster_id"] for r in hp)
    # Round-robin 应使各 cluster 各贡献 4 条（16/4=4）
    for cid in range(n_clusters):
        assert cluster_counts[cid] == 4, f"cluster {cid} contributed {cluster_counts[cid]}"


def test_run_sample_200k_800k_split(tmp_path):
    """验证 200K/800K 默认配比的 anneal + high_priority == pool_1000k。"""
    cluster_path = _make_cluster_assignments(tmp_path, n_per_cluster=10, n_clusters=4)
    stats = run_sample(cluster_path, tmp_path,
                       total_pool_size=40, high_priority_size=8, random_seed=42)
    assert stats.pool_1000k == stats.anneal + stats.high_priority
    assert stats.high_priority <= 8
