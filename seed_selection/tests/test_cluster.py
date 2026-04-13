"""cluster.py 测试：使用小规模合成 embeddings，不依赖真实模型。"""

import json
from pathlib import Path

import numpy as np
import pytest

from seed_selection.cluster import (
    run_cluster,
    _fps_within_clusters,
    _fps_global_cpu,
)


def _write_meta(tmp_path: Path, records: list[dict]) -> Path:
    p = tmp_path / "filtered.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")
    return p


def _write_embeddings(tmp_path: Path, ids: list[str], dim: int = 16) -> Path:
    emb_dir = tmp_path / "embeddings"
    emb_dir.mkdir()
    embs = np.random.default_rng(42).random((len(ids), dim), dtype=np.float64).astype(np.float32)
    np.savez_compressed(emb_dir / "shard_0000.npz",
                        ids=np.array(ids, dtype=object), embeddings=embs)
    return emb_dir


def _read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _rec(id_: str, instruction: str, domain: str) -> dict:
    return {
        "instruction": instruction,
        "_meta": {"id": id_, "domain": domain, "source": "img2svg", "svg_len": 100},
    }


# ── FPS 单元测试 ──────────────────────────────────────────────────────────────

def test_fps_within_clusters_assigns_all_records():
    """_fps_within_clusters 应为所有记录赋非零 fps_round。"""
    np.random.seed(42)
    n, d, k = 50, 8, 5
    emb = np.random.randn(n, d).astype(np.float32)
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, random_state=42, n_init=3).fit(emb)
    fps_rounds = _fps_within_clusters(emb, km.labels_, km.cluster_centers_)

    assert fps_rounds.shape == (n,)
    assert (fps_rounds > 0).all(), "All records should have fps_round >= 1"


def test_fps_within_clusters_round_starts_at_1():
    """每个 cluster 内 fps_round 最小值应为 1（anchor = 距质心最近）。"""
    np.random.seed(7)
    n, d, k = 30, 4, 3
    emb = np.random.randn(n, d).astype(np.float32)
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, random_state=42, n_init=3).fit(emb)
    fps_rounds = _fps_within_clusters(emb, km.labels_, km.cluster_centers_)

    labels = km.labels_
    for cid in range(k):
        mask = labels == cid
        cluster_rounds = fps_rounds[mask]
        assert cluster_rounds.min() == 1, f"Cluster {cid}: min fps_round should be 1"
        # rounds should be 1..n_c without gaps
        assert set(cluster_rounds) == set(range(1, mask.sum() + 1))


def test_fps_within_clusters_single_record():
    """单条记录的 cluster，fps_round 应为 1。"""
    emb = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    labels = np.array([0])
    centroids = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    fps_rounds = _fps_within_clusters(emb, labels, centroids)
    assert fps_rounds[0] == 1


def test_fps_global_cpu_returns_unique_indices():
    """_fps_global_cpu 应返回不重复的索引。"""
    np.random.seed(99)
    emb = np.random.randn(30, 4).astype(np.float32)
    fps_idx = _fps_global_cpu(emb, 10)
    assert len(fps_idx) == 10
    assert len(set(fps_idx.tolist())) == 10


def test_fps_global_cpu_maximizes_coverage():
    """FPS 应选出彼此尽可能远的点（简单验证：相邻选出的点距离不为零）。"""
    # 构造 10 个相互远离的点（均匀分布在超球面上）
    np.random.seed(0)
    n, d = 20, 4
    emb = np.random.randn(n, d).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    fps_idx = _fps_global_cpu(emb, 5)

    # 每对选出的点之间距离均大于 0
    selected = emb[fps_idx]
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            dist = float(np.linalg.norm(selected[i] - selected[j]))
            assert dist > 0, f"FPS selected duplicate-like points at ({i},{j})"


# ── run_cluster 集成测试 ──────────────────────────────────────────────────────

def test_cluster_assigns_all_records(tmp_path):
    """stage1_icon：所有记录应获得 cluster_id、fps_round 等字段。"""
    n = 20
    ids = [f"f:{i}" for i in range(n)]
    records = [_rec(ids[i], f"text {i}", "stage1_icon") for i in range(n)]
    meta = _write_meta(tmp_path, records)
    emb_dir = _write_embeddings(tmp_path, ids)
    out = tmp_path / "cluster.jsonl"

    run_cluster(meta, emb_dir, out,
                k_per_bucket={"stage1_icon": 5}, random_seed=42)

    result = _read(out)
    assert len(result) == n
    for rec in result:
        m = rec["_meta"]
        assert "cluster_id" in m
        assert "cluster_size" in m
        assert "distance_to_centroid" in m
        assert "bucket_key" in m
        assert "fps_round" in m
        assert m["fps_round"] >= 1


def test_cluster_multi_domain(tmp_path):
    """两桶：stage1_icon 有 fps_round；stage2_illustration 有 fps_rank。"""
    ids_icon = [f"icon:{i}" for i in range(10)]
    ids_ill  = [f"ill:{i}"  for i in range(5)]
    all_ids  = ids_icon + ids_ill

    records = (
        [_rec(ids_icon[i], f"icon {i}", "stage1_icon") for i in range(10)]
        + [_rec(ids_ill[i],  f"ill {i}",  "stage2_illustration") for i in range(5)]
    )
    meta = _write_meta(tmp_path, records)
    emb_dir = _write_embeddings(tmp_path, all_ids)
    out = tmp_path / "cluster.jsonl"

    run_cluster(meta, emb_dir, out,
                k_per_bucket={"stage1_icon": 3, "stage2_illustration": 0},
                fps_n_select_per_bucket={"stage2_illustration": 5},
                random_seed=42)

    result = _read(out)
    assert len(result) == 15
    domains = {r["_meta"]["bucket_key"] for r in result}
    assert "stage1_icon" in domains
    assert "stage2_illustration" in domains
    assert "stage2_icon" not in domains

    icon_recs = [r for r in result if r["_meta"]["bucket_key"] == "stage1_icon"]
    illus_recs = [r for r in result if r["_meta"]["bucket_key"] == "stage2_illustration"]

    # stage1_icon 应有 fps_round
    for rec in icon_recs:
        assert "fps_round" in rec["_meta"]
        assert rec["_meta"]["fps_round"] >= 1

    # stage2_illustration 应有 fps_rank（0 = 未被选中，>0 = 被 FPS 选中）
    for rec in illus_recs:
        assert "fps_rank" in rec["_meta"]


def test_cluster_stage2_fps_rank_range(tmp_path):
    """stage2_illustration fps_rank：选中的记录 fps_rank >= 1，其余为 0。"""
    ids = [f"ill:{i}" for i in range(10)]
    records = [_rec(ids[i], f"text {i}", "stage2_illustration") for i in range(10)]
    meta = _write_meta(tmp_path, records)
    emb_dir = _write_embeddings(tmp_path, ids)
    out = tmp_path / "cluster.jsonl"

    # 选 6 条 FPS
    run_cluster(meta, emb_dir, out,
                k_per_bucket={"stage2_illustration": 0},
                fps_n_select_per_bucket={"stage2_illustration": 6},
                random_seed=42)

    result = _read(out)
    assert len(result) == 10
    fps_ranks = [r["_meta"]["fps_rank"] for r in result]
    selected = [r for r in fps_ranks if r > 0]
    unselected = [r for r in fps_ranks if r == 0]
    assert len(selected) == 6
    assert len(unselected) == 4
    # selected ranks 应是 1..6 的排列
    assert sorted(selected) == list(range(1, 7))


def test_cluster_k_exceeds_samples(tmp_path):
    """K > 样本数时应自动降低 K，不崩溃。"""
    ids = [f"f:{i}" for i in range(3)]
    records = [_rec(ids[i], f"t{i}", "stage1_icon") for i in range(3)]
    meta = _write_meta(tmp_path, records)
    emb_dir = _write_embeddings(tmp_path, ids)
    out = tmp_path / "cluster.jsonl"
    run_cluster(meta, emb_dir, out, k_per_bucket={"stage1_icon": 100}, random_seed=42)
    result = _read(out)
    assert len(result) == 3
    for rec in result:
        assert "fps_round" in rec["_meta"]


def test_cluster_npu_mock(tmp_path, monkeypatch):
    """use_npu=True 时调用 kmeans_npu，mock 返回确定结果，不依赖真实 NPU。"""
    n = 20
    ids = [f"f:{i}" for i in range(n)]
    records = [_rec(ids[i], f"text {i}", "stage1_icon") for i in range(n)]
    meta = _write_meta(tmp_path, records)
    emb_dir = _write_embeddings(tmp_path, ids, dim=16)
    out = tmp_path / "cluster.jsonl"

    def mock_kmeans(emb, k, seed, mb, **kwargs):
        k = min(k, len(emb))
        labels = np.arange(len(emb)) % k
        centroids = np.zeros((k, emb.shape[1]), dtype=np.float32)
        return labels.astype(np.int64), centroids

    import seed_selection.cluster as cluster_mod
    monkeypatch.setattr(cluster_mod, "_run_kmeans", mock_kmeans)

    run_cluster(meta, emb_dir, out,
                k_per_bucket={"stage1_icon": 5},
                random_seed=42,
                use_npu=True,
                npu_devices=["npu:0"])

    result = _read(out)
    assert len(result) == n
    for rec in result:
        assert "cluster_id" in rec["_meta"]
        assert "fps_round" in rec["_meta"]
