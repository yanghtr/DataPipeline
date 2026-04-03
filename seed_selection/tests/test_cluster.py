"""cluster.py 测试：使用小规模合成 embeddings，不依赖真实模型。"""

import json
from pathlib import Path

import numpy as np
import pytest

from seed_selection.cluster import run_cluster


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


def test_cluster_assigns_all_records(tmp_path):
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
        meta_out = rec["_meta"]
        assert "cluster_id" in meta_out
        assert "cluster_size" in meta_out
        assert "distance_to_centroid" in meta_out
        assert "bucket_key" in meta_out


def test_cluster_multi_domain(tmp_path):
    """两桶：stage1_icon + stage2_illustration（stage2_icon 已被 exact dedup 覆盖）。"""
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
                k_per_bucket={"stage1_icon": 3, "stage2_illustration": 2},
                random_seed=42)

    result = _read(out)
    assert len(result) == 15
    domains = {r["_meta"]["bucket_key"] for r in result}
    assert "stage1_icon" in domains
    assert "stage2_illustration" in domains
    # stage2_icon 不应出现（已被合并到 stage1_icon）
    assert "stage2_icon" not in domains


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


def test_cluster_npu_mock(tmp_path, monkeypatch):
    """use_npu=True 时调用 kmeans_npu，mock 返回确定结果，不依赖真实 NPU。"""
    n = 20
    ids = [f"f:{i}" for i in range(n)]
    records = [_rec(ids[i], f"text {i}", "stage1_icon") for i in range(n)]
    meta = _write_meta(tmp_path, records)
    emb_dir = _write_embeddings(tmp_path, ids, dim=16)
    out = tmp_path / "cluster.jsonl"

    import numpy as np

    def mock_kmeans_npu(embeddings, k, **kwargs):
        k = min(k, len(embeddings))
        labels = np.arange(len(embeddings)) % k
        centroids = np.zeros((k, embeddings.shape[1]), dtype=np.float32)
        return labels.astype(np.int64), centroids

    import seed_selection.cluster as cluster_mod
    monkeypatch.setattr(cluster_mod, "_run_kmeans",
                        lambda emb, k, seed, mb, use_npu, npu_device, npu_chunk_size:
                        mock_kmeans_npu(emb, k))

    run_cluster(meta, emb_dir, out,
                k_per_bucket={"stage1_icon": 5},
                random_seed=42,
                use_npu=True,
                npu_devices=["npu:0"])

    result = _read(out)
    assert len(result) == n
    for rec in result:
        assert "cluster_id" in rec["_meta"]
