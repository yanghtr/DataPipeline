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


def test_cluster_assigns_all_records(tmp_path):
    n = 20
    ids = [f"f:{i}" for i in range(n)]
    records = [
        {"id": ids[i], "instruction": f"text {i}", "svg_len": 100,
         "domain": "stage1_icon", "source": "img2svg"}
        for i in range(n)
    ]
    meta = _write_meta(tmp_path, records)
    emb_dir = _write_embeddings(tmp_path, ids)
    out = tmp_path / "cluster.jsonl"

    run_cluster(meta, emb_dir, out,
                k_per_bucket={"stage1_icon": 5}, random_seed=42)

    result = _read(out)
    assert len(result) == n
    for rec in result:
        assert "cluster_id" in rec
        assert "cluster_size" in rec
        assert "distance_to_centroid" in rec
        assert "bucket_key" in rec


def test_cluster_multi_domain(tmp_path):
    ids_icon = [f"icon:{i}" for i in range(10)]
    ids_ill  = [f"ill:{i}"  for i in range(5)]
    all_ids  = ids_icon + ids_ill

    records = (
        [{"id": ids_icon[i], "instruction": f"icon {i}", "svg_len": 100,
          "domain": "stage1_icon", "source": "img2svg"} for i in range(10)]
        +
        [{"id": ids_ill[i],  "instruction": f"ill {i}",  "svg_len": 5000,
          "domain": "stage2_illustration", "source": "img2svg"} for i in range(5)]
    )
    meta = _write_meta(tmp_path, records)
    emb_dir = _write_embeddings(tmp_path, all_ids)
    out = tmp_path / "cluster.jsonl"

    run_cluster(meta, emb_dir, out,
                k_per_bucket={"stage1_icon": 3, "stage2_illustration": 2},
                random_seed=42)

    result = _read(out)
    assert len(result) == 15
    domains = {r["bucket_key"] for r in result}
    assert "stage1_icon" in domains
    assert "stage2_illustration" in domains


def test_cluster_k_exceeds_samples(tmp_path):
    """K > 样本数时应自动降低 K，不崩溃。"""
    ids = [f"f:{i}" for i in range(3)]
    records = [{"id": ids[i], "instruction": f"t{i}", "svg_len": 100,
                "domain": "stage1_icon", "source": "img2svg"} for i in range(3)]
    meta = _write_meta(tmp_path, records)
    emb_dir = _write_embeddings(tmp_path, ids)
    out = tmp_path / "cluster.jsonl"
    run_cluster(meta, emb_dir, out, k_per_bucket={"stage1_icon": 100}, random_seed=42)
    result = _read(out)
    assert len(result) == 3
