"""embed.py 测试：使用 dry_run=True（零向量），不需要真实模型。"""

import json
from pathlib import Path

import numpy as np
import pytest

from seed_selection.embed import load_all_embeddings, run_embed


def _write(tmp_path: Path, n: int = 5) -> Path:
    p = tmp_path / "in.jsonl"
    records = [
        {
            "instruction": f"Draw shape {i}.",
            "_meta": {"id": f"f:{i}", "domain": "stage1_icon", "source": "img2svg", "svg_len": 100},
        }
        for i in range(n)
    ]
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")
    return p


def test_embed_dry_run_writes_shards(tmp_path):
    in_path = _write(tmp_path, n=7)
    emb_dir = tmp_path / "embeddings"
    result = run_embed(
        input_path=in_path,
        output_dir=emb_dir,
        model_path="/nonexistent",
        dimension=256,
        batch_size=4,
        shard_size=4,
        dry_run=True,
    )
    assert result["total_records"] == 7
    assert result["shards_written"] == 2
    assert len(list(emb_dir.glob("shard_*.npz"))) == 2


def test_embed_dry_run_correct_shape(tmp_path):
    in_path = _write(tmp_path, n=5)
    emb_dir = tmp_path / "embeddings"
    run_embed(in_path, emb_dir, "/nonexistent", dimension=64,
              shard_size=10, dry_run=True)
    ids, embs = load_all_embeddings(emb_dir)
    assert len(ids) == 5
    assert embs.shape == (5, 64)


def test_embed_resume_skips_existing(tmp_path):
    in_path = _write(tmp_path, n=6)
    emb_dir = tmp_path / "embeddings"
    r1 = run_embed(in_path, emb_dir, "/nonexistent", shard_size=3, dry_run=True)
    assert r1["shards_written"] == 2
    r2 = run_embed(in_path, emb_dir, "/nonexistent", shard_size=3, dry_run=True)
    assert r2["shards_written"] == 0
    assert r2["shards_skipped"] == 2


def test_load_all_embeddings(tmp_path):
    in_path = _write(tmp_path, n=4)
    emb_dir = tmp_path / "embeddings"
    run_embed(in_path, emb_dir, "/nonexistent", dimension=32, shard_size=10, dry_run=True)
    ids, embs = load_all_embeddings(emb_dir)
    assert set(ids) == {f"f:{i}" for i in range(4)}
    assert embs.dtype == np.float32
