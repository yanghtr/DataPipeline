"""端到端流水线测试（extract → sample），使用 dry_run=True。"""

import json
from pathlib import Path

import numpy as np
import pytest

from seed_selection.config import PipelineConfig, EmbedConfig, NearDedupConfig, ClusterConfig, SamplingConfig
from seed_selection.extract import run_extract
from seed_selection.clean import run_clean
from seed_selection.dedup_exact import run_dedup_exact
from seed_selection.dedup_near import run_dedup_near
from seed_selection.svg_filter import run_svg_filter
from seed_selection.embed import run_embed
from seed_selection.cluster import run_cluster
from seed_selection.sample import run_sample


THRESHOLDS = {"stage1_icon": 0.8, "stage2_illustration": 0.7}


def test_full_pipeline(tmp_path, all_input_paths):
    root = tmp_path / "out"
    root.mkdir()

    # 1. extract
    raw = root / "raw.jsonl"
    run_extract(all_input_paths, raw)

    # 2. clean
    cleaned = root / "cleaned.jsonl"
    run_clean(raw, cleaned)

    # 3. exact dedup
    exact_kept = root / "exact_kept.jsonl"
    stats_exact = run_dedup_exact(cleaned, exact_kept)
    assert stats_exact.kept > 0

    # 4. near dedup
    near_kept = root / "near_kept.jsonl"
    run_dedup_near(exact_kept, near_kept, thresholds=THRESHOLDS)

    # 5. svg filter
    filtered = root / "filtered.jsonl"
    run_svg_filter(near_kept, filtered, bottom_pct=0.10)

    # 6. embed (dry_run → zero vectors)
    emb_dir = root / "embeddings"
    run_embed(filtered, emb_dir, model_path="/nonexistent",
              dimension=32, shard_size=50, dry_run=True)

    # 7. cluster
    cluster_out = root / "clusters.jsonl"
    run_cluster(
        meta_path=filtered,
        embed_dir=emb_dir,
        output_path=cluster_out,
        k_per_bucket={"stage1_icon": 2, "stage2_illustration": 2},
        random_seed=42,
    )

    # 8. sample
    records = [json.loads(l) for l in cluster_out.read_text().splitlines() if l.strip()]
    n = len(records)
    pool_size = max(1, n // 2)
    hp_size = max(1, n // 4)

    stats_sample = run_sample(cluster_out, root,
                               total_pool_size=pool_size,
                               high_priority_size=hp_size,
                               random_seed=42)

    assert stats_sample.pool_1000k == stats_sample.anneal + stats_sample.high_priority
    assert (root / "pool_1000k.jsonl").exists()
    assert (root / "anneal_pool.jsonl").exists()
    assert (root / "high_priority_pool.jsonl").exists()

    # Verify _meta structure in final outputs
    valid_domains = {"stage1_icon", "stage2_illustration"}
    for line in (root / "anneal_pool.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        assert "instruction" in rec
        assert "_meta" in rec
        meta = rec["_meta"]
        assert "id" in meta
        assert "cluster_id" in meta
        assert "distance_to_centroid" in meta
        assert "bucket_key" in meta
        # stage2_icon 应已被 exact dedup 覆盖，不出现在输出中
        assert meta.get("domain") in valid_domains, f"unexpected domain: {meta.get('domain')}"
