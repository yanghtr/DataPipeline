"""
Step 7 — cluster.py

对每个 domain bucket 分别运行 MiniBatchKMeans，
并将聚类结果（cluster_id, cluster_size, distance_to_centroid）
写回到 cluster_assignments.jsonl。

输入：
  - svg_filtered_kept.jsonl（原始元数据，含 id / domain 等）
  - embeddings/ 目录（所有 shard_*.npz）
输出：
  - cluster_assignments.jsonl（在原记录基础上追加 cluster 信息）
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from sklearn.cluster import MiniBatchKMeans

from .embed import load_all_embeddings
from .io_utils import DOMAINS, read_jsonl


@dataclass
class ClusterStats:
    cluster_counts: dict[str, int] = field(default_factory=dict)     # domain → K
    record_counts: dict[str, int] = field(default_factory=dict)      # domain → n_records

    def report(self) -> str:
        lines = []
        for domain in DOMAINS:
            n = self.record_counts.get(domain, 0)
            k = self.cluster_counts.get(domain, 0)
            avg = n // k if k else 0
            lines.append(f"  {domain}: {n:,} 条, K={k:,}, 均值 {avg}/cluster")
        return "[cluster] 结果:\n" + "\n".join(lines)


def _run_kmeans(
    embeddings: np.ndarray,
    k: int,
    random_seed: int,
    minibatch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """返回 (labels, centroids)。"""
    k = min(k, len(embeddings))   # K 不得超过样本数
    logger.info(f"  MiniBatchKMeans K={k}, n={len(embeddings):,}")
    km = MiniBatchKMeans(
        n_clusters=k,
        batch_size=minibatch_size,
        random_state=random_seed,
        n_init=3,
        max_iter=100,
        verbose=0,
    )
    labels = km.fit_predict(embeddings)
    centroids = km.cluster_centers_
    return labels, centroids


def run_cluster(
    meta_path: Path,
    embed_dir: Path,
    output_path: Path,
    k_per_bucket: dict[str, int],
    random_seed: int = 42,
    minibatch_size: int = 50_000,
) -> ClusterStats:
    stats = ClusterStats()

    # 1. 加载所有 embedding（id → embedding）
    logger.info("[cluster] 加载 embeddings ...")
    all_ids, all_embs = load_all_embeddings(embed_dir)
    id_to_idx: dict[str, int] = {id_: i for i, id_ in enumerate(all_ids)}
    logger.info(f"[cluster] 共 {len(all_ids):,} 条 embedding")

    # 2. 读取元数据，按 domain 分组
    domain_records: dict[str, list[dict]] = {d: [] for d in DOMAINS}
    for rec in read_jsonl(meta_path):
        domain = rec.get("domain", "stage1_icon")
        domain_records.setdefault(domain, []).append(rec)

    # 3. 对每个 domain 单独聚类，写出结果
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for domain, records in domain_records.items():
            if not records:
                continue
            k = k_per_bucket.get(domain, 1000)
            stats.record_counts[domain] = len(records)
            logger.info(f"[cluster] {domain}: {len(records):,} 条 → K={k}")

            # 收集该 domain 的 embedding
            valid_records, valid_embs = [], []
            missing = 0
            for rec in records:
                idx = id_to_idx.get(rec["id"])
                if idx is None:
                    missing += 1
                    continue
                valid_records.append(rec)
                valid_embs.append(all_embs[idx])

            if missing:
                logger.warning(f"[cluster] {domain}: {missing} 条在 embedding 中找不到 ID")

            if not valid_records:
                continue

            emb_matrix = np.vstack(valid_embs)
            labels, centroids = _run_kmeans(emb_matrix, k, random_seed, minibatch_size)

            actual_k = int(labels.max()) + 1
            stats.cluster_counts[domain] = actual_k

            # 计算 cluster_size
            cluster_sizes = Counter(labels.tolist())

            for rec, label, emb in zip(valid_records, labels, valid_embs):
                centroid = centroids[label]
                dist = float(np.linalg.norm(emb - centroid))
                rec["cluster_id"] = int(label)
                rec["cluster_size"] = cluster_sizes[int(label)]
                rec["distance_to_centroid"] = round(dist, 6)
                rec["bucket_key"] = domain
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(stats.report())
    return stats
