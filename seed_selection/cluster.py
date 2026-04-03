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
import os
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from sklearn.cluster import MiniBatchKMeans

from .embed import load_all_embeddings
from .io_utils import DOMAINS, read_jsonl, update_meta


def _limit_blas_threads() -> None:
    """
    限制 BLAS/OpenMP 线程数为 1，防止在多进程环境下 OpenBLAS 尝试创建
    超过编译上限的线程（在 192 核等大机器上会触发 double free / BrokenProcessPool）。

    必须在任何 numpy/sklearn 调用之前设置环境变量；同时通过 threadpoolctl
    在运行时再次限制（scikit-learn 依赖，必然已安装）。
    """
    import os
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
    except ImportError:
        pass


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
    use_npu: bool = False,
    npu_device: str = "npu:0",
    npu_chunk_size: int = 50_000,
    use_faiss: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    返回 (labels, centroids)。后端优先级：use_npu > use_faiss > MiniBatchKMeans。

    use_npu:   torch_npu Lloyd's（精确，NPU/GPU 加速）
    use_faiss: faiss-cpu Lloyd's（精确，CPU BLAS，5–15× 快于 MiniBatch）
    默认:      sklearn MiniBatchKMeans（近似，纯 CPU，无额外依赖）
    """
    k = min(k, len(embeddings))   # K 不得超过样本数

    if use_npu:
        logger.info(f"  KMeans(NPU) K={k}, n={len(embeddings):,}, device={npu_device}")
        from .kmeans_npu import kmeans_npu
        labels, centroids = kmeans_npu(
            embeddings, k,
            device=npu_device,
            n_init=3,
            max_iter=100,
            random_seed=random_seed,
            chunk_size=npu_chunk_size,
        )
        return labels, centroids

    if use_faiss:
        logger.info(f"  KMeans(faiss-cpu) K={k}, n={len(embeddings):,}")
        from .kmeans_faiss import kmeans_faiss
        labels, centroids = kmeans_faiss(
            embeddings, k,
            n_init=3,
            max_iter=100,
            random_seed=random_seed,
        )
        return labels, centroids

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


def _cluster_domain_worker(args: tuple) -> tuple[str, int, int]:
    """
    Worker 函数：对单个 domain 运行 KMeans，写出带 cluster 字段的临时 JSONL。
    返回: (domain, n_records, actual_k)
    """
    _limit_blas_threads()   # 防止 OpenBLAS 在大机器上超过编译线程上限

    (domain, records_json, emb_bytes, emb_shape,
     k, random_seed, minibatch_size,
     use_npu, npu_device, npu_chunk_size,
     use_faiss,
     tmp_path_str) = args

    records = [json.loads(r) for r in records_json]
    emb_matrix = np.frombuffer(emb_bytes, dtype=np.float32).reshape(emb_shape)

    labels, centroids = _run_kmeans(
        emb_matrix, k, random_seed, minibatch_size,
        use_npu=use_npu, npu_device=npu_device, npu_chunk_size=npu_chunk_size,
        use_faiss=use_faiss,
    )
    actual_k = int(labels.max()) + 1
    cluster_sizes = Counter(labels.tolist())

    with open(tmp_path_str, "w", encoding="utf-8") as fout:
        for rec, label, emb in zip(records, labels, emb_matrix):
            centroid = centroids[label]
            dist = float(np.linalg.norm(emb - centroid))
            update_meta(rec,
                cluster_id=int(label),
                cluster_size=cluster_sizes[int(label)],
                distance_to_centroid=round(dist, 6),
                bucket_key=domain,
            )
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return domain, len(records), actual_k


def run_cluster(
    meta_path: Path,
    embed_dir: Path,
    output_path: Path,
    k_per_bucket: dict[str, int],
    random_seed: int = 42,
    minibatch_size: int = 50_000,
    num_workers: int = 1,
    use_npu: bool = False,
    npu_devices: list[str] | None = None,
    npu_chunk_size: int = 50_000,
    use_faiss: bool = False,
) -> ClusterStats:
    """
    npu_devices: 每个 bucket worker 按顺序 round-robin 分配的 NPU 设备列表。
      - 单卡：["npu:0"]
      - 双卡：["npu:0", "npu:1"]（2 个 bucket 各用 1 张卡，并行）
      - 8 卡：["npu:0", ..., "npu:7"]（2 个 bucket 分别用 npu:0 / npu:1，其余空闲）
    """
    npu_devices = npu_devices or ["npu:0"]
    stats = ClusterStats()

    # 1. 加载所有 embedding（id → embedding）
    logger.info("[cluster] 加载 embeddings ...")
    all_ids, all_embs = load_all_embeddings(embed_dir)
    id_to_idx: dict[str, int] = {id_: i for i, id_ in enumerate(all_ids)}
    logger.info(f"[cluster] 共 {len(all_ids):,} 条 embedding")

    # 2. 读取元数据，按 domain 分组
    domain_records: dict[str, list[dict]] = {d: [] for d in DOMAINS}
    for rec in read_jsonl(meta_path):
        domain = rec.get("_meta", {}).get("domain", "stage1_icon")
        domain_records.setdefault(domain, []).append(rec)

    # 3. 对每个 domain 单独聚类，写出结果
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # 准备 worker 参数（预先收集 embeddings，避免在 worker 中重复加载）
        worker_args = []
        domain_order: list[str] = []

        for domain, records in domain_records.items():
            if not records:
                continue
            k = k_per_bucket.get(domain, 1000)
            stats.record_counts[domain] = len(records)
            logger.info(f"[cluster] {domain}: {len(records):,} 条 → K={k}")

            valid_records, valid_embs = [], []
            missing = 0
            for rec in records:
                idx = id_to_idx.get(rec.get("_meta", {}).get("id", ""))
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
            tmp_path = os.path.join(tmp_dir, f"cluster_{domain}.jsonl")
            # 按 bucket 顺序 round-robin 分配 NPU 设备（多卡并行）
            bucket_idx = len(worker_args)
            assigned_device = npu_devices[bucket_idx % len(npu_devices)]
            if use_npu:
                logger.info(f"[cluster] {domain} → {assigned_device}")
            worker_args.append((
                domain,
                [json.dumps(r, ensure_ascii=False) for r in valid_records],
                emb_matrix.tobytes(),
                emb_matrix.shape,
                k, random_seed, minibatch_size,
                use_npu, assigned_device, npu_chunk_size,
                use_faiss,
                tmp_path,
            ))
            domain_order.append(domain)

        if num_workers > 1 and len(worker_args) > 1:
            futures = {}
            with ProcessPoolExecutor(
                max_workers=min(num_workers, len(worker_args)),
                initializer=_limit_blas_threads,  # 防止 OpenBLAS 超线程上限崩溃
            ) as exe:
                for arg in worker_args:
                    futures[arg[0]] = exe.submit(_cluster_domain_worker, arg)
            results = {dom: fut.result() for dom, fut in futures.items()}
        else:
            results = {}
            for arg in worker_args:
                r = _cluster_domain_worker(arg)
                results[r[0]] = r

        # 按顺序 merge 输出
        with output_path.open("w", encoding="utf-8") as fout:
            for domain in domain_order:
                if domain not in results:
                    continue
                dom, n_records, actual_k = results[domain]
                stats.cluster_counts[domain] = actual_k
                tmp_path = os.path.join(tmp_dir, f"cluster_{domain}.jsonl")
                if os.path.exists(tmp_path):
                    with open(tmp_path, encoding="utf-8") as fin:
                        fout.write(fin.read())

    logger.info(stats.report())
    return stats
