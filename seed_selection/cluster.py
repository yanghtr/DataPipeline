"""
Step 7 — cluster.py

对每个 domain bucket 分别运行 KMeans + FPS，
并将聚类/采样结果写回到 cluster_assignments.jsonl。

stage1_icon:
  KMeans(K=100K) 将 2.25M 条数据划分为 100K 个 cluster，
  再对每个 cluster 内部运行 Farthest Point Sampling（FPS），
  记录每条数据的 fps_round（FPS 轮次，从 1 开始）。
  下游 sample.py 按 fps_round 分层：
    round 1       → 高优 100K（每 cluster 第 1 条）
    round 2–3     → 中优 200K（每 cluster 第 2–3 条）
    round 4–7     → 低优 400K（每 cluster 第 4–7 条）

stage2_illustration:
  直接在全局 0.45M 条数据上运行 NPU 加速的 FPS，
  记录每条数据的 fps_rank（FPS 选择顺序，从 1 开始；未被选中则为 0）。
  下游 sample.py 按 fps_rank 直接分层：
    rank 1–50K    → 高优 50K
    rank 50K–150K → 中优 100K
    rank 150K–300K→ 低优 150K

输入：
  - svg_filtered_kept.jsonl（原始元数据，含 id / domain 等）
  - embeddings/ 目录（所有 shard_*.npz）
输出：
  - cluster_assignments.jsonl（在原记录基础上追加 cluster/fps 字段）
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


def _fps_within_clusters(
    emb_matrix: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """
    在每个 KMeans cluster 内部执行 Greedy Farthest Point Sampling（FPS）。

    算法：
      1. 将 emb_matrix 按 label 排序（argsort），获取各 cluster 的连续切片
      2. 对每个 cluster：
         a. anchor = 距质心最近的点（与原算法一致，作为 FPS 起点）
         b. min_dist[i] = ||emb[i] - anchor||²（初始化）
         c. 循环 FPS：nxt = argmax(min_dist)，更新 min_dist = min(min_dist, ||emb - emb[nxt]||²)
         d. 按选择顺序赋 fps_round = 1, 2, 3, ...

    返回：
        fps_rounds: (N,) int32 数组，与 emb_matrix / records 行对齐，
                    值为 1-based FPS 轮次（cluster 内部顺序）。

    时间复杂度：O(Σ n_c²)，对 K=100K、avg n_c=22 约为 Σ 22² × 100K = 48M ops，
               使用 numpy 向量化，CPU 上约 10–30 秒。
    """
    n = len(emb_matrix)
    fps_rounds = np.zeros(n, dtype=np.int32)

    # 按 label 排序，获取连续的 cluster 切片
    sort_idx = np.argsort(labels, kind="stable")
    sorted_labels = labels[sort_idx]

    # 找每个 cluster 的起始位置
    unique_labels, cluster_starts = np.unique(sorted_labels, return_index=True)

    for ci, (cid, start) in enumerate(zip(unique_labels, cluster_starts)):
        end = cluster_starts[ci + 1] if ci + 1 < len(cluster_starts) else n
        global_idx = sort_idx[start:end]   # 在原始 emb_matrix 中的行索引
        cluster_embs = emb_matrix[global_idx]  # (n_c, D)
        n_c = len(cluster_embs)

        if n_c == 1:
            fps_rounds[global_idx[0]] = 1
            continue

        centroid = centroids[cid]  # (D,)

        # anchor = 距质心最近的点
        diff = cluster_embs - centroid  # (n_c, D)
        dists_to_centroid = np.einsum("ij,ij->i", diff, diff)  # (n_c,)
        anchor_local = int(np.argmin(dists_to_centroid))

        # 初始化 min_dist：各点到 anchor 的平方距离
        anchor_emb = cluster_embs[anchor_local]  # (D,)
        diff0 = cluster_embs - anchor_emb        # (n_c, D)
        min_dist = np.einsum("ij,ij->i", diff0, diff0)  # (n_c,)
        min_dist[anchor_local] = -1.0  # 标记已选择

        selection_order = np.empty(n_c, dtype=np.int32)
        selection_order[0] = anchor_local

        for step in range(1, n_c):
            nxt = int(np.argmax(min_dist))
            selection_order[step] = nxt
            # 更新 min_dist
            nxt_emb = cluster_embs[nxt]
            diff_nxt = cluster_embs - nxt_emb    # (n_c, D)
            new_d = np.einsum("ij,ij->i", diff_nxt, diff_nxt)
            min_dist = np.minimum(min_dist, new_d)
            min_dist[nxt] = -1.0  # 标记已选择

        # 赋 fps_round（1-based）
        for round_idx, local_idx in enumerate(selection_order):
            fps_rounds[global_idx[local_idx]] = round_idx + 1

    return fps_rounds


def _fps_global_npu(
    emb_matrix: np.ndarray,
    n_select: int,
    device: str = "npu:0",
    chunk_size: int = 50_000,
) -> np.ndarray:
    """
    NPU/GPU 加速的全局 Greedy FPS。

    选出 n_select 个点，返回其在 emb_matrix 中的下标数组（长度 n_select），
    按 FPS 选择顺序排列（第 0 个 = 起始点，第 1 个 = 第一步选的点，…）。

    内存设计：
      - X 常驻 NPU：N×D×4B
      - norms 常驻 NPU：N×4B
      - min_dist 常驻 NPU：N×4B
      - 每步只需一次 GEMV（X @ center）：N×4B 读 + N×4B 写，无 (N,K,D) 展开

    显存峰值（D=256, N=450K, FP32）：
      X = 450K×256×4B ≈ 440MB；norms + min_dist ≈ 3.5MB → 合计 < 500MB，安全。

    速度估算（Ascend 910B）：
      每步 1 次 GEMV（N=450K）≈ 0.5ms → n_select=300K 约 2.5 分钟。
    """
    try:
        import torch
        try:
            import torch_npu  # noqa: F401
        except ImportError:
            if "npu" in device:
                raise RuntimeError(
                    "torch_npu not found. Install with: pip install torch torch_npu"
                )
    except ImportError:
        raise ImportError(
            "FPS global requires PyTorch. "
            "For Ascend NPU: pip install torch torch_npu. "
            "For CUDA: pip install torch."
        )

    import torch

    n, d = emb_matrix.shape
    n_select = min(n_select, n)

    logger.info(
        f"[fps_global_npu] N={n:,}, n_select={n_select:,}, device={device}"
    )

    X = torch.from_numpy(emb_matrix.astype(np.float32)).to(device)  # (N, D)
    norms = (X * X).sum(dim=1)  # (N,) 各点 ||x||²，常驻 NPU

    # 起始点：距全局质心最近的点
    mean_vec = X.mean(dim=0)  # (D,)
    diff_mean = X - mean_vec.unsqueeze(0)  # (N, D)
    start_idx = int((diff_mean * diff_mean).sum(dim=1).argmin().item())

    selected = [start_idx]
    # min_dist[i] = 当前已选集合中，点 i 到最近已选点的平方距离
    center = X[start_idx]  # (D,)
    min_dist = norms - 2.0 * (X @ center) + float((center * center).sum().item())
    min_dist[start_idx] = -1.0  # 标记已选择

    log_interval = max(1, n_select // 20)  # 每 5% 日志一次

    for step in range(1, n_select):
        nxt = int(min_dist.argmax().item())
        selected.append(nxt)
        if step % log_interval == 0:
            logger.info(f"  [fps_global_npu] step {step:,}/{n_select:,}")

        center = X[nxt]
        new_d = norms - 2.0 * (X @ center) + float((center * center).sum().item())
        min_dist = torch.minimum(min_dist, new_d)
        min_dist[nxt] = -1.0

    logger.info(f"[fps_global_npu] 完成，选出 {len(selected):,} 个点")
    return np.array(selected, dtype=np.int64)


def _fps_global_cpu(
    emb_matrix: np.ndarray,
    n_select: int,
) -> np.ndarray:
    """
    CPU 回退版全局 FPS（警告：速度慢，建议仅在小数据集或测试时使用）。

    使用 BLAS GEMV（numpy dot）计算距离，无需 PyTorch。
    N=450K, n_select=300K 约需 30–90 分钟（取决于 CPU 核心 BLAS 实现）。
    """
    logger.warning(
        "[fps_global_cpu] CPU FPS 速度极慢，N=450K×300K steps 约 30-90 分钟。"
        "强烈建议使用 NPU/GPU 后端。"
    )

    n, d = emb_matrix.shape
    n_select = min(n_select, n)
    X = emb_matrix.astype(np.float32)
    norms = (X * X).sum(axis=1)  # (N,)

    # 起始点：距全局质心最近
    mean_vec = X.mean(axis=0)
    diff_mean = X - mean_vec
    start_idx = int(np.argmin((diff_mean * diff_mean).sum(axis=1)))

    selected = [start_idx]
    center = X[start_idx]
    min_dist = norms - 2.0 * (X @ center) + float((center * center).sum())
    min_dist[start_idx] = -1.0

    log_interval = max(1, n_select // 20)

    for step in range(1, n_select):
        nxt = int(np.argmax(min_dist))
        selected.append(nxt)
        if step % log_interval == 0:
            logger.info(f"  [fps_global_cpu] step {step:,}/{n_select:,}")

        center = X[nxt]
        new_d = norms - 2.0 * (X @ center) + float((center * center).sum())
        min_dist = np.minimum(min_dist, new_d)
        min_dist[nxt] = -1.0

    return np.array(selected, dtype=np.int64)


def _fps_global(
    emb_matrix: np.ndarray,
    n_select: int,
    use_npu: bool = False,
    npu_device: str = "npu:0",
    chunk_size: int = 50_000,
) -> np.ndarray:
    """
    分派到 NPU 或 CPU 的全局 FPS 入口。

    返回 fps_indices: (n_select,) int64，按 FPS 选择顺序排列。
    """
    if use_npu:
        return _fps_global_npu(emb_matrix, n_select, device=npu_device, chunk_size=chunk_size)
    else:
        return _fps_global_cpu(emb_matrix, n_select)


def _run_kmeans(
    embeddings: np.ndarray,
    k: int,
    random_seed: int,
    minibatch_size: int,
    n_init: int = 3,
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
            n_init=n_init,
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
            n_init=n_init,
            max_iter=100,
            random_seed=random_seed,
        )
        return labels, centroids

    logger.info(f"  MiniBatchKMeans K={k}, n={len(embeddings):,}")
    km = MiniBatchKMeans(
        n_clusters=k,
        batch_size=minibatch_size,
        random_state=random_seed,
        n_init=n_init,
        max_iter=100,
        verbose=0,
    )
    labels = km.fit_predict(embeddings)
    centroids = km.cluster_centers_
    return labels, centroids


def _cluster_domain_worker(args: tuple) -> tuple[str, int, int]:
    """
    Worker 函数：对单个 domain 运行 KMeans + FPS，写出带 cluster/fps 字段的临时 JSONL。

    stage1_icon:
      KMeans(K=100K) → 类内 FPS → 写 cluster_id / cluster_size /
      distance_to_centroid / fps_round（FPS 轮次，1-based）

    stage2_illustration:
      直接全局 FPS(n_select=300K) → 写 fps_rank（FPS 选择顺序，1-based；
      未被选中则为 0）。注意：stage2 不运行 KMeans，cluster_id 固定为 0。

    返回: (domain, n_records, actual_k)
    """
    _limit_blas_threads()

    (domain, records_json, emb_bytes, emb_shape,
     k, fps_n_select,
     random_seed, minibatch_size, n_init,
     use_npu, npu_device, npu_chunk_size,
     use_faiss,
     tmp_path_str) = args

    records = [json.loads(r) for r in records_json]
    emb_matrix = np.frombuffer(emb_bytes, dtype=np.float32).reshape(emb_shape)
    n = len(records)

    # -----------------------------------------------------------------------
    # stage2_illustration：直接全局 FPS，不运行 KMeans
    # -----------------------------------------------------------------------
    if domain == "stage2_illustration":
        logger.info(
            f"[cluster] {domain}: 全局 FPS, n_select={fps_n_select:,}, n={n:,}"
        )
        fps_indices = _fps_global(
            emb_matrix, fps_n_select,
            use_npu=use_npu, npu_device=npu_device, chunk_size=npu_chunk_size,
        )
        # fps_rank[i] = FPS 中选中的顺序（1-based），未选中为 0
        fps_rank = np.zeros(n, dtype=np.int32)
        for rank, idx in enumerate(fps_indices):
            fps_rank[idx] = rank + 1

        with open(tmp_path_str, "w", encoding="utf-8") as fout:
            for rec, emb, rank in zip(records, emb_matrix, fps_rank):
                update_meta(rec,
                    cluster_id=0,
                    cluster_size=n,
                    distance_to_centroid=0.0,
                    fps_rank=int(rank),
                    bucket_key=domain,
                )
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

        return domain, n, 1

    # -----------------------------------------------------------------------
    # stage1_icon（及其他 bucket）：KMeans → 类内 FPS
    # -----------------------------------------------------------------------
    labels, centroids = _run_kmeans(
        emb_matrix, k, random_seed, minibatch_size,
        n_init=n_init,
        use_npu=use_npu, npu_device=npu_device, npu_chunk_size=npu_chunk_size,
        use_faiss=use_faiss,
    )
    actual_k = int(labels.max()) + 1
    cluster_sizes = Counter(labels.tolist())

    logger.info(
        f"[cluster] {domain}: KMeans 完成 actual_k={actual_k:,}，"
        f"开始类内 FPS ..."
    )
    fps_rounds = _fps_within_clusters(emb_matrix, labels, centroids)

    with open(tmp_path_str, "w", encoding="utf-8") as fout:
        for rec, label, emb, fps_r in zip(records, labels, emb_matrix, fps_rounds):
            centroid = centroids[label]
            dist = float(np.linalg.norm(emb - centroid))
            update_meta(rec,
                cluster_id=int(label),
                cluster_size=cluster_sizes[int(label)],
                distance_to_centroid=round(dist, 6),
                fps_round=int(fps_r),
                bucket_key=domain,
            )
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return domain, n, actual_k


def run_cluster(
    meta_path: Path,
    embed_dir: Path,
    output_path: Path,
    k_per_bucket: dict[str, int],
    fps_n_select_per_bucket: dict[str, int] | None = None,
    random_seed: int = 42,
    minibatch_size: int = 50_000,
    n_init: int = 1,
    num_workers: int = 1,
    use_npu: bool = False,
    npu_devices: list[str] | None = None,
    npu_chunk_size: int = 40_000,
    use_faiss: bool = False,
) -> ClusterStats:
    """
    npu_devices: 每个 bucket worker 按顺序 round-robin 分配的 NPU 设备列表。
      - 单卡：["npu:0"]
      - 双卡：["npu:0", "npu:1"]（2 个 bucket 各用 1 张卡，并行）
      - 8 卡：["npu:0", ..., "npu:7"]（2 个 bucket 分别用 npu:0 / npu:1，其余空闲）

    fps_n_select_per_bucket: 每个 bucket 的全局 FPS 选取数（仅 stage2_illustration 使用）。
      如果不指定，默认选取整个 bucket 的所有数据（即不截断）。

    n_init: KMeans 初始化次数（K=100K 时推荐 1，节省时间；K 较小时可设 3）。
    npu_chunk_size: K=100K 时建议设为 40000（峰值 16GB，安全）；
                    K=12000 时可设 50000（峰值 2.4GB）。
    """
    npu_devices = npu_devices or ["npu:0"]
    fps_n_select_per_bucket = fps_n_select_per_bucket or {}
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

    # 3. 对每个 domain 单独处理，写出结果
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        worker_args = []
        domain_order: list[str] = []

        for domain, records in domain_records.items():
            if not records:
                continue
            k = k_per_bucket.get(domain, 1000)
            n_select = fps_n_select_per_bucket.get(domain, len(records))
            stats.record_counts[domain] = len(records)
            logger.info(
                f"[cluster] {domain}: {len(records):,} 条 → K={k}, fps_n_select={n_select:,}"
            )

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
            bucket_idx = len(worker_args)
            assigned_device = npu_devices[bucket_idx % len(npu_devices)]
            if use_npu:
                logger.info(f"[cluster] {domain} → {assigned_device}")
            worker_args.append((
                domain,
                [json.dumps(r, ensure_ascii=False) for r in valid_records],
                emb_matrix.tobytes(),
                emb_matrix.shape,
                k, n_select,
                random_seed, minibatch_size, n_init,
                use_npu, assigned_device, npu_chunk_size,
                use_faiss,
                tmp_path,
            ))
            domain_order.append(domain)

        # 当 use_npu=True 时，检查各 bucket 是否共享同一张卡：
        # 多个进程并发操作同一 NPU，其显存由硬件层面共享。
        # 若所有 bucket 分配到相同 device，强制串行避免并发 OOM。
        assigned_devices = [arg[11] for arg in worker_args]  # npu_device 字段
        npu_conflict = (
            use_npu
            and len(worker_args) > 1
            and len(set(assigned_devices)) < len(assigned_devices)
        )
        if npu_conflict:
            logger.warning(
                f"[cluster] 检测到多个 bucket 共享同一 NPU 设备 "
                f"({set(assigned_devices)})，强制串行执行以防 OOM。"
                f"若要并行，请在 npu_devices 中提供足够多的独立设备。"
            )

        if num_workers > 1 and len(worker_args) > 1 and not npu_conflict:
            futures = {}
            with ProcessPoolExecutor(
                max_workers=min(num_workers, len(worker_args)),
                initializer=_limit_blas_threads,
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
