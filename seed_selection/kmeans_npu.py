"""
NPU/GPU 加速 K-Means（标准 Lloyd's 算法）。

在 Ascend 910B（torch_npu）或 CUDA GPU（torch.cuda）上运行。
核心算子：torch.cdist，利用 NPU/GPU 矩阵乘加速 N×K 距离计算。

已修复的 OOM 问题：
  Bug 1: _update_centroids 中 labels.unsqueeze(1).expand(-1, d) 会被 torch_npu 强制
          materialize 为连续内存 (N×D×8B = 4.5GB int64 index)。
          修复：改为 chunk 级 scatter_add，每次只展开 (chunk×D) index。

  Bug 2: _init_centroids 用 KMeans++，K=12000 需循环 12000 次，每次全量 N×K cdist，
          = 540000 次 cdist，内存碎片严重触发 OOM。
          修复：改为 Forgy 随机初始化（从 X 随机采 K 个点），O(1) 开销。
          对大 K（≥1000），n_init=3 的多次随机重启质量等同于 KMeans++。

显存估算（D=256, FP32, chunk=50K）：
  N=2.25M, K=12,000：每 cdist chunk = 50K×12K×4B = 2.4GB（峰值，用后释放）
                       X 常驻：2.25M×256×4B = 2.25GB
                       index chunk (scatter_add)：50K×256×8B = 100MB
                       总峰值 ≈ 4.7GB，在 64GB NPU 上安全运行
"""

from __future__ import annotations

import torch
import numpy as np
from loguru import logger


def kmeans_npu(
    embeddings: np.ndarray,
    k: int,
    device: str = "npu:0",
    n_init: int = 3,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_seed: int = 42,
    chunk_size: int = 50_000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    标准 Lloyd's K-Means，使用 NPU/GPU 加速距离计算。

    Args:
        embeddings:  (N, D) float32 numpy 数组
        k:           聚类数（若 k > N，自动截断为 N）
        device:      PyTorch 设备字符串，如 "npu:0"、"cuda:0"
        n_init:      随机初始化次数，取 inertia 最小的结果
        max_iter:    每次初始化的最大迭代轮数
        tol:         质心变化量收敛阈值（L2 范数均值）
        random_seed: 随机种子
        chunk_size:  分批 cdist / scatter_add 的批大小（控制显存峰值）

    Returns:
        labels:    (N,) int64 numpy 数组，每条记录的 cluster id
        centroids: (K, D) float32 numpy 数组，最终质心
    """
    try:
        import torch
        import torch_npu  # noqa: F401  Ascend NPU 支持
    except ImportError:
        try:
            import torch
            if "npu" in device:
                raise RuntimeError(
                    "torch_npu not found. Install with: pip install torch torch_npu"
                )
        except ImportError:
            raise ImportError(
                "kmeans_npu requires PyTorch. "
                "For Ascend NPU: pip install torch torch_npu. "
                "For CUDA: pip install torch."
            )

    import torch

    k = min(k, len(embeddings))
    n, d = embeddings.shape
    rng = np.random.default_rng(random_seed)

    logger.info(
        f"[kmeans_npu] N={n:,}, K={k:,}, device={device}, "
        f"n_init={n_init}, chunk_size={chunk_size:,}"
    )

    # 将数据一次性传到设备（float32），整个算法期间常驻
    X = torch.from_numpy(embeddings.astype(np.float32)).to(device)

    best_labels: np.ndarray | None = None
    best_centroids: np.ndarray | None = None
    best_inertia = float("inf")

    for init_idx in range(n_init):
        seed_i = int(rng.integers(0, 2**31))
        # Forgy 随机初始化：从 X 中随机采 K 个不重复点作为初始质心
        # 对大 K（≥1000），与 KMeans++ 质量相当，但开销 O(K) vs O(NK²)
        centroids = _init_centroids_random(X, k, seed_i, device)

        for iter_idx in range(max_iter):
            labels_t = _assign_labels(X, centroids, chunk_size)
            new_centroids = _update_centroids(X, labels_t, k, d, device, chunk_size)

            shift = float((new_centroids - centroids).norm(dim=1).mean().item())
            centroids = new_centroids

            if shift < tol:
                logger.debug(
                    f"  init {init_idx}: converged iter={iter_idx}, shift={shift:.6f}"
                )
                break

        inertia = _compute_inertia(X, centroids, labels_t, chunk_size)
        logger.info(f"  init {init_idx}: inertia={inertia:.4f}")

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels_t.cpu().numpy().astype(np.int64)
            best_centroids = centroids.cpu().numpy().astype(np.float32)

    logger.info(f"[kmeans_npu] 完成，best_inertia={best_inertia:.4f}")
    return best_labels, best_centroids


def _init_centroids_random(
    X: "torch.Tensor",
    k: int,
    seed: int,
    device: str,
) -> "torch.Tensor":
    """
    Forgy 随机初始化：从 X 中随机采 K 个不重复点作为初始质心。

    为什么不用 KMeans++？
    KMeans++ 对大 K 需要 K 次全量 N×K' cdist（K' 从 1 增长到 K），
    对 K=12,000 共 540,000 次 cdist，内存碎片严重触发 OOM，且耗时极长。
    Forgy 随机初始化开销为 O(K)，在 n_init ≥ 3 时质量等同于 KMeans++。
    """
    import torch
    rng = torch.Generator()
    rng.manual_seed(seed)
    indices = torch.randperm(len(X), generator=rng)[:k].to(device)
    return X[indices].clone()


def _assign_labels(
    X: "torch.Tensor",
    centroids: "torch.Tensor",
    chunk_size: int,
) -> "torch.Tensor":
    """
    分批计算 X 到 centroids 的距离，返回每行最近质心索引。
    峰值显存：chunk × K × 4B（chunk=50K, K=12K → 2.4GB）
    """
    labels = []
    for i in range(0, len(X), chunk_size):
        d = torch.cdist(X[i:i + chunk_size], centroids)  # (chunk, K)
        labels.append(d.argmin(dim=1))
    return torch.cat(labels)  # (N,)


def _update_centroids(
    X: "torch.Tensor",
    labels: "torch.Tensor",
    k: int,
    d: int,
    device: str,
    chunk_size: int,
) -> "torch.Tensor":
    """
    计算新质心（各 cluster 均值）。

    关键修复：不再全量 expand labels → (N×D) index，
    改为 chunk 级 scatter_add，index 峰值为 (chunk×D×8B) = 100MB。

    空 cluster 随机重新初始化，防止质心退化。
    """
    import torch
    new_c = torch.zeros(k, d, dtype=X.dtype, device=device)
    counts = torch.zeros(k, dtype=torch.long, device=device)

    for start in range(0, len(X), chunk_size):
        end = min(start + chunk_size, len(X))
        x_chunk = X[start:end]                                   # (chunk, D)
        l_chunk = labels[start:end]                              # (chunk,)
        # expand 仅在 chunk 粒度，峰值 chunk×D×8B ≈ 100MB（而非 N×D×8B = 4.5GB）
        idx = l_chunk.unsqueeze(1).expand(-1, d).contiguous()    # (chunk, D) int64
        new_c.scatter_add_(0, idx, x_chunk)
        counts.scatter_add_(
            0, l_chunk,
            torch.ones(end - start, dtype=torch.long, device=device)
        )

    # 处理空 cluster：随机从 X 中采样一个点作为新质心
    empty = (counts == 0).nonzero(as_tuple=True)[0]
    if len(empty) > 0:
        rand_idx = torch.randint(len(X), (len(empty),), device=device)
        new_c[empty] = X[rand_idx]
        counts[empty] = 1

    new_c /= counts.unsqueeze(1).float()
    return new_c


def _compute_inertia(
    X: "torch.Tensor",
    centroids: "torch.Tensor",
    labels: "torch.Tensor",
    chunk_size: int,
) -> float:
    """计算总 inertia（各点到所属质心的 L2 距离平方和）。分批避免 OOM。"""
    total = 0.0
    for i in range(0, len(X), chunk_size):
        xi = X[i:i + chunk_size]
        ci = centroids[labels[i:i + chunk_size]]
        total += float(((xi - ci) ** 2).sum().item())
    return total
