"""
NPU/GPU 加速 K-Means（标准 Lloyd's 算法）。

在 Ascend 910B（torch_npu）或 CUDA GPU（torch.cuda）上运行。
核心算子：torch.cdist，利用 NPU/GPU 矩阵乘加速 N×K 距离计算。

显存估算（D=256, FP32）：
  N=2.25M, K=12,000：完整 dists 矩阵 = 108 GB → 必须分批（chunk_size=50,000 → 2.4 GB/批）
  N=0.45M, K=6,000： 完整 dists 矩阵 = 10.8 GB → chunk=50,000 → 0.5 GB/批

使用示例：
    from seed_selection.kmeans_npu import kmeans_npu
    labels, centroids = kmeans_npu(embeddings, k=12000, device="npu:0")
"""

from __future__ import annotations

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
        device:      PyTorch 设备字符串，如 "npu:0"、"npu:1"、"cuda:0"
        n_init:      随机初始化次数，取 inertia 最小的结果
        max_iter:    每次初始化的最大迭代轮数
        tol:         质心变化量收敛阈值（L2 范数均值）
        random_seed: 随机种子（保证 n_init 间可复现）
        chunk_size:  分批 cdist 的批大小，控制显存峰值

    Returns:
        labels:    (N,) int64 数组，每条记录的 cluster id
        centroids: (K, D) float32 数组，最终质心
    """
    try:
        import torch
        import torch_npu  # noqa: F401  Ascend NPU 支持
    except ImportError:
        try:
            import torch
            if not torch.cuda.is_available() and "cuda" in device:
                raise RuntimeError(f"CUDA not available but device={device!r} requested")
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

    logger.info(f"[kmeans_npu] N={n:,}, K={k:,}, device={device}, n_init={n_init}")

    # 将数据一次性传到设备（float32）
    X = torch.from_numpy(embeddings.astype(np.float32)).to(device)

    best_labels: np.ndarray | None = None
    best_centroids: np.ndarray | None = None
    best_inertia = float("inf")

    for init_idx in range(n_init):
        seed_i = int(rng.integers(0, 2**31))
        centroids = _init_centroids(X, k, seed_i, device)

        for iter_idx in range(max_iter):
            labels_t = _assign_labels(X, centroids, chunk_size)
            new_centroids = _update_centroids(X, labels_t, k, d, device)

            shift = float((new_centroids - centroids).norm(dim=1).mean().item())
            centroids = new_centroids

            if shift < tol:
                logger.debug(f"  init {init_idx}: converged at iter {iter_idx}, shift={shift:.6f}")
                break

        inertia = _compute_inertia(X, centroids, labels_t, chunk_size)
        logger.info(f"  init {init_idx}: inertia={inertia:.4f}")

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels_t.cpu().numpy().astype(np.int64)
            best_centroids = centroids.cpu().numpy().astype(np.float32)

    logger.info(f"[kmeans_npu] 完成，best_inertia={best_inertia:.4f}")
    return best_labels, best_centroids


def _init_centroids(X: "torch.Tensor", k: int, seed: int, device: str) -> "torch.Tensor":
    """K-Means++ 初始化：第一个质心随机选，后续按距最近质心的概率选取。"""
    import torch
    n = X.shape[0]
    rng_t = torch.Generator(device=device)
    rng_t.manual_seed(seed)

    # 第一个质心
    idx = torch.randint(n, (1,), generator=rng_t, device=device).item()
    centroids = [X[idx]]

    for _ in range(1, k):
        # 计算每个点到已选质心的最小距离
        C = torch.stack(centroids)                       # (c, D)
        dists = _min_dist_to_centroids(X, C)             # (N,)
        probs = dists / dists.sum()
        idx = torch.multinomial(probs, 1, generator=rng_t).item()
        centroids.append(X[idx])

    return torch.stack(centroids)   # (K, D)


def _min_dist_to_centroids(X: "torch.Tensor", C: "torch.Tensor") -> "torch.Tensor":
    """返回每个点到 C 中最近质心的 L2 距离（分批以节省显存）。"""
    import torch
    chunk = 50_000
    min_dists = []
    for i in range(0, len(X), chunk):
        d = torch.cdist(X[i:i+chunk], C)   # (chunk, |C|)
        min_dists.append(d.min(dim=1).values)
    return torch.cat(min_dists)


def _assign_labels(
    X: "torch.Tensor",
    centroids: "torch.Tensor",
    chunk_size: int,
) -> "torch.Tensor":
    """分批计算 X 到 centroids 的距离，返回每行的最近质心索引。"""
    import torch
    labels = []
    for i in range(0, len(X), chunk_size):
        d = torch.cdist(X[i:i+chunk_size], centroids)   # (chunk, K)
        labels.append(d.argmin(dim=1))
    return torch.cat(labels)   # (N,)


def _update_centroids(
    X: "torch.Tensor",
    labels: "torch.Tensor",
    k: int,
    d: int,
    device: str,
) -> "torch.Tensor":
    """
    计算新质心（各 cluster 均值）。
    空 cluster 随机重新初始化，防止退化。
    """
    import torch
    new_c = torch.zeros(k, d, dtype=X.dtype, device=device)
    counts = torch.zeros(k, dtype=torch.long, device=device)

    new_c.scatter_add_(0, labels.unsqueeze(1).expand(-1, d), X)
    counts.scatter_add_(0, labels, torch.ones(len(X), dtype=torch.long, device=device))

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
    """计算总 inertia（各点到所属质心的 L2 距离平方和）。"""
    import torch
    total = 0.0
    for i in range(0, len(X), chunk_size):
        xi = X[i:i+chunk_size]
        li = labels[i:i+chunk_size]
        ci = centroids[li]
        total += float(((xi - ci) ** 2).sum().item())
    return total
