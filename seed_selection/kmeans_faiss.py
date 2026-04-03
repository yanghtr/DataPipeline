"""
faiss 加速 K-Means（精确 Lloyd's 算法，CPU BLAS 优化）。

faiss.Kmeans 是标准 Lloyd's 算法的高度优化 C++ 实现，底层使用 BLAS/LAPACK。
在 CPU 上比 sklearn MiniBatchKMeans 快 5–15×（精确而非近似）。

注意：
  - faiss-gpu 仅支持 CUDA，不支持 Ascend NPU
  - faiss-cpu 纯 CPU，无需 GPU 即可使用
  - 安装：pip install faiss-cpu

使用示例：
    from seed_selection.kmeans_faiss import kmeans_faiss
    labels, centroids = kmeans_faiss(embeddings, k=12000)
"""

from __future__ import annotations

import numpy as np
from loguru import logger


def kmeans_faiss(
    embeddings: np.ndarray,
    k: int,
    n_init: int = 3,
    max_iter: int = 100,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    精确 Lloyd's K-Means，使用 faiss（CPU BLAS 优化）。

    Args:
        embeddings:  (N, D) float32 numpy 数组（必须 C-contiguous）
        k:           聚类数（若 k > N，自动截断为 N）
        n_init:      随机重启次数（faiss 内部参数 nredo），取 inertia 最小结果
        max_iter:    最大迭代轮数（faiss 内部参数 niter）
        random_seed: 随机种子

    Returns:
        labels:    (N,) int64 数组
        centroids: (K, D) float32 数组
    """
    try:
        import faiss
    except ImportError:
        raise ImportError(
            "faiss not found. Install with: pip install faiss-cpu\n"
            "(faiss-gpu is CUDA-only and does not support Ascend NPU)"
        )

    k = min(k, len(embeddings))
    n, d = embeddings.shape

    logger.info(f"[kmeans_faiss] N={n:,}, K={k:,}, n_init={n_init}, max_iter={max_iter}")

    # faiss 要求 float32 C-contiguous
    X = np.ascontiguousarray(embeddings.astype(np.float32))

    km = faiss.Kmeans(
        d,           # 向量维度
        k,           # 聚类数
        niter=max_iter,
        nredo=n_init,       # 多次随机重启取最优
        seed=random_seed,
        verbose=False,
        spherical=False,    # 不做 L2 归一化（我们的 embedding 已归一化）
    )
    km.train(X)

    # 获取 assignment：index.search 返回 (N, 1) 最近邻
    _, labels_2d = km.index.search(X, 1)
    labels = labels_2d.flatten().astype(np.int64)
    centroids = km.centroids.astype(np.float32)

    # 计算 inertia 用于日志
    inertia = float(np.sum((X - centroids[labels]) ** 2))
    logger.info(f"[kmeans_faiss] 完成，inertia={inertia:.4f}")

    return labels, centroids
