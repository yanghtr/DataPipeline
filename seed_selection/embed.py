"""
Step 6 — embed.py

用 Qwen3-Embedding-0.6B 对 instruction 文本生成 256 维向量，按 shard 输出。

特性：
- 按 shard_size 切片，每个 shard 写为 embeddings/shard_{i:04d}.npz
- Resume：已有且非空的 shard 文件直接跳过
- 支持 dry_run（用零向量代替，不加载模型）
- 截断到目标维度后重新 L2 归一化

CPU 时间参考（Qwen3-Embedding-0.6B）：
  batch_size=16，CPU：约 30–60 samples/s → 3M 条约 18–22 小时
  建议 GPU 运行或分批 resume。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from tqdm import tqdm

from .io_utils import read_jsonl


def _load_model(model_path: str, device: str):
    from sentence_transformers import SentenceTransformer
    logger.info(f"[embed] 加载模型: {model_path}（device={device}）")
    return SentenceTransformer(model_path, device=device)


def _embed_batch(model, texts: list[str], dimension: int) -> np.ndarray:
    embs = model.encode(
        texts,
        batch_size=len(texts),
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    embs = np.array(embs, dtype=np.float32)
    embs = embs[:, :dimension]
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.clip(norms, 1e-8, None)
    return embs


def run_embed(
    input_path: Path,
    output_dir: Path,
    model_path: str,
    dimension: int = 256,
    batch_size: int = 16,
    device: str = "cpu",
    shard_size: int = 100_000,
    dry_run: bool = False,
) -> dict[str, int]:
    """
    返回 {"total_records": N, "shards_written": M, "shards_skipped": K}。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取所有 id 和 instruction（只保留这两个字段，节省内存）
    logger.info(f"[embed] 读取 {input_path.name} ...")
    ids, texts = [], []
    for rec in read_jsonl(input_path):
        ids.append(rec["id"])
        texts.append(rec["instruction"])

    total = len(ids)
    n_shards = (total + shard_size - 1) // shard_size
    logger.info(f"[embed] 共 {total:,} 条，分 {n_shards} 个 shard（shard_size={shard_size:,}）")

    if dry_run:
        logger.warning("[embed] dry_run=True，使用零向量，不加载模型")

    model = None if dry_run else _load_model(model_path, device)

    shards_written = 0
    shards_skipped = 0

    for shard_idx in range(n_shards):
        shard_path = output_dir / f"shard_{shard_idx:04d}.npz"
        start = shard_idx * shard_size
        end = min(start + shard_size, total)

        if shard_path.exists() and shard_path.stat().st_size > 0:
            logger.info(f"[embed] shard {shard_idx:04d} 已存在，跳过")
            shards_skipped += 1
            continue

        shard_ids = ids[start:end]
        shard_texts = texts[start:end]
        n = len(shard_ids)

        if dry_run:
            embs = np.zeros((n, dimension), dtype=np.float32)
        else:
            all_embs = []
            for i in tqdm(
                range(0, n, batch_size),
                desc=f"shard {shard_idx:04d}",
                unit="batch",
            ):
                batch = shard_texts[i : i + batch_size]
                all_embs.append(_embed_batch(model, batch, dimension))
            embs = np.vstack(all_embs)

        np.savez_compressed(
            shard_path,
            ids=np.array(shard_ids, dtype=object),
            embeddings=embs,
        )
        shards_written += 1
        logger.info(
            f"[embed] shard {shard_idx:04d} 写出: {n} 条 → {shard_path.name}"
        )

    logger.info(
        f"[embed] 完成: total={total:,}, "
        f"shards_written={shards_written}, shards_skipped={shards_skipped}"
    )
    return {
        "total_records": total,
        "shards_written": shards_written,
        "shards_skipped": shards_skipped,
    }


def load_all_embeddings(output_dir: Path) -> tuple[list[str], np.ndarray]:
    """加载所有 shard，返回 (ids_list, embeddings_matrix)。"""
    shard_files = sorted(output_dir.glob("shard_*.npz"))
    if not shard_files:
        raise FileNotFoundError(f"No shard files found in {output_dir}")

    all_ids: list[str] = []
    all_embs: list[np.ndarray] = []
    for sf in shard_files:
        data = np.load(sf, allow_pickle=True)
        all_ids.extend(data["ids"].tolist())
        all_embs.append(data["embeddings"])

    return all_ids, np.vstack(all_embs)
