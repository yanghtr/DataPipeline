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
from concurrent.futures import ProcessPoolExecutor
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


def _embed_shards_on_device(args: tuple) -> tuple[int, int]:
    """
    Worker 函数：在指定设备上处理分配给该 worker 的 shard 列表。
    返回: (shards_written, shards_skipped)
    """
    (shard_indices, output_dir_str, ids_list, texts_list, shard_size,
     model_path, dimension, batch_size, device_str, dry_run) = args

    output_dir = Path(output_dir_str)
    written = skipped = 0

    if not dry_run:
        model = _load_model(model_path, device_str)
    else:
        model = None

    for shard_idx in shard_indices:
        shard_path = output_dir / f"shard_{shard_idx:04d}.npz"
        start = shard_idx * shard_size
        # ids_list and texts_list are already the shard-specific slices
        shard_ids = ids_list[shard_idx]
        shard_texts = texts_list[shard_idx]
        n = len(shard_ids)

        if shard_path.exists() and shard_path.stat().st_size > 0:
            skipped += 1
            continue

        if dry_run:
            embs = np.zeros((n, dimension), dtype=np.float32)
        else:
            all_embs = []
            for i in tqdm(range(0, n, batch_size), desc=f"shard {shard_idx:04d}", unit="batch"):
                batch = shard_texts[i : i + batch_size]
                all_embs.append(_embed_batch(model, batch, dimension))
            embs = np.vstack(all_embs)

        np.savez_compressed(
            shard_path,
            ids=np.array(shard_ids, dtype=object),
            embeddings=embs,
        )
        written += 1

    return written, skipped


def run_embed(
    input_path: Path,
    output_dir: Path,
    model_path: str,
    dimension: int = 256,
    batch_size: int = 16,
    device: str = "cpu",
    shard_size: int = 100_000,
    dry_run: bool = False,
    num_devices: int = 1,
) -> dict[str, int]:
    """
    返回 {"total_records": N, "shards_written": M, "shards_skipped": K}。

    num_devices > 1 时，shard 均分到各卡（设备命名：{device}:0, {device}:1, ...）。
    CPU 模式固定为单进程（num_devices 被忽略）。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取所有 id 和 instruction
    logger.info(f"[embed] 读取 {input_path.name} ...")
    ids, texts = [], []
    for rec in read_jsonl(input_path):
        ids.append(rec["_meta"]["id"])
        texts.append(rec["instruction"])

    total = len(ids)
    n_shards = (total + shard_size - 1) // shard_size
    logger.info(f"[embed] 共 {total:,} 条，分 {n_shards} 个 shard（shard_size={shard_size:,}）")

    if dry_run:
        logger.warning("[embed] dry_run=True，使用零向量，不加载模型")

    # 预先切分为 shard 列表（避免跨进程传大数组）
    shard_ids_list = []
    shard_texts_list = []
    for si in range(n_shards):
        s, e = si * shard_size, min((si + 1) * shard_size, total)
        shard_ids_list.append(ids[s:e])
        shard_texts_list.append(texts[s:e])

    # 多卡仅在非 CPU 模式下生效
    effective_devices = 1 if device == "cpu" else max(1, num_devices)

    if effective_devices > 1 and n_shards > 1:
        # 按卡数均分 shard 索引
        shard_groups: list[list[int]] = [[] for _ in range(effective_devices)]
        for si in range(n_shards):
            shard_groups[si % effective_devices].append(si)

        worker_args = []
        for device_id, group in enumerate(shard_groups):
            if not group:
                continue
            device_str = f"{device}:{device_id}"
            worker_args.append((
                group,
                str(output_dir),
                shard_ids_list,
                shard_texts_list,
                shard_size,
                model_path, dimension, batch_size, device_str, dry_run,
            ))

        logger.info(f"[embed] 使用 {effective_devices} 个设备（{device}:0 ~ {device}:{effective_devices-1}）")
        shards_written = shards_skipped = 0
        with ProcessPoolExecutor(max_workers=effective_devices) as exe:
            for written, skipped in exe.map(_embed_shards_on_device, worker_args):
                shards_written += written
                shards_skipped += skipped
    else:
        # 单设备顺序处理
        written, skipped = _embed_shards_on_device((
            list(range(n_shards)),
            str(output_dir),
            shard_ids_list,
            shard_texts_list,
            shard_size,
            model_path, dimension, batch_size, device, dry_run,
        ))
        shards_written, shards_skipped = written, skipped

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
