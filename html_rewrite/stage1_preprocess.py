"""
Stage 1：批量预处理引擎。

输入：原始 JSONL（含 html、url 等字段）
输出：
  - preprocessed JSONL（与原始输入严格同序，便于与 Stage 2 输出一一对应）
  - stats JSONL（按相同顺序记录逐条预处理统计）

顺序保证策略：
  并发处理后按原始索引排序，原子写出（.tmp 重命名），resume 时读取已有
  输出合并后一并按序覆盖写入。
"""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from loguru import logger

from .config import HtmlRewriteConfig
from .preprocess import preprocess


def run_preprocess(cfg: HtmlRewriteConfig, limit: int | None = None) -> None:
    """
    读取 cfg.input_path，并行预处理，按原始输入顺序写入 cfg.preprocessed_path。

    Args:
        cfg:   流水线配置
        limit: 仅处理前 N 条（调试用）
    """
    input_path = Path(cfg.input_path)
    output_path = Path(cfg.preprocessed_path)
    stats_log_path = Path(cfg.stats_log_path)

    # ── 1. 读取全量输入 ───────────────────────────────────────────────────────
    records: list[dict] = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if limit is not None:
        records = records[:limit]

    logger.info(f"[preprocess] 共 {len(records):,} 条输入，来自 {input_path}")

    # ── 2. Resume：读取已完成结果（id → result），用于后续合并排序 ───────────
    existing: dict[str, dict] = {}          # id -> preprocessed record
    if cfg.resume and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line.strip())
                    if r.get("id"):
                        existing[r["id"]] = r
                except Exception:
                    pass
        if existing:
            logger.info(f"[preprocess] resume：已完成 {len(existing):,} 条，将跳过")

    done_ids = set(existing.keys())
    # 保留原始索引，只处理未完成的
    indexed_todo = [(i, rec) for i, rec in enumerate(records) if _get_id(rec) not in done_ids]

    logger.info(f"[preprocess] 待处理 {len(indexed_todo):,} 条，workers={cfg.num_workers}，输出={output_path}")

    # ── 3. 并行处理，结果收集进 dict（orig_idx → result） ────────────────────
    new_results: dict[int, dict] = {}
    collect_lock = threading.Lock()
    ok_count = 0
    err_count = 0
    count_lock = threading.Lock()

    def process_one(orig_idx: int, rec: dict) -> None:
        nonlocal ok_count, err_count
        rec_id = _get_id(rec)
        try:
            preprocessed_html, stats = preprocess(rec.get("html", ""), cfg)
            meta = {k: rec[k] for k in ("url", "final_url", "crawl_time", "page_type", "part", "crawl_type") if k in rec}
            result = {
                "id": rec_id,
                "_meta": meta,
                "preprocessed_html": preprocessed_html,
                "preprocess_stats": stats.to_dict(),
            }
            with collect_lock:
                new_results[orig_idx] = result
            with count_lock:
                ok_count += 1
        except Exception as exc:
            logger.warning(f"[preprocess] 失败 id={rec_id}: {exc}")
            with count_lock:
                err_count += 1

        total = ok_count + err_count
        if total % 50 == 0:
            logger.info(f"[preprocess] 进度 {total:,}/{len(indexed_todo):,}  ok={ok_count:,}  error={err_count:,}")

    if indexed_todo:
        with ThreadPoolExecutor(max_workers=cfg.num_workers) as exe:
            futures = [exe.submit(process_one, idx, rec) for idx, rec in indexed_todo]
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as exc:
                    logger.error(f"[preprocess] worker 未捕获异常: {exc}")

    # ── 4. 合并已有结果 + 新结果，按原始索引排序后原子写出 ────────────────────
    # 将 existing 映射回原始索引
    existing_by_idx: dict[int, dict] = {}
    for i, rec in enumerate(records):
        rec_id = _get_id(rec)
        if rec_id in existing:
            existing_by_idx[i] = existing[rec_id]

    # 新结果优先（支持强制重跑覆盖）
    all_by_idx: dict[int, dict] = {**existing_by_idx, **new_results}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_log_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_out = output_path.with_suffix(".tmp")
    tmp_stats = stats_log_path.with_suffix(".tmp")

    with open(tmp_out, "w", encoding="utf-8") as fo, \
         open(tmp_stats, "w", encoding="utf-8") as fs:
        for idx in sorted(all_by_idx.keys()):
            result = all_by_idx[idx]
            fo.write(json.dumps(result, ensure_ascii=False) + "\n")
            fs.write(json.dumps(
                {"id": result["id"], "stats": result["preprocess_stats"]},
                ensure_ascii=False,
            ) + "\n")

    os.replace(tmp_out, output_path)
    os.replace(tmp_stats, stats_log_path)

    logger.info(
        f"[preprocess] 完成：ok={ok_count:,}  error={err_count:,}  "
        f"输出={output_path}（共 {len(all_by_idx):,} 条，按原始顺序）"
    )


def _get_id(rec: dict) -> str:
    """使用 url 作为唯一 id（FineWebEdu 中 url 是主键）。"""
    return rec.get("url") or rec.get("id") or rec.get("_meta", {}).get("id", "")
