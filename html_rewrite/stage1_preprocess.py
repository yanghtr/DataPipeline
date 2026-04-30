"""
Stage 1：批量预处理引擎。

输入：原始 JSONL（含 html、url 等字段）
输出：
  - preprocessed JSONL（preprocessed_html + preprocess_stats）
  - stats JSONL（与 preprocessed 同内容的 stats 字段，便于独立分析）
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from loguru import logger

from .config import HtmlRewriteConfig
from .preprocess import preprocess


def run_preprocess(cfg: HtmlRewriteConfig, limit: int | None = None) -> None:
    """
    读取 cfg.input_path，并行预处理，结果写入 cfg.preprocessed_path。

    Args:
        cfg:   流水线配置
        limit: 仅处理前 N 条（调试用）
    """
    input_path = Path(cfg.input_path)
    output_path = Path(cfg.preprocessed_path)
    stats_log_path = Path(cfg.stats_log_path)

    # ── 1. 读取输入 ──────────────────────────────────────────────────────────
    records: list[dict] = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if limit is not None:
        records = records[:limit]

    logger.info(f"[preprocess] 共 {len(records):,} 条输入，来自 {input_path}")

    # ── 2. Resume：收集已完成的 id ──────────────────────────────────────────
    done_ids: set[str] = set()
    if cfg.resume and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line.strip())
                    if r.get("id"):
                        done_ids.add(r["id"])
                except Exception:
                    pass
        if done_ids:
            logger.info(f"[preprocess] resume：已完成 {len(done_ids):,} 条，将跳过")

    todo = [r for r in records if _get_id(r) not in done_ids]
    logger.info(f"[preprocess] 待处理 {len(todo):,} 条，workers={cfg.num_workers}，输出={output_path}")

    if not todo:
        logger.info("[preprocess] 全部已完成，退出")
        return

    # ── 3. 并行处理 ──────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_log_path.parent.mkdir(parents=True, exist_ok=True)

    write_lock = threading.Lock()
    ok_count = 0
    err_count = 0
    count_lock = threading.Lock()

    def process_one(rec: dict) -> None:
        nonlocal ok_count, err_count
        rec_id = _get_id(rec)
        html = rec.get("html", "")

        try:
            preprocessed_html, stats = preprocess(html, cfg)

            meta = {k: rec[k] for k in ("url", "final_url", "crawl_time", "page_type", "part", "crawl_type") if k in rec}
            result = {
                "id": rec_id,
                "_meta": meta,
                "preprocessed_html": preprocessed_html,
                "preprocess_stats": stats.to_dict(),
            }

            with write_lock:
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                with open(stats_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"id": rec_id, "stats": stats.to_dict()}, ensure_ascii=False) + "\n")

            with count_lock:
                ok_count += 1

        except Exception as exc:
            logger.warning(f"[preprocess] 失败 id={rec_id}: {exc}")
            with count_lock:
                err_count += 1

        total = ok_count + err_count
        if total % 50 == 0:
            logger.info(f"[preprocess] 进度 {total:,}/{len(todo):,}  ok={ok_count:,}  error={err_count:,}")

    with ThreadPoolExecutor(max_workers=cfg.num_workers) as exe:
        futures = [exe.submit(process_one, rec) for rec in todo]
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as exc:
                logger.error(f"[preprocess] worker 未捕获异常: {exc}")

    logger.info(f"[preprocess] 完成：ok={ok_count:,}  error={err_count:,}  输出={output_path}")


def _get_id(rec: dict) -> str:
    """使用 url 作为唯一 id（FineWebEdu 中 url 是主键）。"""
    return rec.get("url") or rec.get("id") or rec.get("_meta", {}).get("id", "")
