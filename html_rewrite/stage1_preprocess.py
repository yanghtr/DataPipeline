"""
Stage 1：批量预处理引擎。

输入：原始 JSONL（含 html、url 等字段）
输出：
  - preprocessed JSONL（仅 keep 样本，与 Stage 2 输出一一对应）
  - reject JSONL（被过滤或异常样本）
  - stats JSONL（按原始输入顺序记录逐条预处理统计与过滤结果）
  - summary JSON / histogram PNG（聚合统计与分布图）

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
from .preprocess import PreprocessStats, preprocess
from .preprocess.analysis import write_plots, write_reject_reason_report, write_summary
from .preprocess.filtering import decide_keep
from .preprocess.language import ensure_langid_available
from .preprocess.parser import ensure_lxml_available


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
    reject_log_path = Path(cfg.reject_log_path)
    summary_log_path = Path(cfg.summary_log_path)
    stats_plot_dir = Path(cfg.stats_plot_dir)
    reject_reason_report_path = summary_log_path.with_name(
        f"{summary_log_path.stem}_reject_reasons.json"
    )

    # 缺关键解析依赖时直接停止，避免静默降级或整批错误 reject。
    ensure_lxml_available()
    if cfg.enable_language_filter:
        ensure_langid_available()

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

    # ── 2. Resume：读取 keep / reject 结果（id → result），用于后续合并排序 ───
    existing_kept: dict[str, dict] = {}
    if cfg.resume and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line.strip())
                    if r.get("id"):
                        existing_kept[r["id"]] = r
                except Exception:
                    pass
    existing_rejected: dict[str, dict] = {}
    if cfg.resume and reject_log_path.exists():
        with open(reject_log_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line.strip())
                    if r.get("id"):
                        existing_rejected[r["id"]] = r
                except Exception:
                    pass
    existing_done = len(existing_kept) + len(existing_rejected)
    if existing_done:
        logger.info(
            f"[preprocess] resume：已完成 {existing_done:,} 条，将跳过 "
            f"(keep={len(existing_kept):,}, reject={len(existing_rejected):,})"
        )

    done_ids = set(existing_kept.keys()) | set(existing_rejected.keys())
    # 保留原始索引，只处理未完成的
    indexed_todo = [(i, rec) for i, rec in enumerate(records) if _get_id(rec) not in done_ids]

    logger.info(f"[preprocess] 待处理 {len(indexed_todo):,} 条，workers={cfg.num_workers}，输出={output_path}")

    # ── 3. 并行处理，结果收集进 dict（orig_idx → result） ────────────────────
    new_kept: dict[int, dict] = {}
    new_rejected: dict[int, dict] = {}
    new_stats_entries: dict[int, dict] = {}
    collect_lock = threading.Lock()
    kept_count = 0
    rejected_count = 0
    err_count = 0
    count_lock = threading.Lock()

    def process_one(orig_idx: int, rec: dict) -> None:
        nonlocal kept_count, rejected_count, err_count
        rec_id = _get_id(rec)
        meta = {k: rec[k] for k in ("url", "final_url", "crawl_time", "page_type", "part", "crawl_type") if k in rec}
        try:
            preprocessed_html, stats = preprocess(rec.get("html", ""), cfg)
            decision = decide_keep(preprocessed_html, stats, cfg)
            stats_dict = stats.to_dict()
            stats_entry = {
                "id": rec_id,
                "status": "kept" if decision.keep else "rejected",
                "reject_reason": decision.reason,
                "reject_details": decision.details,
                "stats": stats_dict,
            }
            with collect_lock:
                new_stats_entries[orig_idx] = stats_entry
                if decision.keep:
                    new_kept[orig_idx] = {
                        "id": rec_id,
                        "_meta": meta,
                        "preprocessed_html": preprocessed_html,
                        "preprocess_stats": stats_dict,
                    }
                else:
                    new_rejected[orig_idx] = {
                        "id": rec_id,
                        "_meta": meta,
                        "reject_reason": decision.reason,
                        "reject_details": decision.details,
                        "preprocess_stats": stats_dict,
                    }
            with count_lock:
                if decision.keep:
                    kept_count += 1
                else:
                    rejected_count += 1
        except Exception as exc:
            logger.warning(f"[preprocess] 失败 id={rec_id}: {exc}")
            fallback_stats = PreprocessStats(original_chars=len(rec.get("html", ""))).to_dict()
            with collect_lock:
                new_stats_entries[orig_idx] = {
                    "id": rec_id,
                    "status": "rejected",
                    "reject_reason": "preprocess_exception",
                    "reject_details": {
                        "rule": "preprocess_exception",
                        "error": str(exc),
                    },
                    "stats": fallback_stats,
                }
                new_rejected[orig_idx] = {
                    "id": rec_id,
                    "_meta": meta,
                    "reject_reason": "preprocess_exception",
                    "reject_details": {
                        "rule": "preprocess_exception",
                        "error": str(exc),
                    },
                    "preprocess_stats": fallback_stats,
                    "error": str(exc),
                }
            with count_lock:
                rejected_count += 1
                err_count += 1

        total = kept_count + rejected_count
        if total % 50 == 0:
            logger.info(
                f"[preprocess] 进度 {total:,}/{len(indexed_todo):,}  "
                f"keep={kept_count:,}  reject={rejected_count:,}  error={err_count:,}"
            )

    if indexed_todo:
        with ThreadPoolExecutor(max_workers=cfg.num_workers) as exe:
            futures = [exe.submit(process_one, idx, rec) for idx, rec in indexed_todo]
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as exc:
                    logger.error(f"[preprocess] worker 未捕获异常: {exc}")

    # ── 4. 合并已有结果 + 新结果，按原始索引排序后原子写出 ────────────────────
    existing_kept_by_idx: dict[int, dict] = {}
    existing_rejected_by_idx: dict[int, dict] = {}
    existing_stats_by_idx: dict[int, dict] = {}
    for i, rec in enumerate(records):
        rec_id = _get_id(rec)
        if rec_id in existing_kept:
            existing_kept_by_idx[i] = existing_kept[rec_id]
            existing_stats_by_idx[i] = {
                "id": rec_id,
                "status": "kept",
                "reject_reason": None,
                "reject_details": None,
                "stats": existing_kept[rec_id].get("preprocess_stats", {}),
            }
        elif rec_id in existing_rejected:
            existing_rejected_by_idx[i] = existing_rejected[rec_id]
            existing_stats_by_idx[i] = {
                "id": rec_id,
                "status": "rejected",
                "reject_reason": existing_rejected[rec_id].get("reject_reason"),
                "reject_details": existing_rejected[rec_id].get("reject_details"),
                "stats": existing_rejected[rec_id].get("preprocess_stats", {}),
            }

    # 新结果优先（支持强制重跑覆盖）
    kept_by_idx: dict[int, dict] = {**existing_kept_by_idx, **new_kept}
    rejected_by_idx: dict[int, dict] = {**existing_rejected_by_idx, **new_rejected}
    stats_by_idx: dict[int, dict] = {**existing_stats_by_idx, **new_stats_entries}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_log_path.parent.mkdir(parents=True, exist_ok=True)
    reject_log_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_out = output_path.with_suffix(".tmp")
    tmp_reject = reject_log_path.with_suffix(".tmp")
    tmp_stats = stats_log_path.with_suffix(".tmp")

    with open(tmp_out, "w", encoding="utf-8") as fo:
        for idx in sorted(kept_by_idx.keys()):
            fo.write(json.dumps(kept_by_idx[idx], ensure_ascii=False) + "\n")

    with open(tmp_reject, "w", encoding="utf-8") as fr:
        for idx in sorted(rejected_by_idx.keys()):
            fr.write(json.dumps(rejected_by_idx[idx], ensure_ascii=False) + "\n")

    with open(tmp_stats, "w", encoding="utf-8") as fs:
        for idx in sorted(stats_by_idx.keys()):
            fs.write(json.dumps(stats_by_idx[idx], ensure_ascii=False) + "\n")

    os.replace(tmp_out, output_path)
    os.replace(tmp_reject, reject_log_path)
    os.replace(tmp_stats, stats_log_path)

    ordered_stats_entries = [stats_by_idx[idx] for idx in sorted(stats_by_idx.keys())]
    reject_reason_report = write_reject_reason_report(reject_reason_report_path, ordered_stats_entries)
    summary = write_summary(summary_log_path, ordered_stats_entries, cfg)
    summary["reject_reason_report_path"] = str(reject_reason_report_path)
    summary["reject_reasons_detailed"] = reject_reason_report.get("reasons", {})
    written_plots = write_plots(stats_plot_dir, ordered_stats_entries, cfg)

    if written_plots:
        summary["plots"] = written_plots
    tmp_summary = summary_log_path.with_suffix(".tmp")
    tmp_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_summary, summary_log_path)

    logger.info(
        f"[preprocess] 完成：keep={len(kept_by_idx):,}  reject={len(rejected_by_idx):,}  "
        f"error={err_count:,}  输出={output_path}（keep，按原始顺序）"
    )


def _get_id(rec: dict) -> str:
    """使用 url 作为唯一 id（FineWebEdu 中 url 是主键）。"""
    return rec.get("url") or rec.get("id") or rec.get("_meta", {}).get("id", "")
