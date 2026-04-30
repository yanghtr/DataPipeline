"""
Stage 1：批量预处理引擎。

输入：原始 JSONL（含 html、url 等字段）
输出：
  - preprocessed JSONL（仅 keep 样本，与 Stage 2 输出一一对应）
  - reject JSONL（被过滤或异常样本）
  - stats JSONL（逐条预处理统计与过滤结果）
  - summary JSON / histogram PNG（聚合统计与分布图）

输出策略：
  最终 JSONL 文件采用 append-only 写入，不保证与输入同序。resume 基于 record_uid
  去重；summary / plot 在 shard 完成后重新聚合生成。
"""

from __future__ import annotations

import json
import os
import shutil
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
from .run_layout import (
    build_record_uid,
    build_manifest_payload,
    build_stage1_aggregate_summary,
    ensure_manifest,
    stage1_shard_paths,
)


def run_preprocess(cfg: HtmlRewriteConfig, limit: int | None = None) -> None:
    """
    读取输入 JSONL，并行预处理，append-only 写出 Stage 1 结果。

    Args:
        cfg:   流水线配置
        limit: 仅处理前 N 条（调试用）
    """
    # 缺关键解析依赖时直接停止，避免静默降级或整批错误 reject。
    ensure_lxml_available()
    if cfg.enable_language_filter:
        ensure_langid_available()

    input_mode, shard_specs = cfg.resolve_input_shards()
    if cfg.use_sharded_run():
        run_root_dir = Path(cfg.run_root_dir)
        manifest = build_manifest_payload(
            input_mode=input_mode,
            input_dir=cfg.input_dir,
            input_filename_template=cfg.input_filename_template,
            input_start_index=cfg.input_start_index,
            input_end_index_exclusive=cfg.input_end_index_exclusive,
            output_shard_name_template=cfg.output_shard_name_template,
            shard_specs=shard_specs,
        )
        ensure_manifest(run_root_dir, manifest)

        remaining_limit = limit
        for shard in shard_specs:
            if remaining_limit is not None and remaining_limit <= 0:
                break
            input_path = Path(shard.input_path)
            records = _read_records(
                input_path,
                remaining_limit,
                source_shard=shard.shard_name,
                source_input_path=shard.input_path,
            )
            shard_paths = stage1_shard_paths(run_root_dir, shard.shard_name)
            _run_preprocess_records(
                records=records,
                cfg=cfg,
                output_path=shard_paths.output_path,
                stats_log_path=shard_paths.stats_log_path,
                reject_log_path=shard_paths.reject_log_path,
                summary_log_path=shard_paths.summary_log_path,
                stats_plot_dir=shard_paths.stats_plot_dir,
                source_label=f"{input_path} -> stage1/{shard.shard_name}",
            )
            if remaining_limit is not None:
                remaining_limit -= len(records)

        agg_path = build_stage1_aggregate_summary(run_root_dir, shard_specs)
        logger.info(f"[preprocess] aggregate summary 已写出：{agg_path}")
        return

    input_paths = [Path(shard.input_path) for shard in shard_specs]
    records: list[dict] = []
    for shard in shard_specs:
        records.extend(
            _read_records(
                Path(shard.input_path),
                source_shard=shard.shard_name,
                source_input_path=shard.input_path,
            )
        )
    if limit is not None:
        records = records[:limit]

    logger.info(
        f"[preprocess] 共 {len(records):,} 条输入，来自 {len(input_paths)} 个文件："
        f" {[str(p) for p in input_paths]}"
    )
    _run_preprocess_records(
        records=records,
        cfg=cfg,
        output_path=Path(cfg.preprocessed_path),
        stats_log_path=Path(cfg.stats_log_path),
        reject_log_path=Path(cfg.reject_log_path),
        summary_log_path=Path(cfg.summary_log_path),
        stats_plot_dir=Path(cfg.stats_plot_dir),
        source_label=", ".join(str(p) for p in input_paths),
    )


def _run_preprocess_records(
    *,
    records: list[dict],
    cfg: HtmlRewriteConfig,
    output_path: Path,
    stats_log_path: Path,
    reject_log_path: Path,
    summary_log_path: Path,
    stats_plot_dir: Path,
    source_label: str,
) -> None:
    reject_reason_report_path = summary_log_path.with_name(
        f"{summary_log_path.stem}_reject_reasons.json"
    )
    if not cfg.resume:
        _reset_stage1_outputs(
            output_path=output_path,
            reject_log_path=reject_log_path,
            stats_log_path=stats_log_path,
            summary_log_path=summary_log_path,
            reject_reason_report_path=reject_reason_report_path,
            stats_plot_dir=stats_plot_dir,
        )

    existing_keep_keys = _load_existing_keys(output_path)
    existing_reject_keys = _load_existing_keys(reject_log_path)
    existing_done = len(existing_keep_keys | existing_reject_keys)
    if existing_done:
        logger.info(
            f"[preprocess] resume：已完成 {existing_done:,} 条，将跳过 "
            f"(keep={len(existing_keep_keys):,}, reject={len(existing_reject_keys):,})"
        )

    indexed_todo = [
        (i, rec)
        for i, rec in enumerate(records)
        if not _record_is_done(rec, i, existing_keep_keys | existing_reject_keys)
    ]

    logger.info(
        f"[preprocess] 待处理 {len(indexed_todo):,} 条，workers={cfg.num_workers}，输出={output_path}，来源={source_label}"
    )

    # ── 3. 并行处理，结果完成后直接追加到 keep / reject 文件 ─────────────────
    keep_lock = threading.Lock()
    reject_lock = threading.Lock()
    kept_count = 0
    rejected_count = 0
    err_count = 0
    count_lock = threading.Lock()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    reject_log_path.parent.mkdir(parents=True, exist_ok=True)
    stats_log_path.parent.mkdir(parents=True, exist_ok=True)

    def process_one(orig_idx: int, rec: dict) -> None:
        nonlocal kept_count, rejected_count, err_count
        rec_id = _get_id(rec)
        record_uid = _get_record_uid(rec, orig_idx)
        input_shard = rec.get("_source_shard", "")
        input_index = rec.get("_input_index", orig_idx)
        source_input_path = rec.get("_source_input_path", "")
        meta = {k: rec[k] for k in ("url", "final_url", "crawl_time", "page_type", "part", "crawl_type") if k in rec}
        try:
            preprocessed_html, stats = preprocess(rec.get("html", ""), cfg)
            decision = decide_keep(preprocessed_html, stats, cfg)
            stats_dict = stats.to_dict()
            if decision.keep:
                keep_record = {
                    "record_uid": record_uid,
                    "id": rec_id,
                    "input_shard": input_shard,
                    "input_index": input_index,
                    "source_input_path": source_input_path,
                    "_meta": meta,
                    "preprocessed_html": preprocessed_html,
                    "preprocess_stats": stats_dict,
                }
                _append_jsonl_record(output_path, keep_record, keep_lock)
            else:
                reject_record = {
                    "record_uid": record_uid,
                    "id": rec_id,
                    "input_shard": input_shard,
                    "input_index": input_index,
                    "source_input_path": source_input_path,
                    "_meta": meta,
                    "reject_reason": decision.reason,
                    "reject_details": decision.details,
                    "preprocess_stats": stats_dict,
                }
                _append_jsonl_record(reject_log_path, reject_record, reject_lock)
            with count_lock:
                if decision.keep:
                    kept_count += 1
                else:
                    rejected_count += 1
        except Exception as exc:
            logger.warning(f"[preprocess] 失败 id={rec_id}: {exc}")
            fallback_stats = PreprocessStats(original_chars=len(rec.get("html", ""))).to_dict()
            error_details = {
                "rule": "preprocess_exception",
                "error": str(exc),
            }
            _append_jsonl_record(
                reject_log_path,
                {
                    "record_uid": record_uid,
                    "id": rec_id,
                    "input_shard": input_shard,
                    "input_index": input_index,
                    "source_input_path": source_input_path,
                    "_meta": meta,
                    "reject_reason": "preprocess_exception",
                    "reject_details": error_details,
                    "preprocess_stats": fallback_stats,
                    "error": str(exc),
                },
                reject_lock,
            )
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

    stats_entries = _build_stats_entries_from_outputs(output_path, reject_log_path)
    _write_jsonl_atomic(stats_log_path, stats_entries)
    reject_reason_report = write_reject_reason_report(reject_reason_report_path, stats_entries)
    summary = write_summary(summary_log_path, stats_entries, cfg)
    summary["reject_reason_report_path"] = str(reject_reason_report_path)
    summary["reject_reasons_detailed"] = reject_reason_report.get("reasons", {})
    written_plots = write_plots(stats_plot_dir, stats_entries, cfg)

    if written_plots:
        summary["plots"] = written_plots
    tmp_summary = summary_log_path.with_suffix(".tmp")
    tmp_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_summary, summary_log_path)

    logger.info(
        f"[preprocess] 完成：keep={summary.get('kept', 0):,}  reject={summary.get('rejected', 0):,}  "
        f"error={err_count:,}  输出={output_path}（append-only，无序）"
    )


def _read_records(
    path: Path,
    limit: int | None = None,
    *,
    source_shard: str | None = None,
    source_input_path: str | None = None,
) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for input_index, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            shard_name = source_shard or "part-00000"
            record["_source_shard"] = shard_name
            record["_input_index"] = input_index
            record["_source_input_path"] = source_input_path or str(path)
            record["_record_uid"] = build_record_uid(shard_name, input_index)
            records.append(record)
            if limit is not None and len(records) >= limit:
                break
    return records


def _read_jsonl_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def _append_jsonl_record(path: Path, record: dict, lock: threading.Lock) -> None:
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_jsonl_atomic(path: Path, records: list[dict]) -> None:
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def _load_existing_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    if not path.exists():
        return keys
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue
            key = record.get("record_uid") or record.get("id")
            if key:
                keys.add(key)
    return keys


def _record_is_done(rec: dict, input_index: int, existing_keys: set[str]) -> bool:
    record_uid = _get_record_uid(rec, input_index)
    raw_id = _get_id(rec)
    return record_uid in existing_keys or (raw_id in existing_keys if raw_id else False)


def _reset_stage1_outputs(
    *,
    output_path: Path,
    reject_log_path: Path,
    stats_log_path: Path,
    summary_log_path: Path,
    reject_reason_report_path: Path,
    stats_plot_dir: Path,
) -> None:
    for path in (output_path, reject_log_path, stats_log_path, summary_log_path, reject_reason_report_path):
        if path.exists():
            path.unlink()
    if stats_plot_dir.exists():
        shutil.rmtree(stats_plot_dir)


def _build_stats_entries_from_outputs(output_path: Path, reject_log_path: Path) -> list[dict]:
    stats_entries: list[dict] = []
    for record in _read_jsonl_records(output_path):
        stats_entries.append(
            {
                "record_uid": record.get("record_uid"),
                "id": record.get("id"),
                "input_shard": record.get("input_shard"),
                "input_index": record.get("input_index"),
                "source_input_path": record.get("source_input_path"),
                "status": "kept",
                "reject_reason": None,
                "reject_details": None,
                "stats": record.get("preprocess_stats", {}),
            }
        )
    for record in _read_jsonl_records(reject_log_path):
        stats_entries.append(
            {
                "record_uid": record.get("record_uid"),
                "id": record.get("id"),
                "input_shard": record.get("input_shard"),
                "input_index": record.get("input_index"),
                "source_input_path": record.get("source_input_path"),
                "status": "rejected",
                "reject_reason": record.get("reject_reason"),
                "reject_details": record.get("reject_details"),
                "stats": record.get("preprocess_stats", {}),
            }
        )
    return stats_entries


def _get_id(rec: dict) -> str:
    """保留原始网页 id / url；真正的流水线唯一键使用 record_uid。"""
    return rec.get("url") or rec.get("id") or rec.get("_meta", {}).get("id", "")


def _get_record_uid(rec: dict, input_index: int) -> str:
    if rec.get("_record_uid"):
        return rec["_record_uid"]
    shard_name = rec.get("_source_shard", "part-00000")
    return build_record_uid(shard_name, int(rec.get("_input_index", input_index)))
