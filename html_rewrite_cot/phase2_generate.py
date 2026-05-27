"""Phase 2：VLM reasoning 生成（ThreadPoolExecutor + utils/api_client）。"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from loguru import logger

from html_rewrite_cot.config import PipelineConfig
from html_rewrite_cot.models import (
    SampleRecord,
    append_done_id,
    append_jsonl,
    load_done_ids,
)
from html_rewrite_cot.pipeline.assembler import assemble_final_answer
from html_rewrite_cot.pipeline.postprocess import check_quality, postprocess_reasoning
from html_rewrite_cot.pipeline.vlm import call_vlm


def _make_output_paths(config: PipelineConfig, jsonl_file: str) -> tuple[Path, Path | None]:
    """返回 (output_jsonl_path, debug_jsonl_path)。"""
    stem = Path(jsonl_file).stem
    out_path = Path(config.output.output_dir) / f"{stem}.jsonl"
    dbg_path = (
        Path(config.output.debug_dir) / f"{stem}.jsonl"
        if config.output.debug_dir
        else None
    )
    return out_path, dbg_path


def _process_one(
    record: SampleRecord,
    config: PipelineConfig,
    out_path: Path,
    dbg_path: Path | None,
    done_path: Path,
    write_lock: threading.Lock,
    call_log_path: str | None,
    stats: dict,
    stats_lock: threading.Lock,
) -> None:
    sample_id = record.sample_id
    t0 = time.monotonic()

    # 跳过 Phase 1 失败的样本
    if record.extraction_status == "failed":
        logger.warning(f"[phase2] {sample_id}: skipped (phase1 failed)")
        with stats_lock:
            stats["skipped"] += 1
        return

    # outline 未完成（不应发生，防御性检查）
    if record.outline_text is None or record.html_outline_json is None:
        logger.warning(f"[phase2] {sample_id}: skipped (no outline)")
        with stats_lock:
            stats["skipped"] += 1
        return

    # 构造图片完整路径
    image_full_path = str(Path(config.input.image_root) / record.image_rel_path)

    try:
        vlm_raw = call_vlm(
            image_path=image_full_path,
            outline_text=record.outline_text,
            raw_html=record.raw_html,
            image_format=record.image_format,
            config=config.vlm,
            call_log_path=call_log_path,
        )
    except Exception as e:
        elapsed = time.monotonic() - t0
        logger.error(f"[phase2] {sample_id}: VLM failed ({elapsed:.1f}s): {e}")
        record.generation_status = "failed"
        record.generation_warnings = [str(e)]
        with stats_lock:
            stats["failed"] += 1
        return

    # 后处理
    reasoning_text, gen_warnings = postprocess_reasoning(vlm_raw)
    final_answer = assemble_final_answer(reasoning_text, record.raw_html)
    quality = check_quality(reasoning_text, record.raw_html)

    record.vlm_model = config.vlm.model
    record.vlm_reasoning_raw = vlm_raw
    record.reasoning_text = reasoning_text
    record.final_answer = final_answer
    record.generation_warnings = gen_warnings
    record.quality_metadata = quality
    record.generation_status = "warning" if gen_warnings else "ok"

    elapsed = time.monotonic() - t0
    if gen_warnings:
        logger.warning(f"[phase2] {sample_id}: ok with warnings ({elapsed:.1f}s): {gen_warnings}")
        with stats_lock:
            stats["warning"] += 1
    else:
        logger.info(f"[phase2] {sample_id}: ok ({elapsed:.1f}s, {quality['reasoning_word_count']} words)")
        with stats_lock:
            stats["ok"] += 1

    # 写输出（加锁）
    with write_lock:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        append_jsonl(record.to_panguml_output(), out_path)
        if dbg_path:
            dbg_path.parent.mkdir(parents=True, exist_ok=True)
            append_jsonl(record.to_debug_dict(api_base=config.vlm.url), dbg_path)
        append_done_id(sample_id, done_path)


def run_phase2(
    all_records: list[SampleRecord],
    config: PipelineConfig,
) -> None:
    """
    执行 Phase 2：批量调用 VLM 生成 reasoning，写出 panguml 输出和 debug 输出。

    resume=True 时读取 done 文件跳过已完成样本。
    """
    run_dir = Path(config.runtime.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    done_path = run_dir / "phase2_done.txt"

    # 按输入文件分组，各自对应独立输出文件
    from collections import defaultdict
    by_file: dict[str, list[SampleRecord]] = defaultdict(list)
    for r in all_records:
        by_file[r.jsonl_file].append(r)

    # Resume：读取已完成 sample_id
    done_ids: set[str] = set()
    if config.runtime.resume:
        done_ids = load_done_ids(done_path)
        # Also scan debug output files: if the process was killed between writing the
        # panguml output and writing the done_id, the sample is in debug but not in done.txt.
        # Scanning debug (which contains sample_id) makes resume fully idempotent.
        if config.output.debug_dir:
            import json as _json
            for jsonl_file in by_file:
                _, dbg_path = _make_output_paths(config, jsonl_file)
                if dbg_path and dbg_path.exists():
                    with open(dbg_path, "r", encoding="utf-8") as _f:
                        for _line in _f:
                            _line = _line.strip()
                            if not _line:
                                continue
                            try:
                                _d = _json.loads(_line)
                                sid = _d.get("sample_id")
                                if sid:
                                    done_ids.add(sid)
                            except Exception:
                                pass
        logger.info(f"[phase2] 已完成 {len(done_ids)} 条（resume，含 debug 扫描）")

    to_process: list[tuple[SampleRecord, Path, Path | None]] = []
    for jsonl_file, records in by_file.items():
        out_path, dbg_path = _make_output_paths(config, jsonl_file)
        for r in records:
            if r.sample_id in done_ids:
                continue
            to_process.append((r, out_path, dbg_path))

    logger.info(f"[phase2] 待处理 {len(to_process)} 条，共 {len(all_records)} 条")
    if not to_process:
        logger.info("[phase2] 所有记录已完成，跳过 Phase 2")
        return

    # 日志路径
    call_log_path = str(run_dir / "vlm_calls.jsonl")

    write_lock = threading.Lock()
    stats: dict[str, int] = {"ok": 0, "warning": 0, "failed": 0, "skipped": 0}
    stats_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=config.runtime.num_workers) as executor:
        futures = {
            executor.submit(
                _process_one,
                record, config, out_path, dbg_path, done_path,
                write_lock, call_log_path, stats, stats_lock,
            ): record.sample_id
            for record, out_path, dbg_path in to_process
        }
        completed = 0
        total = len(futures)
        for future in as_completed(futures):
            completed += 1
            sid = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"[phase2] {sid}: unhandled exception: {e}")
                with stats_lock:
                    stats["failed"] += 1
            if completed % 10 == 0 or completed == total:
                with stats_lock:
                    s = dict(stats)
                logger.info(
                    f"[phase2] 进度 {completed}/{total} | "
                    f"ok={s['ok']} warn={s['warning']} fail={s['failed']} skip={s['skipped']}"
                )

    logger.info(
        f"[phase2] 完成：ok={stats['ok']} warning={stats['warning']} "
        f"failed={stats['failed']} skipped={stats['skipped']}"
    )
