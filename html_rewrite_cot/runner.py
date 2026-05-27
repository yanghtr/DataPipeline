"""主流程编排：读取输入 → Phase 1 → Phase 2 → 统计汇报。"""

from __future__ import annotations

import json
import time
from pathlib import Path

from loguru import logger

from html_rewrite_cot.config import PipelineConfig
from html_rewrite_cot.models import SampleRecord, build_sample_records
from html_rewrite_cot.phase1_outline import run_phase1
from html_rewrite_cot.phase2_generate import run_phase2


def _setup_logger(run_dir: str) -> None:
    log_path = Path(run_dir) / "pipeline.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_path), rotation="100 MB", encoding="utf-8", level="DEBUG")


def _load_all_records(config: PipelineConfig) -> list[SampleRecord]:
    """从所有配置的 JSONL 文件读取 SampleRecord。"""
    all_records: list[SampleRecord] = []
    for jsonl_file in config.input.jsonl_files:
        path = Path(jsonl_file)
        if not path.exists():
            logger.error(f"输入文件不存在：{path}")
            continue
        records = build_sample_records(path)
        logger.info(f"读取 {path.name}：{len(records)} 条样本")
        all_records.extend(records)
    return all_records


def _print_summary(all_records: list[SampleRecord], elapsed: float) -> None:
    p1_ok = sum(1 for r in all_records if r.extraction_status in ("ok", "warning"))
    p1_fail = sum(1 for r in all_records if r.extraction_status == "failed")
    p2_ok = sum(1 for r in all_records if r.generation_status in ("ok", "warning"))
    p2_fail = sum(1 for r in all_records if r.generation_status == "failed")
    p2_skip = sum(1 for r in all_records if r.generation_status == "pending")

    word_counts = [
        r.quality_metadata["reasoning_word_count"]
        for r in all_records
        if r.quality_metadata and r.generation_status in ("ok", "warning")
    ]
    avg_words = sum(word_counts) / len(word_counts) if word_counts else 0

    has_layout = sum(
        1 for r in all_records
        if r.quality_metadata and r.quality_metadata.get("has_layout_analysis_section")
    )
    has_colors = sum(
        1 for r in all_records
        if r.quality_metadata and r.quality_metadata.get("has_colors_observed_section")
    )
    has_plan = sum(
        1 for r in all_records
        if r.quality_metadata and r.quality_metadata.get("has_structure_implementation_plan_section")
    )
    html_in_reason = sum(
        1 for r in all_records
        if r.quality_metadata and r.quality_metadata.get("contains_html_in_reasoning")
    )
    timeouts = sum(1 for r in all_records if r.extraction_timeout)

    summary = {
        "total": len(all_records),
        "elapsed_s": round(elapsed, 1),
        "phase1": {"ok_or_warning": p1_ok, "failed": p1_fail, "playwright_timeout": timeouts},
        "phase2": {"ok_or_warning": p2_ok, "failed": p2_fail, "skipped": p2_skip},
        "quality": {
            "avg_reasoning_words": round(avg_words, 1),
            "has_layout_analysis": has_layout,
            "has_colors_observed": has_colors,
            "has_structure_plan": has_plan,
            "html_in_reasoning_count": html_in_reason,
        },
    }
    logger.info("=== 流水线汇总 ===")
    logger.info(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def run(
    config: PipelineConfig,
    limit: int | None = None,
    phase: str = "all",
) -> None:
    """
    运行完整流水线。

    Args:
        config: 流水线配置
        limit:  仅处理前 N 条（调试用）
        phase:  "phase1" | "phase2" | "all"
    """
    _setup_logger(config.runtime.run_dir)
    t0 = time.monotonic()

    logger.info(f"=== html_rewrite_cot 流水线启动 (phase={phase}) ===")
    logger.info(f"输入文件: {config.input.jsonl_files}")
    logger.info(f"image_root: {config.input.image_root}")
    logger.info(f"output_dir: {config.output.output_dir}")

    all_records = _load_all_records(config)
    if not all_records:
        logger.error("没有读取到任何样本，退出")
        return

    if limit is not None:
        all_records = all_records[:limit]
        logger.info(f"--limit {limit}: 仅处理前 {len(all_records)} 条")

    if phase in ("phase1", "all"):
        logger.info("--- 开始 Phase 1：HTML outline 提取 ---")
        run_phase1(all_records, config)

    if phase in ("phase2", "all"):
        # 单独跑 phase2 时 phase1 未执行，从 outline cache 加载数据
        if phase == "phase2":
            from html_rewrite_cot.models import load_outline_cache
            cache_path = Path(config.runtime.run_dir) / "outlines.jsonl"
            cached = load_outline_cache(cache_path)
            for r in all_records:
                if r.sample_id in cached:
                    r.apply_outline_cache(cached[r.sample_id])
            logger.info(f"[runner] 从 outline cache 加载 {len(cached)} 条记录")
        logger.info("--- 开始 Phase 2：VLM reasoning 生成 ---")
        run_phase2(all_records, config)

    elapsed = time.monotonic() - t0
    _print_summary(all_records, elapsed)
    logger.info(f"=== 流水线完成，总耗时 {elapsed:.1f}s ===")
