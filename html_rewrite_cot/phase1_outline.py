"""Phase 1：HTML outline 批量提取（asyncio + Playwright 共享 browser）。"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from loguru import logger

from html_rewrite_cot.config import PipelineConfig
from html_rewrite_cot.models import SampleRecord, append_jsonl, load_outline_cache
from html_rewrite_cot.pipeline.outliner import extract_outline
from html_rewrite_cot.pipeline.renderer import render_outline_text


async def _process_one(
    record: SampleRecord,
    config: PipelineConfig,
    browser,
    sem: asyncio.Semaphore,
    cache_path: Path,
    cache_lock: asyncio.Lock,
    stats: dict,
) -> None:
    async with sem:
        t0 = time.monotonic()
        sample_id = record.sample_id
        try:
            outline_json = await extract_outline(
                html=record.raw_html,
                image_width=record.image_width,
                image_height=record.image_height,
                config=config.render,
                browser=browser,
            )
            outline_text = render_outline_text(outline_json)

            warnings = outline_json["meta"].get("warnings", [])
            timed_out = any("timed out" in w for w in warnings)

            record.html_outline_json = outline_json
            record.outline_text = outline_text
            record.extraction_warnings = warnings
            record.extraction_timeout = timed_out
            record.extraction_status = "warning" if warnings else "ok"

            elapsed = time.monotonic() - t0
            if timed_out:
                logger.warning(f"[phase1] {sample_id}: playwright timeout ({elapsed:.1f}s)")
                stats["timeout"] += 1
            else:
                logger.info(f"[phase1] {sample_id}: ok ({elapsed:.1f}s)")
                stats["ok"] += 1

        except Exception as e:
            elapsed = time.monotonic() - t0
            logger.error(f"[phase1] {sample_id}: failed ({elapsed:.1f}s): {e}")
            record.extraction_status = "failed"
            record.extraction_warnings = [str(e)]
            stats["failed"] += 1

        # 写入 outline cache（append，加锁保证顺序）
        async with cache_lock:
            append_jsonl(record.to_outline_cache_dict(), cache_path)


async def _run_async(
    records: list[SampleRecord],
    config: PipelineConfig,
    cache_path: Path,
) -> None:
    from playwright.async_api import async_playwright

    sem = asyncio.Semaphore(config.runtime.playwright_concurrency)
    cache_lock = asyncio.Lock()
    stats = {"ok": 0, "timeout": 0, "failed": 0, "skipped": 0}

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        try:
            tasks = [
                _process_one(r, config, browser, sem, cache_path, cache_lock, stats)
                for r in records
            ]
            await asyncio.gather(*tasks)
        finally:
            await browser.close()

    total = sum(stats.values())
    logger.info(
        f"[phase1] 完成：total={total} ok={stats['ok']} "
        f"timeout={stats['timeout']} failed={stats['failed']} skipped={stats['skipped']}"
    )


def run_phase1(
    all_records: list[SampleRecord],
    config: PipelineConfig,
) -> None:
    """
    执行 Phase 1：为所有记录提取 HTML outline。

    resume=True 时：读取 outline cache，跳过已完成的 sample_id。
    结果直接修改 records（原地更新），并写入 outline cache 文件。
    """
    run_dir = Path(config.runtime.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_path = run_dir / "outlines.jsonl"

    # 加载 resume 状态
    cached: dict[str, dict] = {}
    if config.runtime.resume:
        cached = load_outline_cache(cache_path)
        logger.info(f"[phase1] 从 outline cache 读取 {len(cached)} 条已完成记录")

    # 填充已缓存的记录
    to_process: list[SampleRecord] = []
    for r in all_records:
        if r.sample_id in cached:
            cached_data = cached[r.sample_id]
            # retry_timeout=True 时，让 timeout 样本重新入队
            if config.runtime.retry_on_timeout and cached_data.get("extraction_timeout", False):
                to_process.append(r)
            else:
                r.apply_outline_cache(cached_data)
        else:
            to_process.append(r)

    logger.info(
        f"[phase1] 待处理 {len(to_process)} / 总计 {len(all_records)} 条"
    )
    if not to_process:
        logger.info("[phase1] 所有记录已缓存，跳过 Phase 1")
        return

    # 检查 max_input_chars
    skipped_too_long: list[SampleRecord] = []
    processable: list[SampleRecord] = []
    for r in to_process:
        if len(r.raw_html) > config.max_input_chars:
            r.extraction_status = "failed"
            r.extraction_warnings = [
                f"raw_html too long: {len(r.raw_html)} chars > max {config.max_input_chars}"
            ]
            skipped_too_long.append(r)
            # 写入 cache（标记 failed，方便 resume 跳过）
            append_jsonl(r.to_outline_cache_dict(), cache_path)
        else:
            processable.append(r)

    if skipped_too_long:
        logger.warning(f"[phase1] 跳过 {len(skipped_too_long)} 条 raw_html 超长样本")

    if not processable:
        return

    asyncio.run(_run_async(processable, config, cache_path))
