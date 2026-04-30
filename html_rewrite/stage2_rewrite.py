"""
Stage 2：批量模型改写引擎。

输入：preprocessed JSONL（Stage 1 输出，已有序）
输出：output JSONL（与 Stage 1 输出严格同序，可逐行对应比较）

顺序保证策略：与 Stage 1 相同 —— 并发计算后按原始索引排序，原子写出。
"""

from __future__ import annotations

import importlib
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import ModuleType

from loguru import logger

from utils.api_client import call_chat_completion
from .config import HtmlRewriteConfig
from .run_layout import (
    build_manifest_payload,
    build_stage2_aggregate_summary,
    ensure_manifest,
    stage2_shard_paths,
)

# 提取模型输出中的 HTML（```html ... ``` 代码块或裸 HTML）
_HTML_BLOCK_RE = re.compile(r"```(?:html)?\s*(<!DOCTYPE.*?</html>)\s*```", re.DOTALL | re.IGNORECASE)


def _load_prompt_module(name: str) -> ModuleType:
    """动态加载 html_rewrite/prompts/<name>.py，校验接口约定。"""
    module = importlib.import_module(f"html_rewrite.prompts.{name}")
    for attr in ("SYSTEM_PROMPT", "build_user_content"):
        if not hasattr(module, attr):
            raise AttributeError(
                f"prompt 模块 '{name}' 缺少必要属性 '{attr}'，"
                "请确保模块导出 SYSTEM_PROMPT: str 和 build_user_content(preprocessed_html) -> list[dict]"
            )
    return module


def _extract_html(content: str) -> str:
    """从模型输出中提取 HTML。优先匹配代码块，否则直接返回。"""
    m = _HTML_BLOCK_RE.search(content)
    if m:
        return m.group(1).strip()
    if "<!DOCTYPE" in content or "<html" in content:
        return content.strip()
    return content.strip()


def _extract_reasoning(message: dict) -> str | None:
    """兼容 vLLM 新旧字段名，提取 reasoning 文本。"""
    reasoning = message.get("reasoning")
    if reasoning:
        return reasoning
    reasoning_content = message.get("reasoning_content")
    if reasoning_content:
        return reasoning_content
    return None


def run_rewrite(cfg: HtmlRewriteConfig, limit: int | None = None) -> None:
    """
    读取预处理结果，并行调用模型，按原始顺序写入 Stage 2 输出。

    Args:
        cfg:   流水线配置
        limit: 仅处理前 N 条（调试用）
    """
    pm = _load_prompt_module(cfg.prompt_module)
    system_prompt: str = pm.SYSTEM_PROMPT
    build_user_content = pm.build_user_content

    input_mode, shard_specs = cfg.resolve_input_shards()
    if cfg.use_sharded_run():
        run_root_dir = Path(cfg.run_root_dir)
        manifest_path = run_root_dir / "manifest.json"
        if not manifest_path.exists():
            raise RuntimeError("Shard run manifest not found. Please run Stage 1 first.")
        expected_manifest = build_manifest_payload(
            input_mode=input_mode,
            input_dir=cfg.input_dir,
            input_filename_template=cfg.input_filename_template,
            input_start_index=cfg.input_start_index,
            input_end_index_exclusive=cfg.input_end_index_exclusive,
            output_shard_name_template=cfg.output_shard_name_template,
            shard_specs=shard_specs,
        )
        ensure_manifest(run_root_dir, expected_manifest)

        remaining_limit = limit
        for shard in shard_specs:
            if remaining_limit is not None and remaining_limit <= 0:
                break
            shard_paths = stage2_shard_paths(run_root_dir, shard.shard_name)
            if not shard_paths.input_path.exists():
                logger.info(f"[rewrite] 跳过 {shard.shard_name}：Stage 1 keep 输出不存在")
                continue
            records = _read_records(shard_paths.input_path, remaining_limit)
            _run_rewrite_records(
                records=records,
                cfg=cfg,
                output_path=shard_paths.output_path,
                call_log_path=shard_paths.call_log_path,
                source_label=f"{shard_paths.input_path} -> stage2/{shard.shard_name}",
                system_prompt=system_prompt,
                build_user_content=build_user_content,
            )
            if remaining_limit is not None:
                remaining_limit -= len(records)

        agg_path = build_stage2_aggregate_summary(run_root_dir, shard_specs)
        logger.info(f"[rewrite] aggregate summary 已写出：{agg_path}")
        return

    input_path = Path(cfg.preprocessed_path)
    output_path = Path(cfg.output_path)
    call_log_path = Path(cfg.call_log_path)

    records = _read_records(input_path)
    if limit is not None:
        records = records[:limit]

    logger.info(f"[rewrite] 共 {len(records):,} 条输入，来自 {input_path}")
    _run_rewrite_records(
        records=records,
        cfg=cfg,
        output_path=output_path,
        call_log_path=call_log_path,
        source_label=str(input_path),
        system_prompt=system_prompt,
        build_user_content=build_user_content,
    )


def _run_rewrite_records(
    *,
    records: list[dict],
    cfg: HtmlRewriteConfig,
    output_path: Path,
    call_log_path: Path,
    source_label: str,
    system_prompt: str,
    build_user_content,
) -> None:
    # ── 2. Resume：读取已完成结果（id → result），用于后续合并排序 ───────────
    existing: dict[str, dict] = {}
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
            logger.info(f"[rewrite] resume：已完成 {len(existing):,} 条，将跳过")

    done_ids = set(existing.keys())
    indexed_todo = [(i, rec) for i, rec in enumerate(records) if rec.get("id") not in done_ids]

    logger.info(
        f"[rewrite] 待处理 {len(indexed_todo):,} 条，workers={cfg.num_workers}，输出={output_path}，来源={source_label}"
    )

    # ── 3. 并行调用模型，结果收集进 dict ─────────────────────────────────────
    new_results: dict[int, dict] = {}
    collect_lock = threading.Lock()
    ok_count = 0
    err_count = 0
    count_lock = threading.Lock()

    call_log_path.parent.mkdir(parents=True, exist_ok=True)

    def process_one(orig_idx: int, rec: dict) -> None:
        nonlocal ok_count, err_count
        rec_id = rec.get("id", "")
        preprocessed_html = rec.get("preprocessed_html", "")

        try:
            user_content = build_user_content(preprocessed_html)
            resp_data = call_chat_completion(
                url=cfg.url,
                api_key=cfg.api_key,
                model=cfg.model,
                user_content=user_content,
                system=system_prompt,
                timeout=cfg.timeout,
                max_retries=cfg.max_retries,
                ssl_verify=cfg.ssl_verify,
                log_user=cfg.log_user,
                result_log_path=call_log_path,
                extra_params=cfg.generation_params or None,
            )
            message: dict = resp_data["choices"][0]["message"]
            response_text = message.get("content", "") or ""
            reasoning_text = _extract_reasoning(message)
            output_html = _extract_html(response_text)
            usage: dict = resp_data.get("usage", {})

            result = {
                "id": rec_id,
                "_meta": rec.get("_meta", {}),
                "preprocessed_html": preprocessed_html,
                "response": response_text,
                "reasoning": reasoning_text,
                "output_html": output_html,
                "preprocess_stats": rec.get("preprocess_stats", {}),
                "model": resp_data.get("model", cfg.model),
                "usage": usage,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "finish_reason": resp_data["choices"][0].get("finish_reason"),
            }
            with collect_lock:
                new_results[orig_idx] = result
            with count_lock:
                ok_count += 1

        except Exception as exc:
            logger.warning(f"[rewrite] 失败 id={rec_id}: {exc}")
            with count_lock:
                err_count += 1

        total = ok_count + err_count
        if total % 100 == 0:
            logger.info(f"[rewrite] 进度 {total:,}/{len(indexed_todo):,}  ok={ok_count:,}  error={err_count:,}")

    if indexed_todo:
        with ThreadPoolExecutor(max_workers=cfg.num_workers) as exe:
            futures = [exe.submit(process_one, idx, rec) for idx, rec in indexed_todo]
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as exc:
                    logger.error(f"[rewrite] worker 未捕获异常: {exc}")

    # ── 4. 合并已有 + 新结果，按原始索引排序后原子写出 ───────────────────────
    existing_by_idx: dict[int, dict] = {}
    for i, rec in enumerate(records):
        rec_id = rec.get("id", "")
        if rec_id in existing:
            existing_by_idx[i] = existing[rec_id]

    all_by_idx: dict[int, dict] = {**existing_by_idx, **new_results}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = output_path.with_suffix(".tmp")

    with open(tmp_out, "w", encoding="utf-8") as f:
        for idx in sorted(all_by_idx.keys()):
            f.write(json.dumps(all_by_idx[idx], ensure_ascii=False) + "\n")

    os.replace(tmp_out, output_path)

    logger.info(
        f"[rewrite] 完成：ok={ok_count:,}  error={err_count:,}  "
        f"输出={output_path}（共 {len(all_by_idx):,} 条，按原始顺序）"
    )


def _read_records(path: Path, limit: int | None = None) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records
