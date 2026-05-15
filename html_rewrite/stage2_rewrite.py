"""
Stage 2：批量模型改写引擎。

输入：preprocessed JSONL（Stage 1 输出）
输出：output JSONL（append-only，无需与 Stage 1 同序）

输出策略：worker 完成一条就追加一条到最终 output.jsonl；resume 基于 record_uid 去重。
"""

from __future__ import annotations

import importlib
import itertools
import json
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
_NGINX_SERVER_RE = re.compile(r"^\s*server\s+([A-Za-z0-9_.-]+:\d+)(?:\s+[^;]*)?;", re.MULTILINE)


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


def _resolve_api_urls(cfg: HtmlRewriteConfig) -> list[str]:
    if cfg.backend_urls and cfg.backend_urls_from_nginx_conf:
        raise ValueError("Use either `backend_urls` or `backend_urls_from_nginx_conf`, not both.")

    if cfg.backend_urls:
        return list(dict.fromkeys(cfg.backend_urls))

    if cfg.backend_urls_from_nginx_conf:
        conf_path = Path(cfg.backend_urls_from_nginx_conf)
        text = conf_path.read_text(encoding="utf-8")
        path = cfg.backend_url_path or "/v1/chat/completions"
        if not path.startswith("/"):
            path = "/" + path

        urls: list[str] = []
        for target in _NGINX_SERVER_RE.findall(text):
            urls.append(f"http://{target}{path}")

        urls = list(dict.fromkeys(urls))
        if not urls:
            raise ValueError(f"No vLLM backend `server host:port;` entries found in {conf_path}")
        return urls

    return [cfg.url]


def _effective_num_workers(cfg: HtmlRewriteConfig, api_url_count: int) -> int:
    if cfg.num_workers_per_backend is not None:
        if cfg.num_workers_per_backend <= 0:
            raise ValueError("`num_workers_per_backend` must be positive when set.")
        return api_url_count * cfg.num_workers_per_backend
    return cfg.num_workers


def run_rewrite(cfg: HtmlRewriteConfig, limit: int | None = None) -> None:
    """
    读取预处理结果，并行调用模型，append-only 写入 Stage 2 输出。

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
    # ── 2. Resume：读取已完成结果（record_uid → result），用于跳过已完成样本 ──
    if not cfg.resume:
        if output_path.exists():
            output_path.unlink()
        if call_log_path.exists():
            call_log_path.unlink()

    existing = _load_existing_results(output_path, cfg.resume)
    if existing:
        logger.info(f"[rewrite] resume：已完成 {len(existing):,} 条，将跳过")

    done_ids = set(existing.keys())
    indexed_todo = [(i, rec) for i, rec in enumerate(records) if not _rewrite_record_is_done(rec, done_ids)]
    api_urls = _resolve_api_urls(cfg)
    effective_workers = _effective_num_workers(cfg, len(api_urls))
    url_cycle = itertools.cycle(api_urls)
    url_lock = threading.Lock()
    url_semaphores = (
        {api_url: threading.Semaphore(cfg.num_workers_per_backend) for api_url in api_urls}
        if cfg.num_workers_per_backend is not None
        else {}
    )

    logger.info(
        f"[rewrite] 待处理 {len(indexed_todo):,} 条，workers={effective_workers}，"
        f"api_backends={len(api_urls)}，输出={output_path}，来源={source_label}"
    )

    # ── 3. 并行调用模型，成功结果直接追加到最终 output.jsonl ─────────────────
    output_lock = threading.Lock()
    ok_count = 0
    err_count = 0
    count_lock = threading.Lock()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    call_log_path.parent.mkdir(parents=True, exist_ok=True)

    def next_api_url() -> str:
        with url_lock:
            return next(url_cycle)

    def call_one_backend(api_url: str, user_content: list[dict]) -> dict:
        return call_chat_completion(
            url=api_url,
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

    def process_one(orig_idx: int, rec: dict) -> None:
        nonlocal ok_count, err_count
        rec_id = rec.get("id", "")
        preprocessed_html = rec.get("preprocessed_html", "")
        api_url = next_api_url()

        try:
            user_content = build_user_content(preprocessed_html)
            semaphore = url_semaphores.get(api_url)
            if semaphore is None:
                resp_data = call_one_backend(api_url, user_content)
            else:
                with semaphore:
                    resp_data = call_one_backend(api_url, user_content)
            message: dict = resp_data["choices"][0]["message"]
            response_text = message.get("content", "") or ""
            reasoning_text = _extract_reasoning(message)
            output_html = _extract_html(response_text)
            usage: dict = resp_data.get("usage", {})

            result = {
                "record_uid": rec.get("record_uid") or _fallback_record_uid(rec, orig_idx),
                "id": rec_id,
                "input_shard": rec.get("input_shard") or rec.get("_source_shard", ""),
                "input_index": rec.get("input_index", rec.get("_input_index", orig_idx)),
                "source_input_path": rec.get("source_input_path", rec.get("_source_input_path", "")),
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
            _append_jsonl_record(output_path, result, output_lock)
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
        with ThreadPoolExecutor(max_workers=effective_workers) as exe:
            futures = [exe.submit(process_one, idx, rec) for idx, rec in indexed_todo]
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as exc:
                    logger.error(f"[rewrite] worker 未捕获异常: {exc}")

    logger.info(
        f"[rewrite] 完成：ok={ok_count:,}  error={err_count:,}  "
        f"输出={output_path}（累计 {len(existing) + ok_count:,} 条，append-only，无序）"
    )


def _load_existing_results(output_path: Path, resume: bool) -> dict[str, dict]:
    existing: dict[str, dict] = {}
    if not resume:
        return existing

    if not output_path.exists():
        return existing

    with open(output_path, encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line.strip())
            except Exception:
                continue
            record_key = record.get("record_uid") or record.get("id")
            if record_key:
                existing[record_key] = record
    return existing


def _append_jsonl_record(path: Path, record: dict, lock: threading.Lock) -> None:
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _rewrite_record_is_done(rec: dict, existing_keys: set[str]) -> bool:
    record_uid = rec.get("record_uid") or rec.get("_record_uid")
    raw_id = rec.get("id")
    return (record_uid in existing_keys if record_uid else False) or (raw_id in existing_keys if raw_id else False)


def _fallback_record_uid(rec: dict, input_index: int) -> str:
    if rec.get("_record_uid"):
        return rec["_record_uid"]
    shard_name = rec.get("input_shard") or rec.get("_source_shard", "part-00000")
    actual_index = rec.get("input_index", rec.get("_input_index", input_index))
    return f"{shard_name}:{int(actual_index):08d}"


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
