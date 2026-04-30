"""
Stage 2：批量模型改写引擎。

输入：preprocessed JSONL（Stage 1 输出）
输出：最终 output JSONL（含 meta + preprocessed_html + output_html + stats）
"""

from __future__ import annotations

import importlib
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import ModuleType

from loguru import logger

from utils.api_client import call_chat_completion
from .config import HtmlRewriteConfig

# 提取模型输出中 HTML 代码块（```html ... ``` 或裸 HTML）
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
    # 如果包含 <!DOCTYPE，直接返回整个 content
    if "<!DOCTYPE" in content or "<html" in content:
        return content.strip()
    return content.strip()


def run_rewrite(cfg: HtmlRewriteConfig, limit: int | None = None) -> None:
    """
    读取 cfg.preprocessed_path，并行调用模型，结果写入 cfg.output_path。

    Args:
        cfg:   流水线配置
        limit: 仅处理前 N 条（调试用）
    """
    pm = _load_prompt_module(cfg.prompt_module)
    system_prompt: str = pm.SYSTEM_PROMPT
    build_user_content = pm.build_user_content

    input_path = Path(cfg.preprocessed_path)
    output_path = Path(cfg.output_path)
    call_log_path = Path(cfg.call_log_path)

    # ── 1. 读取输入 ──────────────────────────────────────────────────────────
    records: list[dict] = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if limit is not None:
        records = records[:limit]

    logger.info(f"[rewrite] 共 {len(records):,} 条输入，来自 {input_path}")

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
            logger.info(f"[rewrite] resume：已完成 {len(done_ids):,} 条，将跳过")

    todo = [r for r in records if r.get("id") not in done_ids]
    logger.info(f"[rewrite] 待处理 {len(todo):,} 条，workers={cfg.num_workers}，输出={output_path}")

    if not todo:
        logger.info("[rewrite] 全部已完成，退出")
        return

    # ── 3. 并行处理 ──────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    call_log_path.parent.mkdir(parents=True, exist_ok=True)

    write_lock = threading.Lock()
    ok_count = 0
    err_count = 0
    count_lock = threading.Lock()

    def process_one(rec: dict) -> None:
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
            raw_content: str = message["content"]
            output_html = _extract_html(raw_content)
            usage: dict = resp_data.get("usage", {})

            result = {
                "id": rec_id,
                "_meta": rec.get("_meta", {}),
                "preprocessed_html": preprocessed_html,
                "output_html": output_html,
                "preprocess_stats": rec.get("preprocess_stats", {}),
                "model": resp_data.get("model", cfg.model),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "finish_reason": resp_data["choices"][0].get("finish_reason"),
            }

            with write_lock:
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

            with count_lock:
                ok_count += 1

        except Exception as exc:
            logger.warning(f"[rewrite] 失败 id={rec_id}: {exc}")
            with count_lock:
                err_count += 1

        total = ok_count + err_count
        if total % 100 == 0:
            logger.info(f"[rewrite] 进度 {total:,}/{len(todo):,}  ok={ok_count:,}  error={err_count:,}")

    with ThreadPoolExecutor(max_workers=cfg.num_workers) as exe:
        futures = [exe.submit(process_one, rec) for rec in todo]
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as exc:
                logger.error(f"[rewrite] worker 未捕获异常: {exc}")

    logger.info(f"[rewrite] 完成：ok={ok_count:,}  error={err_count:,}  输出={output_path}")
