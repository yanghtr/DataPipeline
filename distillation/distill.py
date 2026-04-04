"""
Step 核心：并行调用 API 进行 SVG 蒸馏，支持 resume / 流式写出 / 错误记录。

输入：  种子 JSONL（含 instruction 字段和 _meta.id）
输出：  蒸馏结果 JSONL，每行一条：
        {"id": ..., "instruction": ..., "status": "ok"|"error",
         "response": "<svg>...</svg>"|null, "error": null|"..."}
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from loguru import logger

from utils.api_client import call_chat_completion
from .config import DistillConfig
from .prompt import SVG_SYSTEM_PROMPT, build_svg_user_content


def run_distill(
    cfg: DistillConfig,
    limit: int | None = None,
) -> None:
    """
    读取 cfg.input_path，并行调用 API，结果流式写入 cfg.output_path。

    Args:
        cfg:   蒸馏配置
        limit: 仅处理前 N 条（调试用）
    """
    input_path = Path(cfg.input_path)
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

    logger.info(f"[distill] 共 {len(records):,} 条输入，来自 {input_path}")

    # ── 2. Resume：收集已完成的 ID ──────────────────────────────────────────
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
            logger.info(f"[distill] resume：已完成 {len(done_ids):,} 条，将跳过")

    todo = [r for r in records if _get_id(r) not in done_ids]
    logger.info(
        f"[distill] 待处理 {len(todo):,} 条，"
        f"workers={cfg.num_workers}，输出={output_path}"
    )

    if not todo:
        logger.info("[distill] 全部已完成，退出")
        return

    # ── 3. 并行处理 ──────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()
    ok_count = 0
    err_count = 0
    count_lock = threading.Lock()

    def process_one(rec: dict) -> None:
        nonlocal ok_count, err_count
        rec_id = _get_id(rec)
        instruction = rec.get("instruction", "")

        result: dict = {"id": rec_id, "instruction": instruction}
        try:
            response = call_chat_completion(
                url=cfg.url,
                api_key=cfg.api_key,
                model=cfg.model,
                user_content=build_svg_user_content(instruction),
                system=SVG_SYSTEM_PROMPT,
                timeout=cfg.timeout,
                max_retries=cfg.max_retries,
                ssl_verify=cfg.ssl_verify,
                log_user=cfg.log_user,
                result_log_path=call_log_path,
            )
            result.update(status="ok", response=response, error=None)
            with count_lock:
                ok_count += 1
        except Exception as exc:
            result.update(status="error", response=None, error=str(exc))
            logger.warning(f"[distill] 失败 id={rec_id}: {exc}")
            with count_lock:
                err_count += 1

        # 流式写出（线程安全）
        with write_lock:
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        # 每 100 条打一次进度
        total = ok_count + err_count
        if total % 100 == 0:
            logger.info(
                f"[distill] 进度 {total:,}/{len(todo):,}"
                f"  ok={ok_count:,}  error={err_count:,}"
            )

    with ThreadPoolExecutor(max_workers=cfg.num_workers) as exe:
        futures = [exe.submit(process_one, rec) for rec in todo]
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as exc:
                logger.error(f"[distill] worker 未捕获异常: {exc}")

    logger.info(
        f"[distill] 完成：ok={ok_count:,}  error={err_count:,}  "
        f"输出={output_path}"
    )


def _get_id(rec: dict) -> str:
    """从记录中提取唯一 ID（兼容 _meta.id 和顶层 id 字段）。"""
    return rec.get("_meta", {}).get("id") or rec.get("id", "")
