#!/usr/bin/env python3
"""
SAgoge 蒸馏数据转换脚本

将 SAgoge 蒸馏数据集转换为统一多模态 canonical schema 格式。

两种运行模式
-----------
convert
    读取原始 JSONL → 并行渲染 SVG → 生成带 _meta 的中间 JSONL。
    中间 JSONL 包含全部记录（含失败记录），便于 filter 阶段统计分析。

filter
    读取中间 JSONL → 根据 _meta 字段过滤 → 生成最终 JSONL（不含 _meta）。

所有路径（图片目录、image-root、中间/最终 JSONL）均由调用方（shell 脚本）显式传入，
本脚本不硬编码任何数据集特定子目录。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from loguru import logger

# 将 utils 包添加到 sys.path（子进程中同样生效）
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.renderers import (  # noqa: E402
    RenderResult,
    check_all_white,
    get_svg_dimensions,
    read_image_size,
    render_svg,
)

# ──────────────────────────────────────────────────────────────────────────────
# Canonical schema builders
# ──────────────────────────────────────────────────────────────────────────────


def build_text_item(text: str) -> dict:
    """Build a canonical text item."""
    return {
        "type": "text",
        "text": {
            "type": "string",
            "format": "utf-8",
            "string": text,
        },
    }


def build_image_item(
    relative_path: str,
    image_format: str,
    width: int = 0,
    height: int = 0,
) -> dict:
    """Build a canonical image item."""
    if image_format not in ("image/jpeg", "image/png"):
        raise ValueError(f"Invalid image format: {image_format!r}")
    return {
        "type": "image",
        "image": {
            "type": "relative_path",
            "format": image_format,
            "relative_path": relative_path,
            "width": width,
            "height": height,
        },
    }


def build_sample(
    user_content: list[dict],
    assistant_text: str,
    train_mode: str,
) -> dict:
    """Build a canonical sample dict."""
    meta_prompt_map = {
        "sft": [""],
        "pretrain": ["", ""],
    }
    if train_mode not in meta_prompt_map:
        raise ValueError(f"Unknown train_mode: {train_mode!r}")
    return {
        "meta_prompt": meta_prompt_map[train_mode],
        "data": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [build_text_item(assistant_text)]},
        ],
    }


def _validate_content_item(item: dict, allow_image: bool) -> tuple[bool, Optional[str]]:
    t = item.get("type")
    if t == "text":
        txt = item.get("text", {})
        if txt.get("type") != "string":
            return False, "text.type must be 'string'"
        if txt.get("format") != "utf-8":
            return False, "text.format must be 'utf-8'"
        if not isinstance(txt.get("string"), str):
            return False, "text.string must be str"
    elif t == "image":
        if not allow_image:
            return False, "image items not allowed in assistant content"
        img = item.get("image", {})
        if img.get("type") != "relative_path":
            return False, "image.type must be 'relative_path'"
        if img.get("format") not in ("image/jpeg", "image/png"):
            return False, f"invalid image.format: {img.get('format')!r}"
        if not img.get("relative_path"):
            return False, "image.relative_path must not be empty"
        if not isinstance(img.get("width"), int) or not isinstance(img.get("height"), int):
            return False, "image width/height must be int"
    else:
        return False, f"unknown item type: {t!r}"
    return True, None


def validate_sample(sample: dict) -> tuple[bool, Optional[str]]:
    """Validate a sample against canonical schema. Returns (valid, error_msg)."""
    if "meta_prompt" not in sample or "data" not in sample:
        return False, "missing meta_prompt or data"
    if not isinstance(sample["meta_prompt"], list):
        return False, "meta_prompt must be list"
    if sample["meta_prompt"] not in ([""], ["", ""]):
        return False, "invalid meta_prompt value"
    if not isinstance(sample["data"], list) or len(sample["data"]) != 2:
        return False, "data must be list of length 2"

    user = sample["data"][0]
    if user.get("role") != "user":
        return False, "data[0].role must be 'user'"
    content = user.get("content")
    if not isinstance(content, list) or not content:
        return False, "user content must be non-empty list"
    for item in content:
        ok, msg = _validate_content_item(item, allow_image=True)
        if not ok:
            return False, f"user content item: {msg}"

    asst = sample["data"][1]
    if asst.get("role") != "assistant":
        return False, "data[1].role must be 'assistant'"
    content = asst.get("content")
    if not isinstance(content, list) or not content:
        return False, "assistant content must be non-empty list"
    for item in content:
        ok, msg = _validate_content_item(item, allow_image=False)
        if not ok:
            return False, f"assistant content item: {msg}"

    return True, None


# ──────────────────────────────────────────────────────────────────────────────
# SVG text extraction (format-specific to SAgoge distillation responses)
# ──────────────────────────────────────────────────────────────────────────────

_SVG_FENCE_RE = re.compile(r"```svg\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def extract_last_svg(response: str) -> Optional[str]:
    """
    Extract the last ```svg...``` fenced code block from the response.
    Falls back to the last bare <svg>...</svg> element if no fenced block is found.
    """
    matches = _SVG_FENCE_RE.findall(response)
    if matches:
        candidate = matches[-1].strip()
        if "<svg" in candidate.lower():
            return candidate

    # Fallback: bare SVG element
    bare = re.findall(r"<svg\b.*?</svg>", response, re.DOTALL | re.IGNORECASE)
    if bare:
        return bare[-1].strip()

    return None


# ──────────────────────────────────────────────────────────────────────────────
# Worker function (module-level so ProcessPoolExecutor can pickle it)
# ──────────────────────────────────────────────────────────────────────────────


def _process_record(args: tuple) -> dict:
    """
    Convert a single raw record into a canonical sample with _meta.

    Parameters (as a single tuple for pickling)
    --------------------------------------------
    idx        : 0-based record index (determines image filename)
    record     : dict parsed from input JSONL
    images_dir : absolute path (str) to the images output directory
    image_root : absolute path (str) used as the base for relative_path computation;
                 relative_path = os.path.relpath(image_abs_path, image_root)
    train_mode      : "sft" or "pretrain"
    shard_size      : number of images per subdirectory; 0 disables sharding
    render_timeout  : per-record cairosvg timeout in seconds; 0 disables timeout

    Returns
    -------
    dict with keys:
      idx         : int
      record      : canonical sample dict (with _meta) — always present
      skip_reason : str or None
    """
    idx, record, images_dir, image_root, train_mode, shard_size, render_timeout = args

    meta: dict = {
        "input_id": record.get("id"),
        "model": record.get("model"),
        "response": record.get("response"),
        "prompt_tokens": record.get("prompt_tokens"),
        "completion_tokens": record.get("completion_tokens"),
        "finish_reason": record.get("finish_reason"),
        "svg_format_valid": False,
        "render_success": False,
        "is_all_white": False,
        "skip_reason": None,
    }

    def _fail(reason: str) -> dict:
        meta["skip_reason"] = reason
        return {"idx": idx, "record": {"_meta": meta}, "skip_reason": reason}

    # 1. User instruction
    instruction = (record.get("instruction") or "").strip()
    if not instruction:
        return _fail("missing_instruction")

    # 2. Extract last SVG code block from the model response
    response_text = record.get("response") or ""
    svg_code = extract_last_svg(response_text)
    if not svg_code:
        return _fail("missing_svg_code")
    meta["svg_format_valid"] = True

    # 3. Determine SVG render dimensions (width/height attrs → viewBox → 0)
    svg_width, svg_height = get_svg_dimensions(svg_code)

    # 4. Render SVG to PNG with white background
    #    Shard into subdirectories to avoid large flat directories (e.g. 5000/shard).
    #    Subdir name is the zero-padded shard index; filename is the global record index.
    image_filename = f"{idx:09d}.png"
    if shard_size > 0:
        shard_idx = idx // shard_size
        image_abs_path = os.path.join(images_dir, f"{shard_idx:05d}", image_filename)
    else:
        image_abs_path = os.path.join(images_dir, image_filename)
    result: RenderResult = render_svg(svg_code, image_abs_path, svg_width, svg_height, timeout=render_timeout)
    meta["render_success"] = result.success
    if not result.success:
        meta["render_error"] = result.error
        return _fail("render_failed")

    # RenderResult already carries actual pixel dimensions from the composited image
    actual_w, actual_h = result.width, result.height

    # 5. All-white check — recorded in _meta, NOT used to skip at convert time;
    #    filtering happens in the separate filter step.
    #    Pass png_bytes directly to avoid re-reading the file from disk.
    meta["is_all_white"] = check_all_white(result.png_bytes or image_abs_path)

    # 7. Build canonical sample
    #    Order: image first, then text (per task spec section 4.3)
    relative_path = os.path.relpath(image_abs_path, image_root)
    user_content = [
        build_image_item(relative_path, "image/png", actual_w, actual_h),
        build_text_item(instruction),
    ]
    # Assistant text: wrap the last SVG in a markdown code fence
    assistant_text = f"```svg\n{svg_code}\n```"

    sample = build_sample(user_content, assistant_text, train_mode)

    valid, err = validate_sample(sample)
    if not valid:
        return _fail(f"schema_validation_failed:{err}")

    sample["_meta"] = meta
    return {"idx": idx, "record": sample, "skip_reason": None}


# ──────────────────────────────────────────────────────────────────────────────
# Convert mode
# ──────────────────────────────────────────────────────────────────────────────


def run_convert(
    input_path: Path,
    images_dir: Path,
    image_root: Path,
    inter_jsonl: Path,
    train_mode: str,
    workers: int,
    shard_size: int,
    render_timeout: int,
    log_path: Optional[Path],
) -> None:
    """Read raw JSONL, render SVGs in parallel, write intermediate JSONL with _meta.

    Args:
        input_path      : raw input JSONL file
        images_dir      : directory where rendered PNG files are written
        image_root      : base directory for computing image relative_path in canonical schema
                          (relative_path = images_dir/filename relative to image_root)
        inter_jsonl     : output path for the intermediate JSONL (with _meta)
        train_mode      : "sft" or "pretrain"
        workers         : number of parallel worker processes
        shard_size      : max images per subdirectory under images_dir; 0 disables sharding
        render_timeout  : per-record cairosvg timeout in seconds; 0 disables timeout
        log_path        : optional log file path
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    inter_jsonl.parent.mkdir(parents=True, exist_ok=True)

    _setup_logging(log_path)
    logger.info(f"[convert] input:          {input_path}")
    logger.info(f"[convert] images_dir:     {images_dir}")
    logger.info(f"[convert] image_root:     {image_root}")
    logger.info(f"[convert] inter_jsonl:    {inter_jsonl}")
    logger.info(f"[convert] train_mode:     {train_mode}")
    logger.info(f"[convert] workers:        {workers}")
    logger.info(f"[convert] shard_size:     {shard_size if shard_size > 0 else 'disabled'}")
    logger.info(f"[convert] render_timeout: {render_timeout if render_timeout > 0 else 'disabled'}s")

    # ── Read all records ──
    records: list[dict] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Line {lineno}: JSON parse error — {e}")

    total = len(records)
    logger.info(f"[convert] total records: {total}")
    if total == 0:
        logger.warning("[convert] No records to process.")
        return

    # ── Parallel processing ──
    worker_args = [
        (idx, rec, str(images_dir), str(image_root), train_mode, shard_size, render_timeout)
        for idx, rec in enumerate(records)
    ]
    results: list[Optional[dict]] = [None] * total

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(_process_record, arg): arg[0] for arg in worker_args}
        done = 0
        for future in as_completed(future_map):
            done += 1
            if done % 1000 == 0 or done == total:
                logger.info(f"[convert] progress: {done}/{total}")
            try:
                res = future.result()
                results[res["idx"]] = res
            except Exception as exc:
                idx = future_map[future]
                logger.error(f"[convert] record {idx} raised exception: {exc}")
                results[idx] = {
                    "idx": idx,
                    "record": {"_meta": {"skip_reason": "exception", "exception": str(exc)}},
                    "skip_reason": "exception",
                }

    # ── Write intermediate JSONL (ordered, all records including failed) ──
    success = skipped = 0
    skip_reasons: dict[str, int] = {}

    with open(inter_jsonl, "w", encoding="utf-8") as f:
        for res in results:
            if res is None:
                continue
            reason = res.get("skip_reason")
            if reason:
                skipped += 1
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            else:
                success += 1
            f.write(json.dumps(res["record"], ensure_ascii=False) + "\n")

    _print_stats(total, success, skipped, skip_reasons, note="(pre-filter)")
    logger.info(f"[convert] intermediate JSONL written: {inter_jsonl}")


# ──────────────────────────────────────────────────────────────────────────────
# Filter mode
# ──────────────────────────────────────────────────────────────────────────────


def run_filter(
    intermediate_jsonl: Path,
    output_jsonl: Path,
    log_path: Optional[Path],
) -> None:
    """Read intermediate JSONL, filter on _meta flags, write final JSONL without _meta.

    Args:
        intermediate_jsonl : intermediate JSONL produced by convert step (contains _meta)
        output_jsonl       : final output JSONL path (no _meta)
        log_path           : optional log file path
    """
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    _setup_logging(log_path)
    logger.info(f"[filter] input:  {intermediate_jsonl}")
    logger.info(f"[filter] output: {output_jsonl}")

    total = kept = filtered = 0
    filter_reasons: dict[str, int] = {}

    with (
        open(intermediate_jsonl, "r", encoding="utf-8") as fin,
        open(output_jsonl, "w", encoding="utf-8") as fout,
    ):
        for lineno, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Line {lineno}: JSON parse error — {e}")
                filtered += 1
                filter_reasons["json_parse_error"] = filter_reasons.get("json_parse_error", 0) + 1
                continue

            meta = sample.get("_meta", {})

            # Determine if this record should be filtered out
            reason: Optional[str] = None
            if meta.get("skip_reason"):
                reason = meta["skip_reason"]
            elif not meta.get("svg_format_valid"):
                reason = "invalid_svg_format"
            elif not meta.get("render_success"):
                reason = "render_failed"
            elif meta.get("is_all_white"):
                reason = "all_white_image"

            if reason:
                filtered += 1
                filter_reasons[reason] = filter_reasons.get(reason, 0) + 1
                logger.debug(f"[filter] line {lineno}: filtered ({reason})")
                continue

            # Write without _meta
            clean = {k: v for k, v in sample.items() if k != "_meta"}
            fout.write(json.dumps(clean, ensure_ascii=False) + "\n")
            kept += 1

    _print_stats(total, kept, filtered, filter_reasons, note="(filter pass)")
    logger.info(f"[filter] final JSONL written: {output_jsonl}")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _setup_logging(log_path: Optional[Path]) -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", colorize=True)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(str(log_path), level="DEBUG", encoding="utf-8")


def _print_stats(
    total: int,
    success: int,
    skipped: int,
    reasons: dict[str, int],
    note: str = "",
) -> None:
    header = f"Statistics {note}".strip()
    logger.info("=" * 60)
    logger.info(f"  {header}")
    logger.info(f"  Total:   {total}")
    logger.info(f"  Success: {success}")
    logger.info(f"  Skipped: {skipped}")
    if reasons:
        logger.info("  Skip reasons:")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            logger.info(f"    {reason}: {count}")
    logger.info("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SAgoge 蒸馏数据转换脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # ── convert subcommand ──
    conv = sub.add_parser(
        "convert",
        help="转换原始 JSONL → 中间 JSONL（含 _meta）",
    )
    conv.add_argument(
        "--input", "-i", required=True, type=Path,
        metavar="FILE",
        help="输入 JSONL 文件路径",
    )
    conv.add_argument(
        "--images-dir", required=True, type=Path,
        metavar="DIR",
        help="渲染后 PNG 图片的输出目录",
    )
    conv.add_argument(
        "--image-root", required=True, type=Path,
        metavar="DIR",
        help="canonical schema 中 relative_path 的计算基准目录（full_path = image_root / relative_path）",
    )
    conv.add_argument(
        "--inter-jsonl", required=True, type=Path,
        metavar="FILE",
        help="中间 JSONL 输出文件路径（含 _meta）",
    )
    conv.add_argument(
        "--train-mode", default="sft", choices=["sft", "pretrain"],
        help="训练模式（默认: sft）",
    )
    conv.add_argument(
        "--workers", type=int, default=os.cpu_count() or 4,
        metavar="N",
        help="并行 worker 数（默认: CPU 核数）",
    )
    conv.add_argument(
        "--shard-size", type=int, default=5000,
        metavar="N",
        help="每个子目录最多存放的图片数（默认: 5000；0 表示不分目录）",
    )
    conv.add_argument(
        "--render-timeout", type=int, default=60,
        metavar="SECS",
        help="单条 SVG 渲染超时秒数（默认: 60；0 表示不限时）",
    )
    conv.add_argument(
        "--log-path", type=Path, default=None,
        metavar="FILE",
        help="日志文件路径（可选）",
    )

    # ── filter subcommand ──
    filt = sub.add_parser(
        "filter",
        help="过滤中间 JSONL → 最终 JSONL（不含 _meta）",
    )
    filt.add_argument(
        "--input", "-i", required=True, type=Path,
        metavar="FILE",
        help="中间 JSONL 文件路径（convert 阶段产出）",
    )
    filt.add_argument(
        "--output-jsonl", required=True, type=Path,
        metavar="FILE",
        help="最终 JSONL 输出文件路径（不含 _meta）",
    )
    filt.add_argument(
        "--log-path", type=Path, default=None,
        metavar="FILE",
        help="日志文件路径（可选）",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "convert":
        run_convert(
            input_path=args.input,
            images_dir=args.images_dir,
            image_root=args.image_root,
            inter_jsonl=args.inter_jsonl,
            train_mode=args.train_mode,
            workers=args.workers,
            shard_size=args.shard_size,
            render_timeout=args.render_timeout,
            log_path=args.log_path,
        )
    else:
        run_filter(
            intermediate_jsonl=args.input,
            output_jsonl=args.output_jsonl,
            log_path=args.log_path,
        )


if __name__ == "__main__":
    main()
