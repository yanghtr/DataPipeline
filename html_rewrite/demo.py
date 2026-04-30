"""单条 debug 工具：对单条数据跑预处理或模型改写，输出结果便于检查。

用法：
    python -m html_rewrite.demo --config html_rewrite/configs/default_local.yaml --stage preprocess
    python -m html_rewrite.demo --config html_rewrite/configs/default_local.yaml --stage preprocess --index 2
    python -m html_rewrite.demo --config html_rewrite/configs/default_local.yaml --stage rewrite --index 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger

from .config import load_config
from .preprocess import preprocess
from .stage2_rewrite import _load_prompt_module, _extract_html, _extract_reasoning
from utils.api_client import call_chat_completion


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m html_rewrite.demo")
    parser.add_argument("--config", type=Path, required=True, help="YAML 配置文件路径")
    parser.add_argument(
        "--stage",
        choices=["preprocess", "rewrite"],
        default="preprocess",
        help="运行阶段",
    )
    parser.add_argument("--index", type=int, default=0, help="取输入文件第 N 条（从 0 开始）")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    if args.stage == "preprocess":
        input_path = Path(cfg.input_path)
        rec = _read_record(input_path, args.index)
        rec_id = rec.get("url") or rec.get("id", "")
        logger.info(f"[demo] 处理第 {args.index} 条，id={rec_id}")

        preprocessed_html, stats = preprocess(rec.get("html", ""), cfg)
        logger.info(f"[demo] 原始字符数={stats.original_chars:,}  清洗后={stats.cleaned_chars:,}  压缩比={stats.compression_ratio}")
        logger.info(f"[demo] 媒体替换={stats.media.replaced}  script 截断={stats.scripts.inline_truncated}  style 截断={stats.styles.inline_truncated}")
        if cfg.enable_language_filter:
            logger.info(
                f"[demo] 语言过滤已启用：可见文本={stats.visible_text_chars:,} chars，"
                "如需查看语言判定，请跑完整 Stage 1 或后续在过滤阶段查看 stats/reject 日志"
            )
        print("\n=== preprocessed_html (前 2000 chars) ===")
        print(preprocessed_html[:2000])
        print("\n=== preprocess_stats ===")
        print(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2))

    elif args.stage == "rewrite":
        input_path = Path(cfg.preprocessed_path)
        rec = _read_record(input_path, args.index)
        rec_id = rec.get("id", "")
        preprocessed_html = rec.get("preprocessed_html", "")
        logger.info(f"[demo] 改写第 {args.index} 条，id={rec_id}")

        pm = _load_prompt_module(cfg.prompt_module)
        user_content = pm.build_user_content(preprocessed_html)

        resp_data = call_chat_completion(
            url=cfg.url,
            api_key=cfg.api_key,
            model=cfg.model,
            user_content=user_content,
            system=pm.SYSTEM_PROMPT,
            timeout=cfg.timeout,
            max_retries=cfg.max_retries,
            ssl_verify=cfg.ssl_verify,
            log_user=cfg.log_user,
            extra_params=cfg.generation_params or None,
        )
        message = resp_data["choices"][0]["message"]
        raw = message.get("content", "") or ""
        reasoning = _extract_reasoning(message)
        output_html = _extract_html(raw)
        usage = resp_data.get("usage", {})
        logger.info(f"[demo] prompt_tokens={usage.get('prompt_tokens')}  completion_tokens={usage.get('completion_tokens')}")
        if reasoning:
            print("\n=== reasoning (前 2000 chars) ===")
            print(reasoning[:2000])
        print("\n=== output_html (前 2000 chars) ===")
        print(output_html[:2000])


def _read_record(path: Path, index: int) -> dict:
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line.strip())
    raise IndexError(f"文件 {path} 中不存在第 {index} 条记录")


if __name__ == "__main__":
    main()
