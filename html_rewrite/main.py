"""html_rewrite 流水线 CLI 入口。

用法：
    python -m html_rewrite.main --config configs/default_local.yaml --stage preprocess
    python -m html_rewrite.main --config configs/default_local.yaml --stage rewrite
    python -m html_rewrite.main --config configs/default_local.yaml --stage all
    python -m html_rewrite.main --config configs/default_local.yaml --stage preprocess --limit 10 --no-resume
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .stage1_preprocess import run_preprocess
from .stage2_rewrite import run_rewrite


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m html_rewrite.main",
        description="HTML 改写数据生产流水线",
    )
    parser.add_argument("--config", type=Path, required=True, help="YAML 配置文件路径")
    parser.add_argument(
        "--stage",
        choices=["preprocess", "rewrite", "all"],
        default="all",
        help="运行阶段：preprocess（仅预处理）/ rewrite（仅模型改写）/ all（全流程）",
    )
    parser.add_argument("--limit", type=int, default=None, help="仅处理前 N 条（调试用）")
    parser.add_argument("--no-resume", action="store_true", help="忽略已有输出，重新处理所有条目")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.no_resume:
        cfg.resume = False

    if args.stage in ("preprocess", "all"):
        run_preprocess(cfg, limit=args.limit)

    if args.stage in ("rewrite", "all"):
        run_rewrite(cfg, limit=args.limit)


if __name__ == "__main__":
    main()
