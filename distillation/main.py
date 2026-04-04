"""
蒸馏流水线 CLI 入口。

用法：
    python -m distillation.main --config distillation/configs/default.yaml
    python -m distillation.main --config ... --limit 10
    python -m distillation.main --config ... --no-resume
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m distillation.main",
        description="SVG 蒸馏流水线",
    )
    parser.add_argument("--config", type=Path, required=True, help="YAML 配置文件路径")
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="只处理前 N 条（调试用）",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="忽略已有输出，从头处理所有记录",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.no_resume:
        cfg.resume = False

    from .distill import run_distill
    run_distill(cfg, limit=args.limit)


if __name__ == "__main__":
    main()
