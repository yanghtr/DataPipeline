"""html_rewrite_cot 流水线 CLI 入口。

用法：
    python -m html_rewrite_cot.main --config configs/my.yaml
    python -m html_rewrite_cot.main --config configs/my.yaml --phase phase1
    python -m html_rewrite_cot.main --config configs/my.yaml --phase phase2
    python -m html_rewrite_cot.main --config configs/my.yaml --limit 5 --no-resume
"""

from __future__ import annotations

import argparse
from pathlib import Path

from html_rewrite_cot.config import load_config
from html_rewrite_cot.runner import run


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m html_rewrite_cot.main",
        description="Image-to-HTML CoT 数据构造流水线",
    )
    parser.add_argument("--config", type=Path, required=True, help="YAML 配置文件路径")
    parser.add_argument(
        "--phase",
        choices=["phase1", "phase2", "all"],
        default="all",
        help="运行阶段：phase1（仅提取 outline）/ phase2（仅 VLM 生成）/ all（完整流水线）",
    )
    parser.add_argument("--limit", type=int, default=None, help="仅处理前 N 条（调试用）")
    parser.add_argument("--no-resume", action="store_true", help="禁用 resume，重新处理所有条目")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.no_resume:
        cfg.runtime.resume = False

    run(cfg, limit=args.limit, phase=args.phase)


if __name__ == "__main__":
    main()
