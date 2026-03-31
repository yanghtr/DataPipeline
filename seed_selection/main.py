"""
种子 Query 筛选流水线 CLI 入口

用法：
  python -m seed_selection.main run --config configs/default.yaml [--resume] [--dry-run]
  python -m seed_selection.main run --config ... --stage embed
  python -m seed_selection.main estimate --config ...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from loguru import logger

from .config import PipelineConfig, load_config


# ── 阶段顺序（用于 resume 检查和 --stage 过滤）──────────────────────────────

STAGE_OUTPUTS: list[tuple[str, str]] = [
    ("extract",     "instruction_pool_raw.jsonl"),
    ("clean",       "instruction_pool_cleaned.jsonl"),
    ("dedup_exact", "exact_dedup_kept.jsonl"),
    ("dedup_near",  "near_dedup_kept.jsonl"),
    ("svg_filter",  "svg_filtered_kept.jsonl"),
    ("embed",       "embeddings"),          # 目录
    ("cluster",     "cluster_assignments.jsonl"),
    ("sample",      "pool_1000k.jsonl"),
]


def _output_exists(output_root: Path, filename: str) -> bool:
    p = output_root / filename
    if p.is_dir():
        return any(p.glob("shard_*.npz"))
    return p.exists() and p.stat().st_size > 0


def _log_stage(name: str) -> None:
    logger.info(f"\n{'='*60}\n阶段: {name}\n{'='*60}")


# ── 各阶段运行函数 ─────────────────────────────────────────────────────────

def run_extract(cfg: PipelineConfig, root: Path, dry_run: bool, dry_run_n: int, **_) -> None:
    from .extract import run_extract as _run
    _run(
        input_paths=[Path(p) for p in cfg.input_paths],
        output_path=root / "instruction_pool_raw.jsonl",
        dry_run_limit=dry_run_n if dry_run else None,
        num_workers=cfg.num_workers,
    )


def run_clean(cfg: PipelineConfig, root: Path, **_) -> None:
    from .clean import run_clean as _run
    _run(root / "instruction_pool_raw.jsonl", root / "instruction_pool_cleaned.jsonl")


def run_dedup_exact(cfg: PipelineConfig, root: Path, **_) -> None:
    from .dedup_exact import run_dedup_exact as _run
    _run(root / "instruction_pool_cleaned.jsonl", root / "exact_dedup_kept.jsonl")


def run_dedup_near(cfg: PipelineConfig, root: Path, **_) -> None:
    from .dedup_near import run_dedup_near as _run
    _run(
        input_path=root / "exact_dedup_kept.jsonl",
        output_path=root / "near_dedup_kept.jsonl",
        thresholds=cfg.near_dedup.thresholds,
        num_perm=cfg.near_dedup.num_perm,
        char_ngram=cfg.near_dedup.char_ngram,
        num_workers=cfg.num_workers,
    )


def run_svg_filter(cfg: PipelineConfig, root: Path, **_) -> None:
    from .svg_filter import run_svg_filter as _run
    _run(
        input_path=root / "near_dedup_kept.jsonl",
        output_path=root / "svg_filtered_kept.jsonl",
        bottom_pct=cfg.svg_filter_bottom_pct,
    )


def run_embed(cfg: PipelineConfig, root: Path, dry_run: bool, **_) -> None:
    from .embed import run_embed as _run
    _run(
        input_path=root / "svg_filtered_kept.jsonl",
        output_dir=root / "embeddings",
        model_path=cfg.embedding.model_path,
        dimension=cfg.embedding.dimension,
        batch_size=cfg.embedding.batch_size,
        device=cfg.embedding.device,
        shard_size=cfg.embedding.shard_size,
        dry_run=dry_run,
        num_devices=cfg.embedding.num_devices,
    )


def run_cluster(cfg: PipelineConfig, root: Path, **_) -> None:
    from .cluster import run_cluster as _run
    _run(
        meta_path=root / "svg_filtered_kept.jsonl",
        embed_dir=root / "embeddings",
        output_path=root / "cluster_assignments.jsonl",
        k_per_bucket=cfg.clustering.k_per_bucket,
        random_seed=cfg.clustering.random_seed,
        minibatch_size=cfg.clustering.minibatch_size,
        num_workers=cfg.num_workers,
    )


def run_sample(cfg: PipelineConfig, root: Path, **_) -> None:
    from .sample import run_sample as _run
    _run(
        input_path=root / "cluster_assignments.jsonl",
        output_dir=root,
        total_pool_size=cfg.sampling.total_pool_size,
        high_priority_size=cfg.sampling.high_priority_pool_size,
        random_seed=cfg.sampling.random_seed,
    )


STAGE_RUNNERS = {
    "extract":     run_extract,
    "clean":       run_clean,
    "dedup_exact": run_dedup_exact,
    "dedup_near":  run_dedup_near,
    "svg_filter":  run_svg_filter,
    "embed":       run_embed,
    "cluster":     run_cluster,
    "sample":      run_sample,
}


# ── estimate ──────────────────────────────────────────────────────────────────

def cmd_estimate(cfg: PipelineConfig) -> None:
    root = Path(cfg.output_root)
    print("\n=== 时间估算（CPU，Qwen3-Embedding-0.6B，batch_size=16）===\n")
    rows = [
        ("extract",     "7.7M 条", "~10 分钟"),
        ("clean",       "~7.7M",   "~3 分钟"),
        ("dedup_exact", "~7.7M",   "~5 分钟"),
        ("dedup_near",  "~4M",     "~20–40 分钟（按 domain 分批）"),
        ("svg_filter",  "~4M",     "~3 分钟（两遍读文件）"),
        ("embed",       "~3M",     "⚠️  18–22 小时（CPU）/ 1–2 小时（GPU）"),
        ("cluster",     "~3M",     "~30 分钟（MiniBatchKMeans）"),
        ("sample",      "~3M",     "~5 分钟"),
    ]
    for stage, data, est in rows:
        skip = "✓ 已完成" if _output_exists(root, dict(STAGE_OUTPUTS)[stage]) else ""
        print(f"  {stage:<14} {data:<12} {est}  {skip}")
    print()


# ── run ───────────────────────────────────────────────────────────────────────

def cmd_run(
    cfg: PipelineConfig,
    resume: bool,
    dry_run: bool,
    dry_run_n: int,
    only_stage: str | None,
) -> None:
    root = Path(cfg.output_root)
    root.mkdir(parents=True, exist_ok=True)

    if dry_run:
        logger.warning(f"[main] dry_run=True，每个文件最多读 {dry_run_n:,} 条，embed 使用零向量")

    for stage_name, output_file in STAGE_OUTPUTS:
        if only_stage and stage_name != only_stage:
            continue
        if resume and _output_exists(root, output_file):
            logger.info(f"[main] [{stage_name}] 已有输出，跳过（--resume）")
            continue

        _log_stage(stage_name)
        t0 = time.time()
        STAGE_RUNNERS[stage_name](
            cfg=cfg,
            root=root,
            dry_run=dry_run,
            dry_run_n=dry_run_n,
        )
        logger.info(f"[main] [{stage_name}] 耗时 {time.time() - t0:.1f}s")

    # 写 stats 快照
    stats_path = root / "run_stats.json"
    stats = {
        "output_root": str(root),
        "dry_run": dry_run,
        "resume": resume,
        "stage_files": {
            s: str(root / f) for s, f in STAGE_OUTPUTS
        },
    }
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    logger.info(f"[main] 流水线完成，统计写入 {stats_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m seed_selection.main",
        description="种子 Query 筛选流水线",
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="YAML 配置文件路径"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = sub.add_parser("run", help="运行流水线")
    p_run.add_argument("--resume", action="store_true", help="跳过已有输出的阶段")
    p_run.add_argument("--dry-run", action="store_true", help="快速验证（零向量 + 限制读取行数）")
    p_run.add_argument("--dry-run-n", type=int, default=1000, help="dry-run 每文件读取行数（默认 1000）")
    p_run.add_argument("--stage", type=str, default=None, help="只运行指定阶段")

    # estimate
    sub.add_parser("estimate", help="打印时间估算")

    # analyze
    sub.add_parser("analyze", help="生成质量报告和可视化图表")

    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.command == "estimate":
        cmd_estimate(cfg)
    elif args.command == "analyze":
        from .analyze import run_analyze
        run_analyze(cfg.output_root)
    elif args.command == "run":
        cmd_run(
            cfg=cfg,
            resume=args.resume,
            dry_run=args.dry_run,
            dry_run_n=args.dry_run_n,
            only_stage=args.stage,
        )


if __name__ == "__main__":
    main()
