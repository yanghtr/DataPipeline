"""
analyze.py — 种子数据质量评估

CLI 子命令：python -m seed_selection.main analyze --config ...

输出到 {output_root}/analysis/：
  report.txt               文字漏斗统计 + 关键指标
  metrics.json             所有指标的 JSON 快照
  01_funnel.png            各阶段记录数瀑布图
  02_bucket_dist.png       六层分级柱状图（每个 bucket 三层叠加）
  03_cluster_size_hist.png 各 bucket cluster 大小分布直方图
  04_instruction_len.png   instruction 长度分布（各域高/中/低优对比）
  05_fps_priority.png      FPS 优先级分布（fps_round/fps_rank，高优应在最左侧）
  06_distance_hist.png     distance_to_centroid 分布（stage1_icon 辅助参考）
  07_source_mix.png        img2svg vs text2svg 比例饼图
  08_umap.png              [可选] embeddings UMAP 投影（需 umap-learn）
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from loguru import logger


# ── 数据加载 ──────────────────────────────────────────────────────────────────

def _read_jsonl_field(path: Path, *field_path: str) -> list:
    """从 JSONL 中提取嵌套字段值列表，跳过不存在的条目。"""
    results = []
    if not path.exists():
        return results
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                val = rec
                for key in field_path:
                    val = val[key]
                results.append(val)
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return results


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _read_jsonl_records(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


# ── 文件定义 ──────────────────────────────────────────────────────────────────

STAGE_FILES = [
    ("extract",     "instruction_pool_raw.jsonl"),
    ("clean",       "instruction_pool_cleaned.jsonl"),
    ("dedup_exact", "exact_dedup_kept.jsonl"),
    ("dedup_near",  "near_dedup_kept.jsonl"),
    ("svg_filter",  "svg_filtered_kept.jsonl"),
    ("cluster",     "cluster_assignments.jsonl"),
    ("pool_1000k",  "pool_1000k.jsonl"),
]

# (label, bucket_key, tier_name)
TIER_DEFS: list[tuple[str, str, str]] = [
    ("stage1_icon_high",          "stage1_icon",         "high"),
    ("stage1_icon_medium",        "stage1_icon",         "medium"),
    ("stage1_icon_low",           "stage1_icon",         "low"),
    ("stage2_illustration_high",   "stage2_illustration", "high"),
    ("stage2_illustration_medium", "stage2_illustration", "medium"),
    ("stage2_illustration_low",    "stage2_illustration", "low"),
]

TIER_COLORS = {"high": "coral", "medium": "steelblue", "low": "mediumseagreen"}
TIER_LABELS = {"high": "High", "medium": "Medium", "low": "Low"}


def _tier_path(root: Path, bucket: str, tier: str) -> Path:
    return root / f"{bucket}_{tier}.jsonl"


# ── 漏斗统计 ──────────────────────────────────────────────────────────────────

def compute_funnel(root: Path) -> list[tuple[str, int]]:
    funnel = []
    for stage, fname in STAGE_FILES:
        count = _count_lines(root / fname)
        funnel.append((stage, count))
    return funnel


# ── 指标计算 ──────────────────────────────────────────────────────────────────

def compute_metrics(root: Path) -> dict:
    """计算所有质量指标，返回 dict。"""
    metrics: dict = {}

    # Funnel
    funnel = compute_funnel(root)
    metrics["funnel"] = funnel
    stages = dict(funnel)

    # 去重率
    extract_n = stages.get("extract", 0)
    exact_n   = stages.get("dedup_exact", 0)
    near_n    = stages.get("dedup_near", 0)
    if extract_n > 0:
        metrics["exact_dedup_rate"] = round(1 - exact_n / extract_n, 4)
    if exact_n > 0:
        metrics["near_dedup_rate"] = round(1 - near_n / exact_n, 4)

    # 六层条数统计 & 不变量验证
    tier_counts: dict[str, dict[str, int]] = {}
    total_tier = 0
    for label, bucket, tier in TIER_DEFS:
        path = _tier_path(root, bucket, tier)
        n = _count_lines(path)
        tier_counts.setdefault(bucket, {})[tier] = n
        total_tier += n
    metrics["tier_counts"] = tier_counts
    pool_n = stages.get("pool_1000k", 0)
    metrics["tier_invariant_ok"] = (total_tier == pool_n)
    metrics["total_tier_records"] = total_tier

    # Cluster 覆盖率（从 cluster_assignments.jsonl 读取）
    cluster_path = root / "cluster_assignments.jsonl"
    if cluster_path.exists():
        domain_clusters: dict[str, set] = defaultdict(set)
        domain_counts: dict[str, int] = defaultdict(int)
        for rec in _read_jsonl_records(cluster_path):
            meta = rec.get("_meta", {})
            bk = meta.get("bucket_key", "unknown")
            cid = meta.get("cluster_id")
            if cid is not None:
                domain_clusters[bk].add(cid)
                domain_counts[bk] += 1
        metrics["cluster_total"] = {bk: len(cids) for bk, cids in domain_clusters.items()}
        metrics["record_per_domain"] = dict(domain_counts)

    # Pool 中 cluster 覆盖率
    pool_path = root / "pool_1000k.jsonl"
    if pool_path.exists():
        pool_domain_clusters: dict[str, set] = defaultdict(set)
        for rec in _read_jsonl_records(pool_path):
            meta = rec.get("_meta", {})
            bk = meta.get("bucket_key", "unknown")
            cid = meta.get("cluster_id")
            if cid is not None:
                pool_domain_clusters[bk].add(cid)
        metrics["pool_cluster_coverage"] = {}
        for bk, pool_cids in pool_domain_clusters.items():
            total_k = metrics.get("cluster_total", {}).get(bk, len(pool_cids))
            metrics["pool_cluster_coverage"][bk] = {
                "pool_clusters": len(pool_cids),
                "total_clusters": total_k,
                "coverage_pct": round(100 * len(pool_cids) / total_k, 1) if total_k else 0,
            }

    # 各层 distance、FPS 指标和 instruction 长度统计
    tier_distance: dict[str, dict[str, dict]] = {}
    tier_fps_metrics: dict[str, dict[str, dict]] = {}
    tier_instr_len: dict[str, dict[str, dict]] = {}

    for label, bucket, tier in TIER_DEFS:
        path = _tier_path(root, bucket, tier)
        records = _read_jsonl_records(path)
        if not records:
            continue

        distances = [r.get("_meta", {}).get("distance_to_centroid", 0.0) for r in records]
        lengths = [len(r.get("instruction", "")) for r in records]

        tier_distance.setdefault(bucket, {})[tier] = {
            "mean":   round(statistics.mean(distances), 6),
            "median": round(statistics.median(distances), 6),
        }
        tier_instr_len.setdefault(bucket, {})[tier] = {
            "mean": round(statistics.mean(lengths), 1),
            "std":  round(statistics.stdev(lengths) if len(lengths) > 1 else 0, 1),
            "min":  min(lengths),
            "max":  max(lengths),
        }

        # FPS 指标：stage1_icon 用 fps_round，stage2_illustration 用 fps_rank
        if bucket == "stage1_icon":
            fps_vals = [r.get("_meta", {}).get("fps_round") for r in records]
            fps_vals = [v for v in fps_vals if v is not None]
            if fps_vals:
                tier_fps_metrics.setdefault(bucket, {})[tier] = {
                    "field": "fps_round",
                    "mean":   round(statistics.mean(fps_vals), 2),
                    "median": statistics.median(fps_vals),
                }
        elif bucket == "stage2_illustration":
            fps_vals = [r.get("_meta", {}).get("fps_rank") for r in records]
            fps_vals = [v for v in fps_vals if v is not None and v > 0]
            if fps_vals:
                tier_fps_metrics.setdefault(bucket, {})[tier] = {
                    "field": "fps_rank",
                    "mean":   round(statistics.mean(fps_vals), 1),
                    "median": statistics.median(fps_vals),
                }

    metrics["tier_distance"] = tier_distance
    metrics["tier_fps_metrics"] = tier_fps_metrics
    metrics["tier_instr_len"] = tier_instr_len

    # Source mix（从 pool_1000k 统计）
    if pool_path.exists():
        sources = _read_jsonl_field(pool_path, "_meta", "source")
        if sources:
            counter = Counter(sources)
            total_s = len(sources)
            metrics["source_mix"] = {src: round(100 * n / total_s, 1)
                                     for src, n in counter.items()}

    return metrics


# ── 报告文本 ──────────────────────────────────────────────────────────────────

def generate_report_text(metrics: dict) -> str:
    lines = ["=" * 64, "种子 Query 质量报告", "=" * 64, ""]

    # Funnel
    lines.append("=== 流水线漏斗 ===")
    funnel = metrics.get("funnel", [])
    prev_n = None
    for stage, n in funnel:
        if prev_n and prev_n > 0 and stage != "pool_1000k":
            pct = f"  ({-100 * (1 - n / prev_n):.1f}%)"
        else:
            pct = ""
        lines.append(f"  {stage:<20} {n:>10,}{pct}")
        prev_n = n
    lines.append("")

    # 去重率
    exact_rate = metrics.get("exact_dedup_rate")
    near_rate  = metrics.get("near_dedup_rate")
    if exact_rate is not None:
        lines.append(f"exact dedup 去除率: {exact_rate * 100:.1f}%  "
                     f"{'✓ 正常(40-60%)' if 0.40 <= exact_rate <= 0.65 else '⚠ 偏离预期'}")
    if near_rate is not None:
        lines.append(f"near  dedup 去除率: {near_rate * 100:.1f}%  "
                     f"{'✓ 正常(3-20%)' if 0.03 <= near_rate <= 0.20 else '⚠ 偏离预期'}")
    lines.append("")

    # 六层分级计数
    lines.append("=== 六层分级计数 ===")
    tier_counts = metrics.get("tier_counts", {})
    total_tier = metrics.get("total_tier_records", 0)
    pool_n = dict(metrics.get("funnel", [])).get("pool_1000k", 0)
    for bucket in sorted(tier_counts):
        tiers = tier_counts[bucket]
        h = tiers.get("high", 0)
        m = tiers.get("medium", 0)
        lo = tiers.get("low", 0)
        lines.append(f"  {bucket}")
        lines.append(f"    高优: {h:>8,}  中优: {m:>8,}  低优: {lo:>8,}  合计: {h+m+lo:>8,}")
    ok = metrics.get("tier_invariant_ok", False)
    lines.append(f"  六层合计 {total_tier:,} vs pool_1000k {pool_n:,}: "
                 f"{'✓ 一致' if ok else '✗ 不一致！'}")
    lines.append("")

    # Cluster 覆盖率
    coverage = metrics.get("pool_cluster_coverage", {})
    if coverage:
        lines.append("=== Cluster 覆盖率（pool_1000k）===")
        for bk, info in sorted(coverage.items()):
            pct = info["coverage_pct"]
            flag = "✓" if pct >= 95 else "⚠"
            lines.append(f"  {bk:<28} {info['pool_clusters']:>5}/{info['total_clusters']:<5} "
                         f"clusters  ({pct}%) {flag}")
        lines.append("")

    # FPS 优先级层间单调性（核心多样性指标）
    tier_fps = metrics.get("tier_fps_metrics", {})
    if tier_fps:
        lines.append("=== FPS 优先级（层间单调性，越小=越多样=越高优）===")
        for bucket in sorted(tier_fps):
            tiers = tier_fps[bucket]
            field = next(iter(tiers.values()), {}).get("field", "fps_round")
            lines.append(f"  {bucket}（{field}）:")
            h_mean  = tiers.get("high",   {}).get("mean")
            m_mean  = tiers.get("medium", {}).get("mean")
            lo_mean = tiers.get("low",    {}).get("mean")
            for tier, label in [("high", "高优"), ("medium", "中优"), ("low", "低优")]:
                info = tiers.get(tier, {})
                if info:
                    lines.append(f"    {label}: 均值 {info['mean']:.2f}  "
                                 f"中位数 {info['median']:.1f}")
            if h_mean is not None and m_mean is not None and lo_mean is not None:
                monotone = h_mean < m_mean < lo_mean
                lines.append(f"    层间单调性 (高<中<低): {'✓ 正常' if monotone else '⚠ 异常'}")
        lines.append("")

    # Distance to Centroid（stage1_icon 辅助参考；stage2 恒为 0.0，跳过）
    tier_dist = metrics.get("tier_distance", {})
    if tier_dist:
        icon_dist = {b: v for b, v in tier_dist.items() if b == "stage1_icon"}
        if icon_dist:
            lines.append("=== Distance to Centroid（stage1_icon 辅助参考）===")
            for bucket in sorted(icon_dist):
                lines.append(f"  {bucket}:")
                tiers = icon_dist[bucket]
                for tier, label in [("high", "高优"), ("medium", "中优"), ("low", "低优")]:
                    info = tiers.get(tier, {})
                    if info:
                        lines.append(f"    {label}: 均值 {info['mean']:.6f}  "
                                     f"中位数 {info['median']:.6f}")
            lines.append("")

    # Instruction 长度
    tier_len = metrics.get("tier_instr_len", {})
    if tier_len:
        lines.append("=== Instruction 长度（高优层，按 bucket）===")
        for bucket in sorted(tier_len):
            info = tier_len[bucket].get("high", {})
            if info:
                std_flag = "✓ 多样性高" if info["std"] > 30 else "⚠ 长度集中"
                lines.append(f"  {bucket}: 均值 {info['mean']:.1f}  "
                             f"std {info['std']:.1f}  [{info['min']}, {info['max']}]  {std_flag}")
        lines.append("")

    # Source mix
    src = metrics.get("source_mix", {})
    if src:
        lines.append("=== Source Mix（pool_1000k）===")
        for s, pct in sorted(src.items()):
            lines.append(f"  {s}: {pct}%")
        lines.append("")

    lines.append("=" * 64)
    return "\n".join(lines)


# ── 可视化 ────────────────────────────────────────────────────────────────────

def _get_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_funnel(metrics: dict, out_dir: Path) -> None:
    plt = _get_mpl()
    funnel = metrics.get("funnel", [])
    if not funnel:
        return
    stages, counts = zip(*funnel)
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(stages, counts, color="steelblue")
    ax.set_ylabel("Records")
    ax.set_title("Pipeline Funnel")
    plt.xticks(rotation=30, ha="right")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{count:,}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "01_funnel.png", dpi=120)
    plt.close()
    logger.info("[analyze] 01_funnel.png 已写出")


def plot_bucket_dist(root: Path, out_dir: Path) -> None:
    """每个 bucket 的三层分级柱状图（grouped bars：高/中/低优）。"""
    plt = _get_mpl()

    # 收集每个 bucket 每层的条数
    bucket_tier_counts: dict[str, dict[str, int]] = {}
    for _label, bucket, tier in TIER_DEFS:
        path = _tier_path(root, bucket, tier)
        n = _count_lines(path)
        bucket_tier_counts.setdefault(bucket, {})[tier] = n

    if not bucket_tier_counts:
        return

    buckets = sorted(bucket_tier_counts)
    tiers = ["high", "medium", "low"]
    n_buckets = len(buckets)
    n_tiers = len(tiers)
    width = 0.22

    import numpy as np
    x = np.arange(n_buckets)

    fig, ax = plt.subplots(figsize=(max(8, n_buckets * 3), 5))
    for i, tier in enumerate(tiers):
        vals = [bucket_tier_counts[b].get(tier, 0) for b in buckets]
        offset = (i - n_tiers / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=TIER_LABELS[tier],
                      color=TIER_COLORS[tier])
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{v:,}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(buckets, rotation=15, ha="right")
    ax.set_ylabel("Records")
    ax.set_title("Six-Tier Distribution by Bucket")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "02_bucket_dist.png", dpi=120)
    plt.close()
    logger.info("[analyze] 02_bucket_dist.png 已写出")


def plot_cluster_size_hist(root: Path, out_dir: Path) -> None:
    plt = _get_mpl()
    cluster_path = root / "cluster_assignments.jsonl"
    if not cluster_path.exists():
        return

    domain_sizes: dict[str, list[int]] = defaultdict(list)
    for rec in _read_jsonl_records(cluster_path):
        meta = rec.get("_meta", {})
        bk = meta.get("bucket_key", "unknown")
        sz = meta.get("cluster_size")
        if sz is not None:
            domain_sizes[bk].append(sz)

    n_domains = len(domain_sizes)
    if n_domains == 0:
        return

    fig, axes = plt.subplots(1, n_domains, figsize=(6 * n_domains, 4))
    if n_domains == 1:
        axes = [axes]

    for ax, (bk, sizes) in zip(axes, sorted(domain_sizes.items())):
        ax.hist(sizes, bins=30, color="coral", edgecolor="white")
        ax.set_title(f"{bk}")
        ax.set_xlabel("Cluster size")
        ax.set_ylabel("Count")

    plt.suptitle("Cluster Size Distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "03_cluster_size_hist.png", dpi=120)
    plt.close()
    logger.info("[analyze] 03_cluster_size_hist.png 已写出")


def plot_instruction_len(root: Path, out_dir: Path) -> None:
    """instruction 长度分布：每个 bucket 一个子图，各层叠加显示。"""
    plt = _get_mpl()

    # 按 bucket 收集各层数据
    bucket_tier_lens: dict[str, dict[str, list[int]]] = {}
    for _label, bucket, tier in TIER_DEFS:
        path = _tier_path(root, bucket, tier)
        records = _read_jsonl_records(path)
        if records:
            lens = [len(r.get("instruction", "")) for r in records]
            bucket_tier_lens.setdefault(bucket, {})[tier] = lens

    if not bucket_tier_lens:
        return

    buckets = sorted(bucket_tier_lens)
    n = len(buckets)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 5), squeeze=False)

    for ax, bucket in zip(axes[0], buckets):
        tier_lens = bucket_tier_lens[bucket]
        for tier in ["high", "medium", "low"]:
            lens = tier_lens.get(tier, [])
            if lens:
                ax.hist(lens, bins=50, alpha=0.55, density=True,
                        label=TIER_LABELS[tier], color=TIER_COLORS[tier])
        ax.set_xlabel("Instruction length (chars)")
        ax.set_ylabel("Density")
        ax.set_title(f"{bucket}\nInstruction Length by Tier")
        ax.legend()

    plt.suptitle("Instruction Length Distribution (per bucket × tier)", y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "04_instruction_len.png", dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("[analyze] 04_instruction_len.png 已写出")


def plot_fps_priority(root: Path, out_dir: Path) -> None:
    """
    FPS 优先级分布图（替代 distance_to_centroid 作为多样性核心指标）。

    stage1_icon：各层 fps_round 分布（高优 round=1，中优 round=2-3，低优 round=4-7）
    stage2_illustration：各层 fps_rank 分布（高优 rank 最小 = 最多样）
    """
    plt = _get_mpl()

    # stage1_icon → fps_round；stage2_illustration → fps_rank
    field_map = {
        "stage1_icon":         "fps_round",
        "stage2_illustration": "fps_rank",
    }

    bucket_tier_vals: dict[str, dict[str, list]] = {}
    for _label, bucket, tier in TIER_DEFS:
        field = field_map.get(bucket)
        if not field:
            continue
        path = _tier_path(root, bucket, tier)
        records = _read_jsonl_records(path)
        if records:
            vals = [r.get("_meta", {}).get(field) for r in records]
            vals = [v for v in vals if v is not None and v > 0]
            if vals:
                bucket_tier_vals.setdefault(bucket, {})[tier] = vals

    if not bucket_tier_vals:
        return

    buckets = sorted(bucket_tier_vals)
    n = len(buckets)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 5), squeeze=False)

    for ax, bucket in zip(axes[0], buckets):
        field = field_map[bucket]
        tier_vals = bucket_tier_vals[bucket]
        for tier in ["high", "medium", "low"]:
            vals = tier_vals.get(tier, [])
            if vals:
                ax.hist(vals, bins=40, alpha=0.55, density=True,
                        label=TIER_LABELS[tier], color=TIER_COLORS[tier])
        ax.set_xlabel(field)
        ax.set_ylabel("Density")
        ax.set_title(f"{bucket}\nFPS Priority Dist. (High tier should peak leftmost)")
        ax.legend()

    plt.suptitle("FPS Priority Distribution by Tier (high = earliest FPS = most diverse)", y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "05_fps_priority.png", dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("[analyze] 05_fps_priority.png 已写出")


def plot_distance_hist(root: Path, out_dir: Path) -> None:
    """distance_to_centroid 分布（仅 stage1_icon；stage2 恒为 0.0，跳过）。"""
    plt = _get_mpl()

    bucket_tier_dists: dict[str, dict[str, list[float]]] = {}
    for _label, bucket, tier in TIER_DEFS:
        if bucket != "stage1_icon":  # stage2 distance 恒为 0.0，无意义
            continue
        path = _tier_path(root, bucket, tier)
        records = _read_jsonl_records(path)
        if records:
            dists = [r.get("_meta", {}).get("distance_to_centroid", 0.0) for r in records]
            bucket_tier_dists.setdefault(bucket, {})[tier] = dists

    if not bucket_tier_dists:
        return

    buckets = sorted(bucket_tier_dists)
    n = len(buckets)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 5), squeeze=False)

    for ax, bucket in zip(axes[0], buckets):
        tier_dists = bucket_tier_dists[bucket]
        for tier in ["high", "medium", "low"]:
            dists = tier_dists.get(tier, [])
            if dists:
                ax.hist(dists, bins=50, alpha=0.55, density=True,
                        label=TIER_LABELS[tier], color=TIER_COLORS[tier])
        ax.set_xlabel("Distance to centroid")
        ax.set_ylabel("Density")
        ax.set_title(f"{bucket}\nReference (High tier should peak leftmost)")
        ax.legend()

    plt.suptitle("Distance to Centroid — stage1_icon only (auxiliary metric)", y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "06_distance_hist.png", dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("[analyze] 06_distance_hist.png 已写出")


def plot_source_mix(root: Path, out_dir: Path) -> None:
    plt = _get_mpl()
    pool_path = root / "pool_1000k.jsonl"
    sources = _read_jsonl_field(pool_path, "_meta", "source")
    if not sources:
        return
    counter = Counter(sources)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(counter.values(), labels=counter.keys(), autopct="%1.1f%%",
           colors=["steelblue", "coral"])
    ax.set_title("Source Mix (pool_1000k)")
    plt.tight_layout()
    plt.savefig(out_dir / "07_source_mix.png", dpi=120)
    plt.close()
    logger.info("[analyze] 07_source_mix.png 已写出")


def plot_umap(root: Path, out_dir: Path, sample_n: int = 50_000) -> None:
    """可选：UMAP 投影（需要 umap-learn 和 embeddings 目录）。按 bucket 着色。"""
    try:
        import umap
    except ImportError:
        logger.info("[analyze] umap-learn 未安装，跳过 07_umap.png")
        return

    embed_dir = root / "embeddings"
    if not embed_dir.exists():
        return

    from .embed import load_all_embeddings
    try:
        all_ids, all_embs = load_all_embeddings(embed_dir)
    except FileNotFoundError:
        return

    # 从 cluster_assignments 获取 bucket_key
    id_to_bucket: dict[str, str] = {}
    cluster_path = root / "cluster_assignments.jsonl"
    for rec in _read_jsonl_records(cluster_path):
        meta = rec.get("_meta", {})
        id_to_bucket[meta.get("id", "")] = meta.get("bucket_key", "unknown")

    import numpy as np
    import random
    rng = random.Random(42)
    n = min(sample_n, len(all_ids))
    indices = rng.sample(range(len(all_ids)), n)
    sampled_embs = all_embs[indices]
    sampled_labels = [id_to_bucket.get(all_ids[i], "unknown") for i in indices]

    logger.info(f"[analyze] UMAP 降维 {n} 条 embeddings ...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_jobs=4)
    coords = reducer.fit_transform(sampled_embs)

    plt = _get_mpl()
    import matplotlib.cm as cm
    unique_labels = sorted(set(sampled_labels))
    colors = cm.tab10.colors
    label_color = {lbl: colors[i % len(colors)] for i, lbl in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(10, 8))
    for lbl in unique_labels:
        mask = [l == lbl for l in sampled_labels]
        xs = [coords[i, 0] for i, m in enumerate(mask) if m]
        ys = [coords[i, 1] for i, m in enumerate(mask) if m]
        ax.scatter(xs, ys, s=1, alpha=0.3, c=[label_color[lbl]], label=lbl)
    ax.legend(markerscale=8)
    ax.set_title(f"UMAP of {n:,} Instruction Embeddings (colored by bucket)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(out_dir / "08_umap.png", dpi=120)
    plt.close()
    logger.info("[analyze] 08_umap.png 已写出")


# ── 入口 ──────────────────────────────────────────────────────────────────────

def run_analyze(output_root: str) -> None:
    """生成完整质量报告。"""
    root = Path(output_root)
    out_dir = root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[analyze] 分析 {root} ...")

    metrics = compute_metrics(root)

    report = generate_report_text(metrics)
    report_path = out_dir / "report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)

    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    try:
        plot_funnel(metrics, out_dir)
        plot_bucket_dist(root, out_dir)
        plot_cluster_size_hist(root, out_dir)
        plot_instruction_len(root, out_dir)
        plot_fps_priority(root, out_dir)
        plot_distance_hist(root, out_dir)
        plot_source_mix(root, out_dir)
        plot_umap(root, out_dir)
    except ImportError as e:
        logger.warning(f"[analyze] 绘图依赖缺失，跳过部分图表: {e}")

    logger.info(f"[analyze] 报告已写入 {out_dir}")
