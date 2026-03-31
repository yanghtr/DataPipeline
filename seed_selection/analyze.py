"""
analyze.py — 种子数据质量评估

CLI 子命令：python -m seed_selection.main analyze --config ...

输出到 {output_root}/analysis/：
  report.txt               文字漏斗统计 + 关键指标
  01_funnel.png            各阶段记录数瀑布图
  02_bucket_dist.png       bucket 分布柱状图
  03_cluster_size_hist.png 各 bucket cluster 大小分布直方图
  04_instruction_len.png   instruction 长度分布（anneal vs hp 对比）
  05_distance_hist.png     distance_to_centroid 分布（anneal vs hp 对比）
  06_source_mix.png        img2svg vs text2svg 比例饼图
  07_umap.png              [可选] embeddings UMAP 投影（需 umap-learn）
"""

from __future__ import annotations

import json
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


# ── 漏斗统计 ──────────────────────────────────────────────────────────────────

STAGE_FILES = [
    ("extract",     "instruction_pool_raw.jsonl"),
    ("clean",       "instruction_pool_cleaned.jsonl"),
    ("dedup_exact", "exact_dedup_kept.jsonl"),
    ("dedup_near",  "near_dedup_kept.jsonl"),
    ("svg_filter",  "svg_filtered_kept.jsonl"),
    ("cluster",     "cluster_assignments.jsonl"),
]

POOL_FILES = [
    ("pool_1000k",       "pool_1000k.jsonl"),
    ("high_priority",    "high_priority_pool.jsonl"),
    ("anneal",           "anneal_pool.jsonl"),
]


def compute_funnel(root: Path) -> list[tuple[str, int]]:
    funnel = []
    for stage, fname in STAGE_FILES:
        count = _count_lines(root / fname)
        funnel.append((stage, count))
    for name, fname in POOL_FILES:
        count = _count_lines(root / fname)
        funnel.append((name, count))
    return funnel


# ── 指标计算 ──────────────────────────────────────────────────────────────────

def compute_metrics(root: Path) -> dict:
    """计算所有质量指标，返回 dict。"""
    metrics: dict = {}

    # Funnel
    funnel = compute_funnel(root)
    metrics["funnel"] = funnel

    # 去重率
    stages = dict(funnel)
    extract_n = stages.get("extract", 0)
    exact_n   = stages.get("dedup_exact", 0)
    near_n    = stages.get("dedup_near", 0)
    if extract_n > 0:
        metrics["exact_dedup_rate"] = round(1 - exact_n / extract_n, 4)
    if exact_n > 0:
        metrics["near_dedup_rate"] = round(1 - near_n / exact_n, 4)

    # Pool 不变量验证
    pool_n = stages.get("pool_1000k", 0)
    hp_n   = stages.get("high_priority", 0)
    anneal_n = stages.get("anneal", 0)
    metrics["pool_invariant_ok"] = (pool_n == hp_n + anneal_n)

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

    # Distance 统计（anneal vs hp）
    for pool_name, fname in [("high_priority", "high_priority_pool.jsonl"),
                              ("anneal", "anneal_pool.jsonl")]:
        records = _read_jsonl_records(root / fname)
        if records:
            distances = [r.get("_meta", {}).get("distance_to_centroid", 0) for r in records]
            lengths = [len(r.get("instruction", "")) for r in records]
            import statistics
            metrics[f"{pool_name}_distance"] = {
                "mean": round(statistics.mean(distances), 6),
                "median": round(statistics.median(distances), 6),
            }
            metrics[f"{pool_name}_instr_len"] = {
                "mean": round(statistics.mean(lengths), 1),
                "std": round(statistics.stdev(lengths) if len(lengths) > 1 else 0, 1),
                "min": min(lengths),
                "max": max(lengths),
            }

    # Source mix（从 pool_1000k 统计）
    sources = _read_jsonl_field(pool_path, "_meta", "source")
    if sources:
        counter = Counter(sources)
        total_s = len(sources)
        metrics["source_mix"] = {src: round(100 * n / total_s, 1) for src, n in counter.items()}

    return metrics


# ── 报告文本 ──────────────────────────────────────────────────────────────────

def generate_report_text(metrics: dict) -> str:
    lines = ["=" * 60, "种子 Query 质量报告", "=" * 60, ""]

    # Funnel
    lines.append("=== 流水线漏斗 ===")
    funnel = metrics.get("funnel", [])
    prev_n = None
    for stage, n in funnel:
        if prev_n and prev_n > 0 and stage not in ("pool_1000k", "high_priority", "anneal"):
            pct = f"  ({-100 * (1 - n / prev_n):.1f}%)"
        else:
            pct = ""
        lines.append(f"  {stage:<20} {n:>10,}{pct}")
        if stage not in ("high_priority", "anneal"):
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

    # Pool 不变量
    ok = metrics.get("pool_invariant_ok", False)
    lines.append(f"Pool 不变量 (pool = anneal + hp): {'✓ 通过' if ok else '✗ 失败！'}")
    lines.append("")

    # Cluster 覆盖率
    coverage = metrics.get("pool_cluster_coverage", {})
    if coverage:
        lines.append("=== Cluster 覆盖率 ===")
        for bk, info in sorted(coverage.items()):
            pct = info["coverage_pct"]
            flag = "✓" if pct >= 95 else "⚠"
            lines.append(f"  {bk:<28} {info['pool_clusters']:>5}/{info['total_clusters']:<5} "
                         f"clusters  ({pct}%) {flag}")
        lines.append("")

    # Distance 分离
    hp_dist   = metrics.get("high_priority_distance", {})
    ann_dist  = metrics.get("anneal_distance", {})
    if hp_dist and ann_dist:
        lines.append("=== Distance to Centroid（越小越中心）===")
        lines.append(f"  high_priority 均值: {hp_dist['mean']:.6f}  (中位数 {hp_dist['median']:.6f})")
        lines.append(f"  anneal        均值: {ann_dist['mean']:.6f}  (中位数 {ann_dist['median']:.6f})")
        sep = ann_dist["mean"] > hp_dist["mean"]
        lines.append(f"  hp < anneal: {'✓ 正常' if sep else '⚠ 无分离，采样未区分中心性'}")
        lines.append("")

    # Instruction 长度
    hp_len  = metrics.get("high_priority_instr_len", {})
    ann_len = metrics.get("anneal_instr_len", {})
    if ann_len:
        lines.append("=== Instruction 长度（anneal 池）===")
        lines.append(f"  均值: {ann_len['mean']:.1f} chars  "
                     f"std: {ann_len['std']:.1f}  "
                     f"[{ann_len['min']}, {ann_len['max']}]")
        std_flag = "✓ 多样性高" if ann_len["std"] > 30 else "⚠ 长度集中"
        lines.append(f"  std 评估: {std_flag}")
        lines.append("")

    # Source mix
    src = metrics.get("source_mix", {})
    if src:
        lines.append("=== Source Mix（1000k 池）===")
        for s, pct in sorted(src.items()):
            lines.append(f"  {s}: {pct}%")
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


# ── 可视化 ────────────────────────────────────────────────────────────────────

def _get_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_funnel(metrics: dict, out_dir: Path) -> None:
    plt = _get_mpl()
    funnel = [(s, n) for s, n in metrics.get("funnel", [])
              if s not in ("high_priority", "anneal")]
    if not funnel:
        return
    stages, counts = zip(*funnel)
    fig, ax = plt.subplots(figsize=(10, 5))
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
    plt = _get_mpl()
    pool_path = root / "pool_1000k.jsonl"
    hp_path   = root / "high_priority_pool.jsonl"
    if not pool_path.exists():
        return

    def bucket_counts(path):
        c = Counter(_read_jsonl_field(path, "_meta", "bucket_key"))
        return c

    pool_cnt = bucket_counts(pool_path)
    hp_cnt   = bucket_counts(hp_path)
    buckets  = sorted(set(pool_cnt) | set(hp_cnt))

    x = range(len(buckets))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - w/2 for i in x], [pool_cnt.get(b, 0) for b in buckets], w, label="pool_1000k")
    ax.bar([i + w/2 for i in x], [hp_cnt.get(b, 0)   for b in buckets], w, label="high_priority")
    ax.set_xticks(list(x))
    ax.set_xticklabels(buckets, rotation=20, ha="right")
    ax.set_ylabel("Records")
    ax.set_title("Bucket Distribution")
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
        ax.set_title(f"{bk}\n(unique sizes shown per record)")
        ax.set_xlabel("Cluster size")
        ax.set_ylabel("Count")

    plt.suptitle("Cluster Size Distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "03_cluster_size_hist.png", dpi=120)
    plt.close()
    logger.info("[analyze] 03_cluster_size_hist.png 已写出")


def plot_instruction_len(root: Path, out_dir: Path) -> None:
    plt = _get_mpl()
    hp_path     = root / "high_priority_pool.jsonl"
    anneal_path = root / "anneal_pool.jsonl"

    def get_lens(path):
        return [len(r.get("instruction", ""))
                for r in _read_jsonl_records(path)]

    hp_lens     = get_lens(hp_path)
    anneal_lens = get_lens(anneal_path)
    if not hp_lens and not anneal_lens:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    if anneal_lens:
        ax.hist(anneal_lens, bins=50, alpha=0.6, label="anneal", color="steelblue", density=True)
    if hp_lens:
        ax.hist(hp_lens, bins=50, alpha=0.6, label="high_priority", color="coral", density=True)
    ax.set_xlabel("Instruction length (chars)")
    ax.set_ylabel("Density")
    ax.set_title("Instruction Length Distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "04_instruction_len.png", dpi=120)
    plt.close()
    logger.info("[analyze] 04_instruction_len.png 已写出")


def plot_distance_hist(root: Path, out_dir: Path) -> None:
    plt = _get_mpl()
    hp_path     = root / "high_priority_pool.jsonl"
    anneal_path = root / "anneal_pool.jsonl"

    def get_distances(path):
        return [r.get("_meta", {}).get("distance_to_centroid", 0)
                for r in _read_jsonl_records(path)]

    hp_dist     = get_distances(hp_path)
    anneal_dist = get_distances(anneal_path)
    if not hp_dist and not anneal_dist:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    if anneal_dist:
        ax.hist(anneal_dist, bins=50, alpha=0.6, label="anneal", color="steelblue", density=True)
    if hp_dist:
        ax.hist(hp_dist, bins=50, alpha=0.6, label="high_priority", color="coral", density=True)
    ax.set_xlabel("Distance to centroid")
    ax.set_ylabel("Density")
    ax.set_title("Distance to Centroid Distribution\n(hp peak should be left of anneal)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "05_distance_hist.png", dpi=120)
    plt.close()
    logger.info("[analyze] 05_distance_hist.png 已写出")


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
    plt.savefig(out_dir / "06_source_mix.png", dpi=120)
    plt.close()
    logger.info("[analyze] 06_source_mix.png 已写出")


def plot_umap(root: Path, out_dir: Path, sample_n: int = 50_000) -> None:
    """可选：UMAP 投影（需要 umap-learn 和 embeddings 目录）。"""
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

    # 采样
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
    plt.savefig(out_dir / "07_umap.png", dpi=120)
    plt.close()
    logger.info("[analyze] 07_umap.png 已写出")


# ── 入口 ──────────────────────────────────────────────────────────────────────

def run_analyze(output_root: str) -> None:
    """生成完整质量报告。"""
    root = Path(output_root)
    out_dir = root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[analyze] 分析 {root} ...")

    # 计算指标
    metrics = compute_metrics(root)

    # 写报告文本
    report = generate_report_text(metrics)
    report_path = out_dir / "report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)

    # 保存 metrics JSON
    import json as _json
    (out_dir / "metrics.json").write_text(
        _json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 生成图表
    try:
        plot_funnel(metrics, out_dir)
        plot_bucket_dist(root, out_dir)
        plot_cluster_size_hist(root, out_dir)
        plot_instruction_len(root, out_dir)
        plot_distance_hist(root, out_dir)
        plot_source_mix(root, out_dir)
        plot_umap(root, out_dir)
    except ImportError as e:
        logger.warning(f"[analyze] 绘图依赖缺失，跳过部分图表: {e}")

    logger.info(f"[analyze] 报告已写入 {out_dir}")
