"""Stage 1 统计汇总与分布图输出。"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Iterable

from loguru import logger

from html_rewrite.config import HtmlRewriteConfig


def write_reject_reason_report(
    report_path: Path,
    stats_entries: list[dict],
) -> dict:
    """单独写出 reject 原因明细，直接列出阈值、id 和对应长度。"""
    grouped: dict[str, list[dict]] = {}
    for entry in stats_entries:
        if entry.get("status") != "rejected":
            continue
        reason = entry.get("reject_reason") or "unknown"
        grouped.setdefault(reason, []).append(entry)

    report = {
        "total_rejected": sum(len(entries) for entries in grouped.values()),
        "reasons": {},
    }

    for reason, entries in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        first_details = entries[0].get("reject_details") or {}
        report["reasons"][reason] = {
            "count": len(entries),
            "rule": first_details.get("rule"),
            "threshold_field": first_details.get("threshold_field"),
            "threshold_value": first_details.get("threshold_value"),
            "records": [
                {
                    "id": entry.get("id", ""),
                    "actual_cleaned_chars": entry.get("reject_details", {}).get(
                        "actual_cleaned_chars",
                        entry.get("stats", {}).get("cleaned_chars"),
                    ),
                    "visible_text_chars": entry.get("stats", {}).get("visible_text_chars"),
                    "original_chars": entry.get("stats", {}).get("original_chars"),
                    "details": entry.get("reject_details") or {},
                }
                for entry in entries
            ],
        }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = report_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(report_path)
    logger.info(f"[preprocess] reject reason report 已写出：{report_path}")
    return report


def write_summary(
    summary_log_path: Path,
    stats_entries: list[dict],
    cfg: HtmlRewriteConfig,
) -> dict:
    """写出 Stage 1 汇总统计 JSON。"""
    kept = sum(1 for entry in stats_entries if entry["status"] == "kept")
    rejected = sum(1 for entry in stats_entries if entry["status"] == "rejected")
    reject_reasons = Counter(
        entry["reject_reason"]
        for entry in stats_entries
        if entry["status"] == "rejected" and entry.get("reject_reason")
    )

    cleaned_chars = _collect_stat(stats_entries, "cleaned_chars")
    original_chars = _collect_stat(stats_entries, "original_chars")
    visible_text_chars = _collect_stat(stats_entries, "visible_text_chars")
    compression_ratio = _collect_stat(stats_entries, "compression_ratio")

    summary = {
        "total_input": len(stats_entries),
        "kept": kept,
        "rejected": rejected,
        "reject_reasons": dict(sorted(reject_reasons.items())),
        "thresholds": {
            "inline_script_max_chars": cfg.inline_script_max_chars,
            "json_payload_max_chars": cfg.json_payload_max_chars,
            "hidden_input_max_chars": cfg.hidden_input_max_chars,
            "html_comment_max_chars": cfg.html_comment_max_chars,
            "inline_style_max_chars": cfg.inline_style_max_chars,
            "min_preprocessed_chars": cfg.min_preprocessed_chars,
            "max_preprocessed_chars": cfg.max_preprocessed_chars,
        },
        "cleaned_chars": _describe_distribution(cleaned_chars),
        "original_chars": _describe_distribution(original_chars),
        "visible_text_chars": _describe_distribution(visible_text_chars),
        "compression_ratio": _describe_distribution(compression_ratio),
        "rule_counts": _build_rule_counts(stats_entries),
    }

    summary_log_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = summary_log_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(summary_log_path)
    logger.info(f"[preprocess] summary 已写出：{summary_log_path}")
    return summary


def write_plots(
    plot_dir: Path,
    stats_entries: list[dict],
    cfg: HtmlRewriteConfig,
) -> list[str]:
    """写出 Stage 1 分布图。缺少 matplotlib 时仅 warning，不中断主流程。"""
    try:
        plt = _get_mpl()
    except Exception as exc:
        logger.warning(f"[preprocess] 跳过分布图输出：matplotlib 不可用 ({exc})")
        return []

    plot_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    cleaned_all = _collect_stat(stats_entries, "cleaned_chars")
    cleaned_kept = _collect_stat((e for e in stats_entries if e["status"] == "kept"), "cleaned_chars")
    cleaned_rejected = _collect_stat((e for e in stats_entries if e["status"] == "rejected"), "cleaned_chars")
    original_chars = _collect_stat(stats_entries, "original_chars")
    visible_text_chars = _collect_stat(stats_entries, "visible_text_chars")
    compression_ratio = _collect_stat(stats_entries, "compression_ratio")

    plots = [
        (
            "01_original_chars_hist.png",
            original_chars,
            "Original HTML Length Distribution",
            "original_chars",
            [],
        ),
        (
            "02_cleaned_chars_hist.png",
            cleaned_all,
            "Preprocessed HTML Length Distribution",
            "cleaned_chars",
            [
                (cfg.min_preprocessed_chars, "min_keep", "forestgreen"),
                (cfg.max_preprocessed_chars, "max_keep", "crimson"),
            ],
        ),
        (
            "03_visible_text_chars_hist.png",
            visible_text_chars,
            "Visible Text Length Distribution",
            "visible_text_chars",
            [],
        ),
        (
            "04_compression_ratio_hist.png",
            compression_ratio,
            "Compression Ratio Distribution",
            "compression_ratio",
            [],
        ),
        (
            "05_inline_script_chars_hist.png",
            _collect_nested_lengths(stats_entries, "scripts", "inline_chars"),
            "Inline Script Length Distribution",
            "inline_script_chars",
            [(cfg.inline_script_max_chars, "truncate", "crimson")],
        ),
        (
            "06_json_payload_chars_hist.png",
            _collect_nested_lengths(stats_entries, "json_payloads", "chars"),
            "JSON Payload Length Distribution",
            "json_payload_chars",
            [(cfg.json_payload_max_chars, "truncate", "crimson")],
        ),
        (
            "07_inline_style_chars_hist.png",
            _collect_nested_lengths(stats_entries, "styles", "inline_chars"),
            "Inline Style Length Distribution",
            "inline_style_chars",
            [(cfg.inline_style_max_chars, "truncate", "crimson")],
        ),
        (
            "08_hidden_input_chars_hist.png",
            _collect_nested_lengths(stats_entries, "hidden_inputs", "value_chars"),
            "Hidden Input Value Length Distribution",
            "hidden_input_chars",
            [(cfg.hidden_input_max_chars, "truncate", "crimson")],
        ),
        (
            "09_html_comment_chars_hist.png",
            _collect_nested_lengths(stats_entries, "comments", "chars"),
            "HTML Comment Length Distribution",
            "html_comment_chars",
            [(cfg.html_comment_max_chars, "truncate", "crimson")],
        ),
    ]

    for filename, values, title, xlabel, thresholds in plots:
        if not values:
            continue
        out_path = plot_dir / filename
        _plot_hist(plt, values, out_path, title=title, xlabel=xlabel, thresholds=thresholds)
        written.append(filename)

    if cleaned_all and (cleaned_kept or cleaned_rejected):
        out_path = plot_dir / "10_cleaned_chars_keep_vs_reject.png"
        _plot_keep_vs_reject_hist(
            plt,
            kept=cleaned_kept,
            rejected=cleaned_rejected,
            out_path=out_path,
            min_chars=cfg.min_preprocessed_chars,
            max_chars=cfg.max_preprocessed_chars,
        )
        written.append(out_path.name)

    reject_reasons = Counter(
        entry["reject_reason"]
        for entry in stats_entries
        if entry["status"] == "rejected" and entry.get("reject_reason")
    )
    if reject_reasons:
        out_path = plot_dir / "11_reject_reason_counts.png"
        _plot_reject_reasons(plt, reject_reasons, out_path)
        written.append(out_path.name)

    logger.info(f"[preprocess] 分布图已写出：{plot_dir}（{len(written)} 个文件）")
    return written


def _build_rule_counts(stats_entries: list[dict]) -> dict:
    total_media = 0
    media_replaced = 0
    media_with_size = 0
    total_inline_scripts = 0
    truncated_inline_scripts = 0
    total_json_payloads = 0
    truncated_json_payloads = 0
    total_inline_styles = 0
    truncated_inline_styles = 0
    total_hidden_inputs = 0
    truncated_hidden_inputs = 0
    total_comments = 0
    truncated_comments = 0

    for entry in stats_entries:
        stats = entry["stats"]
        media = stats.get("media", {})
        total_media += media.get("total", 0)
        media_replaced += media.get("replaced", 0)
        media_with_size += media.get("with_size", 0)

        scripts = stats.get("scripts", {})
        total_inline_scripts += scripts.get("inline_total", 0)
        truncated_inline_scripts += scripts.get("inline_truncated", 0)

        payloads = stats.get("json_payloads", {})
        total_json_payloads += payloads.get("total", 0)
        truncated_json_payloads += payloads.get("truncated", 0)

        styles = stats.get("styles", {})
        total_inline_styles += styles.get("inline_total", 0)
        truncated_inline_styles += styles.get("inline_truncated", 0)

        hidden_inputs = stats.get("hidden_inputs", {})
        total_hidden_inputs += hidden_inputs.get("total", 0)
        truncated_hidden_inputs += hidden_inputs.get("truncated", 0)

        comments = stats.get("comments", {})
        total_comments += comments.get("total", 0)
        truncated_comments += comments.get("truncated", 0)

    return {
        "media": {
            "total": total_media,
            "replaced": media_replaced,
            "with_size": media_with_size,
        },
        "scripts": {
            "inline_total": total_inline_scripts,
            "inline_truncated": truncated_inline_scripts,
        },
        "json_payloads": {
            "total": total_json_payloads,
            "truncated": truncated_json_payloads,
        },
        "styles": {
            "inline_total": total_inline_styles,
            "inline_truncated": truncated_inline_styles,
        },
        "hidden_inputs": {
            "total": total_hidden_inputs,
            "truncated": truncated_hidden_inputs,
        },
        "comments": {
            "total": total_comments,
            "truncated": truncated_comments,
        },
    }


def _describe_distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    sorted_vals = sorted(values)
    return {
        "count": len(sorted_vals),
        "min": sorted_vals[0],
        "p50": _quantile(sorted_vals, 0.50),
        "p90": _quantile(sorted_vals, 0.90),
        "p95": _quantile(sorted_vals, 0.95),
        "p99": _quantile(sorted_vals, 0.99),
        "max": sorted_vals[-1],
        "mean": round(sum(sorted_vals) / len(sorted_vals), 4),
    }


def _quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, int(round((len(sorted_vals) - 1) * q))))
    value = sorted_vals[idx]
    return round(value, 4) if isinstance(value, float) else value


def _collect_stat(entries: Iterable[dict], key: str) -> list[float]:
    values: list[float] = []
    for entry in entries:
        stats = entry.get("stats", {})
        value = stats.get(key)
        if value is not None:
            values.append(value)
    return values


def _collect_nested_lengths(entries: Iterable[dict], group: str, key: str) -> list[int]:
    values: list[int] = []
    for entry in entries:
        stats = entry.get("stats", {})
        group_info = stats.get(group, {})
        for value in group_info.get(key, []):
            if value is not None:
                values.append(int(value))
    return values


def _get_mpl():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _plot_hist(
    plt,
    values: list[float],
    out_path: Path,
    *,
    title: str,
    xlabel: str,
    thresholds: list[tuple[int, str, str]],
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = min(60, max(10, int(len(values) ** 0.5)))
    ax.hist(values, bins=bins, color="steelblue", alpha=0.85, edgecolor="white")
    for threshold, label, color in thresholds:
        if threshold > 0:
            ax.axvline(threshold, color=color, linestyle="--", linewidth=1.8, label=f"{label}={threshold:,}")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    if thresholds:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_keep_vs_reject_hist(
    plt,
    *,
    kept: list[float],
    rejected: list[float],
    out_path: Path,
    min_chars: int,
    max_chars: int,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = 50
    if kept:
        ax.hist(kept, bins=bins, alpha=0.55, color="steelblue", label=f"kept ({len(kept):,})")
    if rejected:
        ax.hist(rejected, bins=bins, alpha=0.55, color="coral", label=f"rejected ({len(rejected):,})")
    if min_chars > 0:
        ax.axvline(min_chars, color="forestgreen", linestyle="--", linewidth=1.8, label=f"min_keep={min_chars:,}")
    if max_chars > 0:
        ax.axvline(max_chars, color="crimson", linestyle="--", linewidth=1.8, label=f"max_keep={max_chars:,}")
    ax.set_title("Preprocessed HTML Length: kept vs rejected")
    ax.set_xlabel("cleaned_chars")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_reject_reasons(plt, reject_reasons: Counter, out_path: Path) -> None:
    items = sorted(reject_reasons.items(), key=lambda item: item[1], reverse=True)
    labels = [label for label, _ in items]
    counts = [count for _, count in items]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, counts, color="coral")
    ax.set_title("Reject Reason Counts")
    ax.set_ylabel("Count")
    plt.xticks(rotation=20, ha="right")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{count:,}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
