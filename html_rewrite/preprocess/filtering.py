"""Stage 1 样本级过滤规则。"""

from __future__ import annotations

from dataclasses import dataclass

from .parser import parse_html_with_lxml


@dataclass(frozen=True)
class FilterDecision:
    keep: bool
    reason: str | None = None
    details: dict | None = None


def decide_keep(
    preprocessed_html: str,
    *,
    min_preprocessed_chars: int,
    max_preprocessed_chars: int,
) -> FilterDecision:
    """基于最终预处理后的 HTML 做样本级 gate。"""
    html = preprocessed_html.strip()
    if not html:
        return FilterDecision(
            keep=False,
            reason="invalid_or_empty_html",
            details={
                "rule": "non_empty_html_required",
                "actual_cleaned_chars": 0,
            },
        )

    if min_preprocessed_chars > 0 and len(html) < min_preprocessed_chars:
        return FilterDecision(
            keep=False,
            reason="too_short_after_preprocess",
            details={
                "threshold_field": "min_preprocessed_chars",
                "threshold_value": min_preprocessed_chars,
                "actual_cleaned_chars": len(html),
            },
        )

    if max_preprocessed_chars > 0 and len(html) > max_preprocessed_chars:
        return FilterDecision(
            keep=False,
            reason="too_long_after_preprocess",
            details={
                "threshold_field": "max_preprocessed_chars",
                "threshold_value": max_preprocessed_chars,
                "actual_cleaned_chars": len(html),
            },
        )

    has_html_tag, has_body_tag = _inspect_html_skeleton(html)
    if not (has_html_tag and has_body_tag):
        return FilterDecision(
            keep=False,
            reason="invalid_or_empty_html",
            details={
                "rule": "html_and_body_tags_required",
                "actual_cleaned_chars": len(html),
                "has_html_tag": has_html_tag,
                "has_body_tag": has_body_tag,
            },
        )

    return FilterDecision(keep=True)


def _inspect_html_skeleton(html: str) -> tuple[bool, bool]:
    soup = parse_html_with_lxml(html)
    return soup.find("html") is not None, soup.find("body") is not None
