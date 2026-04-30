"""Stage 1 样本级过滤规则。"""

from __future__ import annotations

from dataclasses import dataclass

from bs4 import BeautifulSoup


@dataclass(frozen=True)
class FilterDecision:
    keep: bool
    reason: str | None = None


def decide_keep(
    preprocessed_html: str,
    *,
    min_preprocessed_chars: int,
    max_preprocessed_chars: int,
) -> FilterDecision:
    """基于最终预处理后的 HTML 做样本级 gate。"""
    html = preprocessed_html.strip()
    if not html:
        return FilterDecision(keep=False, reason="invalid_or_empty_html")

    if min_preprocessed_chars > 0 and len(html) < min_preprocessed_chars:
        return FilterDecision(keep=False, reason="too_short_after_preprocess")

    if max_preprocessed_chars > 0 and len(html) > max_preprocessed_chars:
        return FilterDecision(keep=False, reason="too_long_after_preprocess")

    if not _has_valid_html_skeleton(html):
        return FilterDecision(keep=False, reason="invalid_or_empty_html")

    return FilterDecision(keep=True)


def _has_valid_html_skeleton(html: str) -> bool:
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        return False
    return soup.find("html") is not None and soup.find("body") is not None
