"""Stage 1 样本级过滤规则。"""

from __future__ import annotations

from dataclasses import dataclass

from html_rewrite.config import HtmlRewriteConfig
from .language import evaluate_main_language
from .stats import PreprocessStats
from .parser import parse_html_with_lxml


@dataclass(frozen=True)
class FilterDecision:
    keep: bool
    reason: str | None = None
    details: dict | None = None


def decide_keep(
    preprocessed_html: str,
    stats: PreprocessStats,
    cfg: HtmlRewriteConfig,
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

    if cfg.min_preprocessed_chars > 0 and len(html) < cfg.min_preprocessed_chars:
        return FilterDecision(
            keep=False,
            reason="too_short_after_preprocess",
            details={
                "threshold_field": "min_preprocessed_chars",
                "threshold_value": cfg.min_preprocessed_chars,
                "actual_cleaned_chars": len(html),
            },
        )

    if cfg.max_preprocessed_chars > 0 and len(html) > cfg.max_preprocessed_chars:
        return FilterDecision(
            keep=False,
            reason="too_long_after_preprocess",
            details={
                "threshold_field": "max_preprocessed_chars",
                "threshold_value": cfg.max_preprocessed_chars,
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

    if cfg.enable_language_filter:
        language_result, language_reject_details = evaluate_main_language(
            preprocessed_html,
            allowed_languages=cfg.allowed_languages,
            min_visible_text_chars=cfg.language_min_visible_text_chars,
            min_letter_chars=cfg.language_min_letter_chars,
            sample_max_chars=cfg.language_sample_max_chars,
            min_latin_ratio=cfg.language_min_latin_ratio,
            min_detector_margin=cfg.language_min_detector_margin,
        )
        stats.language.declared_lang = language_result.declared_lang
        stats.language.detected_lang = language_result.detected_lang
        stats.language.detected_lang_score = language_result.detected_lang_score
        stats.language.detected_lang_top2 = language_result.detected_lang_top2
        stats.language.detected_lang_top2_score = language_result.detected_lang_top2_score
        stats.language.detector_margin = language_result.detector_margin
        stats.language.sample_text_chars = language_result.sample_text_chars
        stats.language.letter_chars = language_result.letter_chars
        stats.language.latin_letter_chars = language_result.latin_letter_chars
        stats.language.latin_ratio = language_result.latin_ratio
        stats.language.passed = language_result.passed
        stats.language.reason = language_result.reason

        if not language_result.passed:
            details = language_reject_details or {}
            details.setdefault("declared_lang", language_result.declared_lang)
            details.setdefault("detected_lang", language_result.detected_lang)
            details.setdefault("detected_lang_score", language_result.detected_lang_score)
            details.setdefault("detected_lang_top2", language_result.detected_lang_top2)
            details.setdefault("detected_lang_top2_score", language_result.detected_lang_top2_score)
            details.setdefault("detector_margin", language_result.detector_margin)
            details.setdefault("actual_visible_text_chars", language_result.visible_text_chars)
            return FilterDecision(
                keep=False,
                reason=language_result.reason,
                details=details,
            )
    else:
        stats.language.passed = None
        stats.language.reason = "language_filter_disabled"

    return FilterDecision(keep=True)


def _inspect_html_skeleton(html: str) -> tuple[bool, bool]:
    soup = parse_html_with_lxml(html)
    return soup.find("html") is not None, soup.find("body") is not None
