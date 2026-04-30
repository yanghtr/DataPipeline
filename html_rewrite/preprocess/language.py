"""页面主语言检测与英文过滤。"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
import unicodedata

from .parser import parse_html_with_lxml
from .text import extract_declared_lang_from_soup, extract_visible_text_from_soup


_LANGID_REQUIRED_MSG = (
    "html_rewrite language filtering requires the `langid` package. "
    "Install `langid>=1.1.6` in the current environment and retry."
)


@dataclass
class LanguageFilterResult:
    declared_lang: str = ""
    detected_lang: str = ""
    detected_lang_score: float | None = None
    detected_lang_top2: str = ""
    detected_lang_top2_score: float | None = None
    detector_margin: float | None = None
    visible_text_chars: int = 0
    sample_text_chars: int = 0
    letter_chars: int = 0
    latin_letter_chars: int = 0
    latin_ratio: float | None = None
    passed: bool | None = None
    reason: str = ""


def ensure_langid_available() -> None:
    try:
        importlib.import_module("langid.langid")
    except ImportError as exc:
        raise RuntimeError(_LANGID_REQUIRED_MSG) from exc


def evaluate_main_language(
    preprocessed_html: str,
    *,
    allowed_languages: list[str],
    min_visible_text_chars: int,
    min_letter_chars: int,
    sample_max_chars: int,
    min_latin_ratio: float,
    min_detector_margin: float,
) -> tuple[LanguageFilterResult, dict | None]:
    ensure_langid_available()
    detector = importlib.import_module("langid.langid")

    soup = parse_html_with_lxml(preprocessed_html)
    visible_text = extract_visible_text_from_soup(soup)
    declared_lang = extract_declared_lang_from_soup(soup)

    letter_chars, latin_letter_chars = _count_letters(visible_text)
    latin_ratio = round(latin_letter_chars / letter_chars, 4) if letter_chars > 0 else 0.0
    result = LanguageFilterResult(
        declared_lang=declared_lang,
        visible_text_chars=len(visible_text),
        letter_chars=letter_chars,
        latin_letter_chars=latin_letter_chars,
        latin_ratio=latin_ratio,
    )

    if result.visible_text_chars < min_visible_text_chars:
        result.passed = False
        result.reason = "language_detection_insufficient_text"
        return result, {
            "rule": "language_min_visible_text_chars",
            "threshold_field": "language_min_visible_text_chars",
            "threshold_value": min_visible_text_chars,
            "actual_visible_text_chars": result.visible_text_chars,
        }

    if result.letter_chars < min_letter_chars:
        result.passed = False
        result.reason = "language_detection_insufficient_letters"
        return result, {
            "rule": "language_min_letter_chars",
            "threshold_field": "language_min_letter_chars",
            "threshold_value": min_letter_chars,
            "actual_letter_chars": result.letter_chars,
            "actual_visible_text_chars": result.visible_text_chars,
        }

    if result.latin_ratio is not None and result.latin_ratio < min_latin_ratio:
        result.passed = False
        result.reason = "language_not_mainly_latin_script"
        return result, {
            "rule": "language_min_latin_ratio",
            "threshold_field": "language_min_latin_ratio",
            "threshold_value": min_latin_ratio,
            "actual_latin_ratio": result.latin_ratio,
            "actual_letter_chars": result.letter_chars,
            "actual_visible_text_chars": result.visible_text_chars,
        }

    sample_text = _sample_text(visible_text, sample_max_chars)
    result.sample_text_chars = len(sample_text)
    rankings = detector.rank(sample_text)
    if not rankings:
        result.passed = False
        result.reason = "language_detector_error"
        return result, {
            "rule": "langid_rank_non_empty",
            "actual_sample_text_chars": result.sample_text_chars,
        }

    top1_lang, top1_score = rankings[0]
    top2_lang, top2_score = rankings[1] if len(rankings) > 1 else ("", None)
    result.detected_lang = top1_lang
    result.detected_lang_score = _round_float(top1_score)
    result.detected_lang_top2 = top2_lang
    result.detected_lang_top2_score = _round_float(top2_score)
    if top2_score is not None:
        result.detector_margin = _round_float(top1_score - top2_score)

    allowed = {lang.lower() for lang in allowed_languages}
    if result.detected_lang.lower() not in allowed:
        result.passed = False
        result.reason = "non_english_page"
        return result, {
            "rule": "allowed_languages",
            "allowed_languages": sorted(allowed),
            "declared_lang": result.declared_lang,
            "detected_lang": result.detected_lang,
            "detected_lang_score": result.detected_lang_score,
            "detected_lang_top2": result.detected_lang_top2,
            "detected_lang_top2_score": result.detected_lang_top2_score,
            "detector_margin": result.detector_margin,
        }

    if result.detector_margin is not None and result.detector_margin < min_detector_margin:
        result.passed = False
        result.reason = "language_detection_low_margin"
        return result, {
            "rule": "language_min_detector_margin",
            "threshold_field": "language_min_detector_margin",
            "threshold_value": min_detector_margin,
            "declared_lang": result.declared_lang,
            "detected_lang": result.detected_lang,
            "detected_lang_score": result.detected_lang_score,
            "detected_lang_top2": result.detected_lang_top2,
            "detected_lang_top2_score": result.detected_lang_top2_score,
            "detector_margin": result.detector_margin,
        }

    result.passed = True
    result.reason = "allowed_language"
    return result, None


def _count_letters(text: str) -> tuple[int, int]:
    letter_chars = 0
    latin_letter_chars = 0
    for ch in text:
        if not ch.isalpha():
            continue
        letter_chars += 1
        if _is_latin_letter(ch):
            latin_letter_chars += 1
    return letter_chars, latin_letter_chars


def _is_latin_letter(ch: str) -> bool:
    try:
        return "LATIN" in unicodedata.name(ch)
    except ValueError:
        return False


def _sample_text(text: str, max_chars: int) -> str:
    text = text.strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text

    third = max(1, max_chars // 3)
    head = text[:third]
    tail = text[-third:]
    remaining = max_chars - len(head) - len(tail)
    mid_start = max(0, (len(text) // 2) - (remaining // 2))
    middle = text[mid_start:mid_start + max(0, remaining)]
    return " ".join(part for part in (head, middle, tail) if part)


def _round_float(value: float | None) -> float | None:
    if value is None:
        return None
    if math.isfinite(value):
        return round(value, 4)
    return value
