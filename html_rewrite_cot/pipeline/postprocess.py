"""VLM 输出后处理 + 质量元信息生成。"""

from __future__ import annotations

import re

_HTML_PATTERNS = [
    r"```html",
    r"<!DOCTYPE\s+html",
    r"<html[\s>]",
    r"<head[\s>]",
    r"<body[\s>]",
    r"<style[\s>]",
]
_FIXED_SECTIONS = {"layout analysis", "colors observed", "structure and implementation plan"}


def postprocess_reasoning(raw: str) -> tuple[str, list[str]]:
    """
    清洗 VLM 输出，返回 (reasoning_text, generation_warnings)。

    处理步骤：
    1. strip 首尾空白
    2. 如果被非 HTML code fence 包裹，剥离外层 fence
    3. 检测 reasoning 中是否包含 HTML/CSS 代码
    4. 检测 markdown code fence 残留
    """
    warnings: list[str] = []
    text = raw.strip()

    # 剥离外层非 HTML code fence（如 ```text ... ``` 或 ``` ... ```）
    fence_match = re.match(r"^```[a-z]*\n?(.*?)```\s*$", text, re.DOTALL)
    if fence_match:
        inner = fence_match.group(1).strip()
        if not any(re.search(p, inner, re.IGNORECASE) for p in _HTML_PATTERNS[:3]):
            text = inner

    # 检测 HTML 内容
    for pat in _HTML_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            warnings.append(f"reasoning contains HTML-like content (pattern: {pat})")
            break

    # 检测 code fence 残留
    if re.search(r"```", text):
        warnings.append("reasoning contains markdown code fences")

    return text, warnings


def check_quality(reasoning_text: str, raw_html: str) -> dict:
    """生成质量元信息。"""
    word_count = len(reasoning_text.split())
    contains_html = bool(
        re.search(r"```html|<!DOCTYPE|<html[\s>]|<head[\s>]|<body[\s>]", reasoning_text, re.IGNORECASE)
    )
    contains_fence = "```" in reasoning_text
    has_layout = bool(re.search(r"Layout Analysis\s*:", reasoning_text, re.IGNORECASE))
    has_colors = bool(re.search(r"Colors Observed\s*:", reasoning_text, re.IGNORECASE))
    has_plan = bool(re.search(r"Structure and Implementation Plan\s*:", reasoning_text, re.IGNORECASE))

    region_sections: set[str] = set()
    for line in reasoning_text.splitlines():
        stripped = line.strip()
        if stripped.endswith(":") and 3 < len(stripped) <= 60:
            key = stripped[:-1].lower()
            if key not in _FIXED_SECTIONS:
                region_sections.add(key)
    region_section_count = len(region_sections)

    # 弱对齐检查
    html_lower = raw_html.lower()
    reasoning_lower = reasoning_text.lower()
    alignment_warnings: list[str] = []
    checks = [
        ("form", ["form", "input", "textarea", "select"]),
        ("sidebar", ["aside", "sidebar", "panel"]),
        ("footer", ["footer", "foot"]),
        ("table", ["<table", "<tr", "<th"]),
        ("image", ["<img", "background-image", "svg"]),
    ]
    for keyword, html_evidence in checks:
        if keyword in reasoning_lower:
            if not any(ev in html_lower for ev in html_evidence):
                alignment_warnings.append(
                    f'reasoning mentions "{keyword}" but no evidence found in html'
                )

    return {
        "reasoning_word_count": word_count,
        "contains_html_in_reasoning": contains_html,
        "contains_markdown_fence_in_reasoning": contains_fence,
        "has_layout_analysis_section": has_layout,
        "has_colors_observed_section": has_colors,
        "has_structure_implementation_plan_section": has_plan,
        "region_section_count": region_section_count,
        "alignment_warnings": alignment_warnings,
    }
