"""hidden input value 截断。可见表单字段保留不动。"""

from __future__ import annotations

from bs4 import BeautifulSoup

from .stats import HiddenInputStats


def truncate_hidden_inputs(soup: BeautifulSoup, stats: HiddenInputStats, max_chars: int) -> None:
    """截断超长 hidden input value。"""
    for tag in soup.find_all("input"):
        tag_type = (tag.get("type") or "").lower()
        if tag_type != "hidden":
            continue

        value = tag.get("value") or ""
        stats.total += 1
        stats.value_chars.append(len(value))

        if len(value) > max_chars:
            tag["value"] = f"__LONG_HIDDEN_VALUE_TRUNCATED_CHARS_{len(value)}__"
            stats.truncated += 1
