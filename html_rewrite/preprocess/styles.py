"""inline <style> 截断。外部 CSS link 保留不动。"""

from __future__ import annotations

from bs4 import BeautifulSoup

from .stats import StyleStats


def truncate_inline(soup: BeautifulSoup, stats: StyleStats, max_chars: int) -> None:
    """截断超长 inline <style>，外部 CSS link 计数后跳过。"""
    for tag in soup.find_all("link", rel=True):
        rel = tag.get("rel") or []
        if isinstance(rel, list):
            rel = " ".join(rel)
        if "stylesheet" in rel.lower():
            stats.external_links += 1

    for tag in soup.find_all("style"):
        content = tag.string or ""
        stats.inline_total += 1
        stats.inline_chars.append(len(content))

        if len(content) > max_chars:
            tag.clear()
            tag.append(soup.new_string("/* original inline CSS truncated */"))
            tag["data-inline-style-truncated"] = "true"
            tag["data-original-chars"] = str(len(content))
            stats.inline_truncated += 1
