"""HTML 格式标准化：统一缩进、去多余空行，记录节点数变化。"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup

from .stats import FormatterStats

_TRACKED_TAGS: tuple[str, ...] = (
    "header", "main", "footer", "nav", "aside",
    "img", "video", "audio", "iframe", "script", "style",
)

_BLANK_LINES_RE = re.compile(r"\n{3,}")


def _count_tags(soup: BeautifulSoup) -> dict[str, int]:
    return {tag: len(soup.find_all(tag)) for tag in _TRACKED_TAGS}


def _node_count(soup: BeautifulSoup) -> int:
    return len(list(soup.descendants))


def serialize(soup: BeautifulSoup, stats: FormatterStats) -> str:
    """序列化 soup 为格式化 HTML，记录统计。"""
    stats.node_count_before = _node_count(soup)
    stats.tag_counts_before = _count_tags(soup)

    try:
        html = soup.prettify(formatter="html5")
    except Exception:
        stats.parse_ok = False
        html = str(soup)

    # 去掉连续 3 行以上的空行，压缩为最多 2 行
    html = _BLANK_LINES_RE.sub("\n\n", html)

    # 用序列化结果重新解析，以获取 after 统计
    try:
        after_soup = BeautifulSoup(html, "lxml")
        stats.node_count_after = _node_count(after_soup)
        stats.tag_counts_after = _count_tags(after_soup)
    except Exception:
        stats.node_count_after = stats.node_count_before
        stats.tag_counts_after = stats.tag_counts_before

    return html
