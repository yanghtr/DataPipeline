"""HTML comment 截断。"""

from __future__ import annotations

from bs4 import BeautifulSoup, Comment

from .stats import CommentStats


def truncate_long(soup: BeautifulSoup, stats: CommentStats, max_chars: int) -> None:
    """截断超长 HTML comment。"""
    for node in soup.find_all(string=lambda text: isinstance(text, Comment)):
        text = str(node)
        stats.total += 1
        stats.chars.append(len(text))

        if len(text) > max_chars:
            node.replace_with(Comment(f" original comment truncated, chars={len(text)} "))
            stats.truncated += 1
