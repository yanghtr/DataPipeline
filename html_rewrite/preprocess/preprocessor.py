"""预处理主编排器：按顺序调用各子模块，返回 (preprocessed_html, stats)。"""

from __future__ import annotations

from bs4 import Comment
from bs4 import BeautifulSoup

from html_rewrite.config import HtmlRewriteConfig
from .stats import PreprocessStats
from . import media, scripts, styles, forms, comments, formatter
from .parser import parse_html_with_lxml

_NON_VISIBLE_PARENTS = {"script", "style", "noscript", "template", "head", "title", "meta"}


def _count_visible_text_chars(soup: BeautifulSoup) -> int:
    parts: list[str] = []
    for text_node in soup.find_all(string=True):
        if isinstance(text_node, Comment):
            continue
        parent = getattr(text_node, "parent", None)
        if parent is not None and getattr(parent, "name", None) in _NON_VISIBLE_PARENTS:
            continue
        text = str(text_node).strip()
        if text:
            parts.append(text)
    return len(" ".join(parts))


def preprocess(html: str, cfg: HtmlRewriteConfig) -> tuple[str, PreprocessStats]:
    """
    对原始 HTML 做最小化预处理。

    Returns:
        (preprocessed_html, stats)
    """
    stats = PreprocessStats(original_chars=len(html))

    soup = parse_html_with_lxml(html)
    stats.formatter.parse_ok = True

    # 1. 媒体路径替换
    media.replace_all(soup, stats.media, cfg.fetch_media_size)

    # 2. inline script 截断
    scripts.truncate_inline(soup, stats.scripts, cfg.inline_script_max_chars)

    # 3. JSON/hydration payload 截断
    scripts.truncate_json_payload(soup, stats.json_payloads, cfg.json_payload_max_chars)

    # 4. inline style 截断
    styles.truncate_inline(soup, stats.styles, cfg.inline_style_max_chars)

    # 5. hidden input value 截断
    forms.truncate_hidden_inputs(soup, stats.hidden_inputs, cfg.hidden_input_max_chars)

    # 6. HTML comment 截断
    comments.truncate_long(soup, stats.comments, cfg.html_comment_max_chars)

    # 7. 格式标准化 + 序列化
    out_html = formatter.serialize(soup, stats.formatter)
    stats.cleaned_chars = len(out_html)
    stats.visible_text_chars = _count_visible_text_chars(soup)

    return out_html, stats
