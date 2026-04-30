"""预处理主编排器：按顺序调用各子模块，返回 (preprocessed_html, stats)。"""

from __future__ import annotations

from html_rewrite.config import HtmlRewriteConfig
from .stats import PreprocessStats
from . import media, scripts, styles, forms, comments, formatter
from .parser import parse_html_with_lxml
from .text import extract_visible_text_from_soup


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
    stats.visible_text_chars = len(extract_visible_text_from_soup(soup))

    return out_html, stats
