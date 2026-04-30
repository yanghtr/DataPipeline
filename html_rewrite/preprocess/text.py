"""可见文本与声明语言抽取工具。"""

from __future__ import annotations

from bs4 import Comment
from bs4 import BeautifulSoup

_NON_VISIBLE_PARENTS = {"script", "style", "noscript", "template", "head", "title", "meta"}


def extract_visible_text_from_soup(soup: BeautifulSoup) -> str:
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
    return " ".join(parts)


def extract_declared_lang_from_soup(soup: BeautifulSoup) -> str:
    html_tag = soup.find("html")
    if html_tag is not None:
        lang = _normalize_lang_code(html_tag.get("lang", ""))
        if lang:
            return lang

    for meta in soup.find_all("meta"):
        http_equiv = (meta.get("http-equiv") or "").strip().lower()
        if http_equiv == "content-language":
            lang = _normalize_lang_code(meta.get("content", ""))
            if lang:
                return lang

        prop = (meta.get("property") or meta.get("name") or "").strip().lower()
        if prop == "og:locale":
            lang = _normalize_lang_code(meta.get("content", ""))
            if lang:
                return lang

    return ""


def _normalize_lang_code(raw: str) -> str:
    if not raw:
        return ""
    first = raw.split(",")[0].strip().lower()
    if not first:
        return ""
    return first.replace("_", "-").split("-")[0]
