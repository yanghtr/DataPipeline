"""inline script 截断和 JSON/hydration payload 截断。"""

from __future__ import annotations

from bs4 import BeautifulSoup

from .stats import ScriptStats, JsonPayloadStats

# 已知的大型 hydration payload id
_HYDRATION_IDS: frozenset[str] = frozenset({
    "__NEXT_DATA__",
    "__NUXT_DATA__",
    "__REMIX_CONTEXT__",
    "nuxt-data",
    "__INITIAL_STATE__",
    "__APP_STATE__",
    "__REDUX_STATE__",
    "__PRELOADED_STATE__",
})


def _is_json_payload(tag) -> bool:
    """判断 script 是否为 JSON/hydration payload。"""
    tag_type = (tag.get("type") or "").lower()
    tag_id = (tag.get("id") or "")
    return (
        tag_type == "application/json"
        or tag_id in _HYDRATION_IDS
        or "application/ld+json" == tag_type
    )


def truncate_inline(soup: BeautifulSoup, stats: ScriptStats, max_chars: int) -> None:
    """截断超长 inline script（非 JSON payload）。"""
    for tag in soup.find_all("script"):
        # 外部 script：跳过
        if tag.get("src"):
            stats.external += 1
            continue

        # JSON payload 由 truncate_json_payload 处理，这里跳过
        if _is_json_payload(tag):
            continue

        content = tag.string or ""
        stats.inline_total += 1
        stats.inline_chars.append(len(content))

        if len(content) > max_chars:
            tag.clear()
            tag["data-inline-script-truncated"] = "true"
            tag["data-original-chars"] = str(len(content))
            stats.inline_truncated += 1


def truncate_json_payload(soup: BeautifulSoup, stats: JsonPayloadStats, max_chars: int) -> None:
    """截断超长 JSON/hydration payload。"""
    for tag in soup.find_all("script"):
        if tag.get("src"):
            continue
        if not _is_json_payload(tag):
            continue

        content = tag.string or ""
        stats.total += 1
        stats.chars.append(len(content))

        if len(content) > max_chars:
            tag.clear()
            tag["data-json-payload-truncated"] = "true"
            tag["data-original-chars"] = str(len(content))
            stats.truncated += 1
