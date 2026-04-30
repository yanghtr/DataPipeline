"""媒体路径替换：将所有媒体资源路径替换为统一 placeholder 格式。"""

from __future__ import annotations

import base64
import re
import struct
from urllib.parse import urlparse

from bs4 import BeautifulSoup, Comment, Tag

from .stats import MediaStats

# 支持的扩展名白名单
_KNOWN_EXTS: frozenset[str] = frozenset({
    ".jpg", ".jpeg", ".png", ".webp", ".gif", ".svg",
    ".mp4", ".webm", ".mp3", ".wav", ".ogg", ".pdf",
})

# base64 mime type → 扩展名映射
_MIME_TO_EXT: dict[str, str] = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/svg+xml": ".svg",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/wav": ".wav",
    "audio/ogg": ".ogg",
    "application/pdf": ".pdf",
}

# CSS url() 匹配（含引号或无引号）
_CSS_URL_RE = re.compile(
    r"""url\(\s*(['"]?)(data:[^)'"]+|[^)'"]+)\1\s*\)""",
    re.IGNORECASE,
)

# data URI 匹配
_DATA_URI_RE = re.compile(
    r"""^data:([^;,]+)(?:;[^,]*)?,(.+)$""",
    re.DOTALL | re.IGNORECASE,
)

_TRACKED_TAGS: list[str] = [
    "header", "main", "footer", "nav", "aside",
    "img", "video", "audio", "iframe", "script", "style",
]


def _ext_from_url(url: str) -> str:
    """从 URL 提取扩展名，失败返回 .media。"""
    try:
        path = urlparse(url).path
        dot = path.rfind(".")
        if dot != -1:
            ext = path[dot:].lower().split("?")[0]
            if ext in _KNOWN_EXTS:
                return ext
    except Exception:
        pass
    return ".media"


def _ext_from_mime(mime: str) -> str:
    """从 mime type 推断扩展名。"""
    return _MIME_TO_EXT.get(mime.strip().lower(), ".media")


def _parse_png_size(data: bytes) -> tuple[int, int] | None:
    """从 PNG 文件头解析宽高（不依赖 Pillow）。"""
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    try:
        w, h = struct.unpack(">II", data[16:24])
        return w, h
    except Exception:
        return None


def _parse_jpeg_size(data: bytes) -> tuple[int, int] | None:
    """从 JPEG 文件头解析宽高（SOF markers）。"""
    if len(data) < 3 or data[:2] != b"\xff\xd8":
        return None
    i = 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            break
        marker = data[i + 1]
        if marker in (0xC0, 0xC1, 0xC2):  # SOF0 / SOF1 / SOF2
            if i + 9 <= len(data):
                h, w = struct.unpack(">HH", data[i + 5:i + 9])
                return w, h
        length = struct.unpack(">H", data[i + 2:i + 4])[0] if i + 4 <= len(data) else 0
        i += 2 + length
    return None


def _size_from_base64(data_uri: str) -> tuple[int | None, int | None]:
    """尝试从 base64 data URI 解析图片宽高。"""
    m = _DATA_URI_RE.match(data_uri)
    if not m:
        return None, None
    mime = m.group(1)
    b64_data = m.group(2)
    if "image" not in mime.lower():
        return None, None
    try:
        raw = base64.b64decode(b64_data[:8192])  # 只解码头部
        size = _parse_png_size(raw) or _parse_jpeg_size(raw)
        if size:
            return size
    except Exception:
        pass
    return None, None


def _make_placeholder(url: str, tag_w: str | None, tag_h: str | None, is_base64: bool = False) -> str:
    """构造 placeholder 路径。"""
    if tag_w and tag_h:
        w, h = tag_w, tag_h
    elif is_base64:
        pw, ph = _size_from_base64(url)
        w = str(pw) if pw else "unknown"
        h = str(ph) if ph else "unknown"
    else:
        w, h = "unknown", "unknown"

    ext = _ext_from_mime(
        _DATA_URI_RE.match(url).group(1) if is_base64 and _DATA_URI_RE.match(url) else ""
    ) if is_base64 else _ext_from_url(url)

    return f"__MEDIA_PLACEHOLDER__/media__width{w}__height{h}{ext}"


def _is_data_uri(val: str) -> bool:
    return val.strip().startswith("data:")


def _replace_attr(tag: Tag, attr: str, stats: MediaStats, tag_type: str, w_attr: str = "width", h_attr: str = "height") -> None:
    """替换单个属性中的媒体路径。"""
    val = tag.get(attr)
    if not val or not isinstance(val, str):
        return
    val = val.strip()
    if not val or val == "#":
        return

    tag_w = tag.get(w_attr)
    tag_h = tag.get(h_attr)
    tag_w = str(tag_w).strip() if tag_w else None
    tag_h = str(tag_h).strip() if tag_h else None

    is_b64 = _is_data_uri(val)
    placeholder = _make_placeholder(val, tag_w, tag_h, is_base64=is_b64)
    tag[attr] = placeholder

    stats.total += 1
    stats.replaced += 1
    if is_b64:
        stats.base64 += 1
    else:
        stats.regular += 1

    if tag_type == "img":
        stats.images += 1
    elif tag_type == "video":
        stats.videos += 1
    elif tag_type == "audio":
        stats.audios += 1
    else:
        stats.iframes += 1

    if (tag_w and tag_w != "unknown") and (tag_h and tag_h != "unknown"):
        stats.with_size += 1
    else:
        stats.without_size += 1


def _replace_css_urls(css_text: str, stats: MediaStats) -> str:
    """替换 CSS 文本中所有 url() 引用。"""
    def _sub(m: re.Match) -> str:
        url = m.group(2).strip()
        if not url or url.startswith("#"):
            return m.group(0)
        is_b64 = _is_data_uri(url)
        placeholder = _make_placeholder(url, None, None, is_base64=is_b64)
        stats.total += 1
        stats.replaced += 1
        if is_b64:
            stats.base64 += 1
            stats.without_size += 1
        else:
            stats.regular += 1
            stats.without_size += 1
        return f"url({placeholder})"

    return _CSS_URL_RE.sub(_sub, css_text)


def replace_all(soup: BeautifulSoup, stats: MediaStats) -> None:
    """对 soup 原地做全部媒体路径替换。"""

    # img[src] 和 img[srcset]
    for tag in soup.find_all("img"):
        if tag.get("src"):
            _replace_attr(tag, "src", stats, "img")
        # srcset：无论是否有 src 都处理，整个属性替换为单个 placeholder
        if tag.get("srcset"):
            first_url = str(tag["srcset"]).split()[0]
            tag_w = str(tag.get("width", "")).strip() or None
            tag_h = str(tag.get("height", "")).strip() or None
            placeholder = _make_placeholder(first_url, tag_w, tag_h)
            tag["srcset"] = placeholder
            stats.total += 1
            stats.replaced += 1
            stats.images += 1
            stats.regular += 1
            if tag_w and tag_h:
                stats.with_size += 1
            else:
                stats.without_size += 1

    # source[src] / source[srcset]
    for tag in soup.find_all("source"):
        if tag.get("src"):
            _replace_attr(tag, "src", stats, "video")
        if tag.get("srcset"):
            placeholder = _make_placeholder(str(tag["srcset"]).split()[0], None, None)
            tag["srcset"] = placeholder
            stats.total += 1
            stats.replaced += 1
            stats.regular += 1
            stats.without_size += 1
            stats.images += 1

    # video[src], video[poster]
    for tag in soup.find_all("video"):
        if tag.get("src"):
            _replace_attr(tag, "src", stats, "video")
        if tag.get("poster"):
            _replace_attr(tag, "poster", stats, "img")

    # audio[src]
    for tag in soup.find_all("audio", src=True):
        _replace_attr(tag, "src", stats, "audio")

    # iframe[src]
    for tag in soup.find_all("iframe", src=True):
        _replace_attr(tag, "src", stats, "iframe")

    # embed[src]
    for tag in soup.find_all("embed", src=True):
        _replace_attr(tag, "src", stats, "iframe")

    # object[data]
    for tag in soup.find_all("object", data=True):
        _replace_attr(tag, "data", stats, "iframe")

    # inline style 属性中的 url()
    for tag in soup.find_all(style=True):
        original = tag["style"]
        if "url(" in original.lower():
            tag["style"] = _replace_css_urls(original, stats)

    # <style> 标签内容中的 url()
    for tag in soup.find_all("style"):
        if tag.string and "url(" in tag.string.lower():
            tag.string = _replace_css_urls(tag.string, stats)
