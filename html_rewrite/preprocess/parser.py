"""HTML 解析依赖检查与统一入口。"""

from __future__ import annotations

from bs4 import BeautifulSoup
from bs4 import FeatureNotFound

_LXML_REQUIRED_MSG = (
    "html_rewrite Stage 1 requires the 'lxml' parser. "
    "Please install `lxml>=5.0` in the current environment and retry."
)


def ensure_lxml_available() -> None:
    """启动前显式检查 lxml 可用，缺失时直接报错停止。"""
    try:
        BeautifulSoup("<html><body></body></html>", "lxml")
    except FeatureNotFound as exc:
        raise RuntimeError(_LXML_REQUIRED_MSG) from exc


def parse_html_with_lxml(html: str) -> BeautifulSoup:
    """统一使用 lxml 解析 HTML，缺失依赖时直接报错。"""
    try:
        return BeautifulSoup(html, "lxml")
    except FeatureNotFound as exc:
        raise RuntimeError(_LXML_REQUIRED_MSG) from exc
