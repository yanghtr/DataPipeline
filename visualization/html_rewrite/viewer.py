#!/usr/bin/env python3
"""
HTML 改写结果多列对比可视化工具。

默认列结构：
  - 第一个 JSONL：同时生成 preprocessed_html 列 + output_html 列
  - 后续 JSONL：只生成 output_html 列（预处理结果相同，避免重复）
  - 原站列：通过本地代理拉取并改写 HTML，尽量以内嵌方式展示真实站点

渲染机制：
  - /render/<col>/<row>：渲染 JSONL 中保存的 HTML，并注入 <base href="原站域名">
  - /live/<row> + /proxy/...：通过本地 Flask 代理拉取原站 HTML/资源，移除
    常见的 frame/CSP 限制并改写资源 URL，尽量提高原站内嵌成功率

用法:
    python visualization/vis_results/html_rewrite/viewer.py \
        --jsonl /path/to/output.jsonl \
        [--sample-n 20] [--random-sample] [--port 7862]

    python visualization/vis_results/html_rewrite/viewer.py \
        --jsonl /path/to/output_a.jsonl /path/to/output_b.jsonl \
        --sample-n 20
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import sys
import threading
import zlib
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin, urlparse, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from flask import jsonify, make_response, request, Flask

try:
    from loguru import logger
except ImportError:  # pragma: no cover - fallback for minimal environments
    import logging

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger("html_rewrite_viewer")

app = Flask(__name__)

# ── 常量 ─────────────────────────────────────────────────────────────────────

FIELD_PREPROCESSED = "preprocessed_html"
FIELD_OUTPUT = "output_html"

PROXYABLE_SCHEMES = {"http", "https"}
REMOTE_TIMEOUT_SECONDS = 20
REMOTE_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# ── HTML / 代理工具 ──────────────────────────────────────────────────────────

_HEAD_RE = re.compile(r"(<head[^>]*>)", re.IGNORECASE)
_BASE_RE = re.compile(r"<base\b[^>]*>", re.IGNORECASE)
_META_FRAME_GUARD_RE = re.compile(
    r"<meta[^>]+http-equiv=[\"']?(?:Content-Security-Policy|X-Frame-Options)[\"']?[^>]*>",
    re.IGNORECASE,
)
_URL_ATTR_RE = re.compile(
    r"(?P<prefix>\b(?:src|href|action|poster|data)\s*=\s*)"
    r"(?:(?P<dq>\"(?P<dqv>[^\"]*)\")|(?P<sq>'(?P<sqv>[^']*)')|(?P<bare>[^\s>]+))",
    re.IGNORECASE,
)
_SRCSET_ATTR_RE = re.compile(
    r"(?P<prefix>\bsrcset\s*=\s*)"
    r"(?:(?P<dq>\"(?P<dqv>[^\"]*)\")|(?P<sq>'(?P<sqv>[^']*)')|(?P<bare>[^\s>]+))",
    re.IGNORECASE | re.DOTALL,
)
_STYLE_BLOCK_RE = re.compile(r"(<style\b[^>]*>)(.*?)(</style>)", re.IGNORECASE | re.DOTALL)
_STYLE_ATTR_RE = re.compile(
    r"(?P<prefix>\bstyle\s*=\s*)"
    r"(?:(?P<dq>\"(?P<dqv>[^\"]*)\")|(?P<sq>'(?P<sqv>[^']*)')|(?P<bare>[^\s>]+))",
    re.IGNORECASE | re.DOTALL,
)
_CSS_URL_RE = re.compile(r"url\((?P<quote>[\"']?)(?P<value>.*?)(?P=quote)\)", re.IGNORECASE)


def _origin(url: str) -> str:
    """提取 URL 的 origin（scheme://host/），用作 <base href>。"""
    try:
        p = urlparse(url)
        return f"{p.scheme}://{p.netloc}/"
    except Exception:
        return url


def inject_base_url(html: str, url: str) -> str:
    """
    若 HTML 中没有 <base> 标签，在 <head> 紧后注入 <base href="origin">。
    这让 HTML 中的相对路径资源（CSS、图片、JS）能从原始域名正确加载。
    """
    if not url or "<base" in html.lower():
        return html
    origin = _origin(url)
    base_tag = f'<base href="{origin}">'
    match = _HEAD_RE.search(html)
    if match:
        return html[: match.end()] + "\n  " + base_tag + html[match.end() :]
    return base_tag + "\n" + html


def _match_attr_parts(match: re.Match[str]) -> tuple[str, str]:
    if match.group("dq") is not None:
        return '"', match.group("dqv") or ""
    if match.group("sq") is not None:
        return "'", match.group("sqv") or ""
    return "", match.group("bare") or ""


def _rebuild_attr(prefix: str, quote_char: str, value: str) -> str:
    if quote_char:
        return f"{prefix}{quote_char}{value}{quote_char}"
    return f"{prefix}{value}"


def _is_proxyable_url(raw_url: str) -> bool:
    lowered = raw_url.strip().lower()
    if not lowered:
        return False
    if lowered.startswith(("#", "javascript:", "mailto:", "tel:", "data:", "blob:", "about:")):
        return False
    return True


def proxy_path_for_url(url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme not in PROXYABLE_SCHEMES or not parts.netloc:
        return url

    path = quote(parts.path or "/", safe="/%:@+;,=~!*'()[]")
    proxied = f"/proxy/{parts.scheme}/{parts.netloc}{path}"
    if parts.query:
        proxied += f"?{parts.query}"
    if parts.fragment:
        proxied += f"#{parts.fragment}"
    return proxied


def _rewrite_url_to_proxy(raw_url: str, base_url: str) -> str:
    if not _is_proxyable_url(raw_url):
        return raw_url

    absolute_url = urljoin(base_url, raw_url)
    parts = urlsplit(absolute_url)
    if parts.scheme not in PROXYABLE_SCHEMES or not parts.netloc:
        return raw_url
    return proxy_path_for_url(absolute_url)


def _rewrite_srcset_value(srcset_value: str, base_url: str) -> str:
    candidates = []
    for chunk in srcset_value.split(","):
        item = chunk.strip()
        if not item:
            continue
        pieces = item.split()
        if not pieces:
            continue
        proxied_url = _rewrite_url_to_proxy(pieces[0], base_url)
        descriptor = " ".join(pieces[1:])
        candidates.append(" ".join(part for part in (proxied_url, descriptor) if part))
    return ", ".join(candidates)


def _rewrite_css_urls(css_text: str, base_url: str) -> str:
    def repl(match: re.Match[str]) -> str:
        quote_char = match.group("quote") or ""
        raw_url = (match.group("value") or "").strip()
        proxied_url = _rewrite_url_to_proxy(raw_url, base_url)
        return f"url({quote_char}{proxied_url}{quote_char})"

    return _CSS_URL_RE.sub(repl, css_text)


def rewrite_live_html(html: str, base_url: str) -> str:
    """
    将原站 HTML 改写为更适合内嵌预览的版本：
      - 移除 meta 形式的 CSP / X-Frame-Options
      - 删除原始 <base>
      - 将常见资源 URL 改写为 /proxy/... 同源代理地址
      - 为动态相对路径补充 <base href="/proxy/...">
    """
    html = _META_FRAME_GUARD_RE.sub("", html)
    html = _BASE_RE.sub("", html)

    def style_block_repl(match: re.Match[str]) -> str:
        return f"{match.group(1)}{_rewrite_css_urls(match.group(2), base_url)}{match.group(3)}"

    def style_attr_repl(match: re.Match[str]) -> str:
        quote_char, value = _match_attr_parts(match)
        return _rebuild_attr(match.group("prefix"), quote_char, _rewrite_css_urls(value, base_url))

    def srcset_attr_repl(match: re.Match[str]) -> str:
        quote_char, value = _match_attr_parts(match)
        return _rebuild_attr(match.group("prefix"), quote_char, _rewrite_srcset_value(value, base_url))

    def url_attr_repl(match: re.Match[str]) -> str:
        quote_char, value = _match_attr_parts(match)
        return _rebuild_attr(match.group("prefix"), quote_char, _rewrite_url_to_proxy(value, base_url))

    html = _STYLE_BLOCK_RE.sub(style_block_repl, html)
    html = _STYLE_ATTR_RE.sub(style_attr_repl, html)
    html = _SRCSET_ATTR_RE.sub(srcset_attr_repl, html)
    html = _URL_ATTR_RE.sub(url_attr_repl, html)

    proxy_base = proxy_path_for_url(base_url)
    injected_head = (
        f'<base href="{proxy_base}">\n'
        '  <meta name="referrer" content="no-referrer-when-downgrade">\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1">'
    )
    match = _HEAD_RE.search(html)
    if match:
        return html[: match.end()] + "\n  " + injected_head + html[match.end() :]
    return f"<head>\n  {injected_head}\n</head>\n{html}"


def _decode_http_body(body: bytes, headers: dict[str, str]) -> bytes:
    encoding = (headers.get("Content-Encoding") or "").lower()
    if "gzip" in encoding:
        try:
            return gzip.decompress(body)
        except Exception:
            return body
    if "deflate" in encoding:
        for wbits in (zlib.MAX_WBITS, -zlib.MAX_WBITS):
            try:
                return zlib.decompress(body, wbits)
            except Exception:
                continue
    return body


def _decode_text_body(body: bytes, headers: dict[str, str]) -> str:
    content_type = headers.get("Content-Type", "")
    charset = "utf-8"
    match = re.search(r"charset=([^\s;]+)", content_type, re.IGNORECASE)
    if match:
        charset = match.group(1).strip().strip('"').strip("'")

    for encoding in (charset, "utf-8", "latin-1"):
        try:
            return body.decode(encoding)
        except Exception:
            continue
    return body.decode("utf-8", errors="replace")


def _looks_like_html(content_type: str) -> bool:
    lowered = (content_type or "").lower()
    return "text/html" in lowered or "application/xhtml+xml" in lowered


def _forward_headers_from_browser() -> dict[str, str]:
    forwarded: dict[str, str] = {
        "User-Agent": REMOTE_USER_AGENT,
        "Accept-Language": request.headers.get("Accept-Language", "zh-CN,zh;q=0.9,en;q=0.8"),
    }
    for name in ("Accept", "Content-Type", "Origin"):
        value = request.headers.get(name)
        if value:
            forwarded[name] = value
    return forwarded


def fetch_remote_resource(
    url: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, str, dict[str, str], bytes]:
    req_headers = {"User-Agent": REMOTE_USER_AGENT, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"}
    if headers:
        req_headers.update({k: v for k, v in headers.items() if v})

    req = Request(url, data=body, headers=req_headers, method=method.upper())
    try:
        with urlopen(req, timeout=REMOTE_TIMEOUT_SECONDS) as resp:
            payload = resp.read()
            resp_headers = {k: v for k, v in resp.headers.items()}
            payload = _decode_http_body(payload, resp_headers)
            return getattr(resp, "status", 200), resp.geturl(), resp_headers, payload
    except HTTPError as exc:
        payload = exc.read()
        resp_headers = {k: v for k, v in exc.headers.items()}
        payload = _decode_http_body(payload, resp_headers)
        return exc.code, exc.geturl() or url, resp_headers, payload
    except URLError as exc:
        raise RuntimeError(f"无法访问原站资源: {exc.reason}") from exc


def _build_proxy_error_page(url: str, message: str) -> str:
    safe_url = url or "—"
    safe_msg = message or "未知错误"
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>原站预览失败</title>
  <style>
    body {{
      margin: 0;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      background: #f8fafc;
      color: #1f2937;
      font-family: system-ui, -apple-system, sans-serif;
      padding: 24px;
    }}
    .card {{
      max-width: 760px;
      background: white;
      border: 1px solid #cbd5e1;
      border-radius: 14px;
      padding: 22px 24px;
      box-shadow: 0 10px 35px rgba(15, 23, 42, 0.08);
    }}
    h1 {{ font-size: 18px; margin-bottom: 10px; }}
    p  {{ line-height: 1.6; margin-bottom: 10px; }}
    code {{
      display: block;
      white-space: pre-wrap;
      word-break: break-all;
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 10px 12px;
      color: #0f172a;
    }}
    a {{
      display: inline-block;
      margin-top: 8px;
      color: #2563eb;
      text-decoration: none;
      font-weight: 600;
    }}
  </style>
</head>
<body>
  <div class="card">
    <h1>原站嵌入失败</h1>
    <p>本地代理已经尝试拉取并重写页面，但当前页面仍无法稳定内嵌。</p>
    <p>{safe_msg}</p>
    <code>{safe_url}</code>
    <a href="{safe_url}" target="_blank" rel="noopener">在新标签打开原站</a>
  </div>
</body>
</html>"""


def _proxy_remote_response(remote_url: str):
    method = request.method.upper()
    body = request.get_data() if method not in {"GET", "HEAD"} else None
    status, final_url, headers, payload = fetch_remote_resource(
        remote_url,
        method=method,
        body=body,
        headers=_forward_headers_from_browser(),
    )
    content_type = headers.get("Content-Type", "application/octet-stream")

    if _looks_like_html(content_type):
        html = _decode_text_body(payload, headers)
        html = rewrite_live_html(html, final_url or remote_url)
        resp = make_response(html, status)
        resp.headers["Content-Type"] = "text/html; charset=utf-8"
        resp.headers["X-Frame-Options"] = "SAMEORIGIN"
        return resp

    resp = make_response(payload, status)
    for header_name in ("Content-Type", "Cache-Control", "ETag", "Last-Modified", "Expires"):
        header_value = headers.get(header_name)
        if header_value:
            resp.headers[header_name] = header_value
    return resp


# ── 数据结构 ─────────────────────────────────────────────────────────────────

class Column:
    def __init__(
        self,
        path: str,
        field: str,
        label: str,
        id_order: list[str],
        meta: dict[str, dict],
        html_data: dict[str, str],
        total_in_file: int,
    ) -> None:
        self.uid = -1
        self.path = path
        self.field = field
        self.label = label
        self.id_order = id_order
        self.meta = meta
        self.html_data = html_data
        self.total_in_file = total_in_file
        self.loaded = len(id_order)


# ── 数据加载 ─────────────────────────────────────────────────────────────────

def _get_id(record: dict) -> str:
    return record.get("id") or record.get("_meta", {}).get("url") or record.get("url", "")


def _compress_stats(stats: dict) -> dict:
    if not stats:
        return {}
    return {
        "original_chars": stats.get("original_chars"),
        "cleaned_chars": stats.get("cleaned_chars"),
        "compression_ratio": stats.get("compression_ratio"),
        "media_total": stats.get("media", {}).get("total"),
        "media_with_size": stats.get("media", {}).get("with_size"),
        "script_truncated": stats.get("scripts", {}).get("inline_truncated"),
        "style_truncated": stats.get("styles", {}).get("inline_truncated"),
    }


def _extract_meta(record: dict) -> dict:
    meta = record.get("_meta", {})
    return {
        "url": meta.get("url") or record.get("id", ""),
        "final_url": meta.get("final_url", ""),
        "page_type": meta.get("page_type", []),
        "model": record.get("model", ""),
        "prompt_tokens": record.get("prompt_tokens"),
        "completion_tokens": record.get("completion_tokens"),
        "finish_reason": record.get("finish_reason", ""),
        "preprocess_stats": _compress_stats(record.get("preprocess_stats", {})),
    }


def load_column(
    path: str,
    field: str,
    label: str,
    sample_n: int,
    random_sample: bool,
    reference_ids: list[str] | None = None,
) -> Column:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    all_records: list[dict] = []
    total = 0

    logger.info(f"[load] 读取 {path}  field={field}")
    with p.open(encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            total += 1
            if field in obj:
                all_records.append(obj)

    if reference_ids is not None:
        ref_index = {rid: idx for idx, rid in enumerate(reference_ids)}
        selected = [obj for obj in all_records if _get_id(obj) in ref_index]
        selected.sort(key=lambda obj: ref_index[_get_id(obj)])
    elif random_sample and 0 < sample_n < len(all_records):
        selected = random.sample(all_records, sample_n)
    elif 0 < sample_n:
        selected = all_records[:sample_n]
    else:
        selected = all_records

    id_order: list[str] = []
    meta: dict[str, dict] = {}
    html_data: dict[str, str] = {}

    for obj in selected:
        rec_id = _get_id(obj)
        if not rec_id or rec_id in html_data:
            continue
        id_order.append(rec_id)
        html_data[rec_id] = obj[field]
        meta[rec_id] = _extract_meta(obj)

    col = Column(path, field, label, id_order, meta, html_data, total)
    logger.info(f"[load] 完成: {col.loaded:,} 条（文件总行 {total:,}）label={label!r}")
    return col


def detect_fields(path: str) -> list[str]:
    found: set[str] = set()
    try:
        with Path(path).open(encoding="utf-8") as f:
            for idx, raw in enumerate(f):
                if idx >= 20:
                    break
                try:
                    obj = json.loads(raw)
                except Exception:
                    continue
                for field in (FIELD_PREPROCESSED, FIELD_OUTPUT):
                    if field in obj:
                        found.add(field)
    except Exception:
        pass
    return [field for field in (FIELD_PREPROCESSED, FIELD_OUTPUT) if field in found]


# ── 全局状态 ─────────────────────────────────────────────────────────────────

_columns: list[Column] = []
_row_ids: list[str] = []
_next_col_uid = 0
_state_lock = threading.Lock()


def _recompute_row_ids() -> None:
    global _row_ids
    _row_ids = list(_columns[0].id_order) if _columns else []
    logger.info(f"[rows] 基准行数: {len(_row_ids):,}")


def _assign_uid(col: Column) -> None:
    global _next_col_uid
    col.uid = _next_col_uid
    _next_col_uid += 1


def _row_meta_at(row_idx: int) -> tuple[str, dict] | None:
    with _state_lock:
        if row_idx < 0 or row_idx >= len(_row_ids):
            return None
        rec_id = _row_ids[row_idx]
        for col in _columns:
            if rec_id in col.meta:
                return rec_id, col.meta[rec_id]
    return None


def _preferred_live_url(meta: dict) -> str:
    return meta.get("final_url") or meta.get("url") or ""


# ── Flask routes ──────────────────────────────────────────────────────────────

_HTML_FILE = Path(__file__).parent / "viewer.html"

_PLACEHOLDER_HTML = """<!DOCTYPE html>
<html><body style="
  margin:0;display:flex;align-items:center;justify-content:center;
  height:100vh;font-family:sans-serif;font-size:14px;color:#94a3b8;background:#f8fafc;">
  — 无数据 —
</body></html>"""


@app.route("/")
def index():
    return _HTML_FILE.read_text(encoding="utf-8"), 200, {
        "Content-Type": "text/html; charset=utf-8"
    }


@app.route("/render/<int:col_idx>/<int:row_idx>")
def render_html(col_idx: int, row_idx: int):
    with _state_lock:
        if col_idx < 0 or col_idx >= len(_columns):
            return _PLACEHOLDER_HTML, 200, {"Content-Type": "text/html; charset=utf-8"}
        if row_idx < 0 or row_idx >= len(_row_ids):
            return _PLACEHOLDER_HTML, 200, {"Content-Type": "text/html; charset=utf-8"}
        col = _columns[col_idx]
        rec_id = _row_ids[row_idx]
        html = col.html_data.get(rec_id)
        meta = col.meta.get(rec_id, {})

    if not html:
        html = _PLACEHOLDER_HTML
    else:
        html = inject_base_url(html, meta.get("final_url") or meta.get("url", ""))

    resp = make_response(html, 200)
    resp.headers["Content-Type"] = "text/html; charset=utf-8"
    resp.headers["X-Frame-Options"] = "SAMEORIGIN"
    return resp


@app.route("/live/<int:row_idx>")
def render_live(row_idx: int):
    row_meta = _row_meta_at(row_idx)
    if row_meta is None:
        return _PLACEHOLDER_HTML, 200, {"Content-Type": "text/html; charset=utf-8"}

    _, meta = row_meta
    live_url = _preferred_live_url(meta)
    if not live_url:
        return _build_proxy_error_page("", "这一行没有可用 URL"), 200, {
            "Content-Type": "text/html; charset=utf-8"
        }

    try:
        return _proxy_remote_response(live_url)
    except Exception as exc:
        logger.warning(f"[live] 预览失败 row={row_idx} url={live_url}: {exc}")
        html = _build_proxy_error_page(live_url, str(exc))
        return html, 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/proxy/<scheme>/<path:remote_path>", methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
def proxy_remote(scheme: str, remote_path: str):
    if scheme not in PROXYABLE_SCHEMES:
        return "unsupported scheme", 400

    netloc, has_slash, path_tail = remote_path.partition("/")
    if not netloc:
        return "missing host", 400

    remote_url = urlunsplit((
        scheme,
        netloc,
        f"/{path_tail}" if has_slash else "/",
        request.query_string.decode("utf-8", errors="ignore"),
        "",
    ))
    try:
        return _proxy_remote_response(remote_url)
    except Exception as exc:
        logger.warning(f"[proxy] 请求失败 url={remote_url}: {exc}")
        wants_html = "text/html" in request.headers.get("Accept", "")
        if wants_html:
            html = _build_proxy_error_page(remote_url, str(exc))
            return html, 502, {"Content-Type": "text/html; charset=utf-8"}
        return str(exc), 502


@app.route("/source/column/<int:col_idx>/<int:row_idx>")
def source_column(col_idx: int, row_idx: int):
    with _state_lock:
        if col_idx < 0 or col_idx >= len(_columns):
            return "— 无数据 —", 200, {"Content-Type": "text/plain; charset=utf-8"}
        if row_idx < 0 or row_idx >= len(_row_ids):
            return "— 无数据 —", 200, {"Content-Type": "text/plain; charset=utf-8"}
        col = _columns[col_idx]
        rec_id = _row_ids[row_idx]
        html = col.html_data.get(rec_id, "")

    return html or "— 无数据 —", 200, {"Content-Type": "text/plain; charset=utf-8"}


@app.route("/source/live/<int:row_idx>")
def source_live(row_idx: int):
    row_meta = _row_meta_at(row_idx)
    if row_meta is None:
        return "— 无数据 —", 200, {"Content-Type": "text/plain; charset=utf-8"}

    _, meta = row_meta
    live_url = _preferred_live_url(meta)
    if not live_url:
        return "— 无可用 URL —", 200, {"Content-Type": "text/plain; charset=utf-8"}

    try:
        status, final_url, headers, payload = fetch_remote_resource(
            live_url,
            method="GET",
            headers={
                "User-Agent": REMOTE_USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        )
    except Exception as exc:
        return f"加载失败: {exc}", 502, {"Content-Type": "text/plain; charset=utf-8"}

    content_type = headers.get("Content-Type", "")
    if not _looks_like_html(content_type):
        msg = f"返回内容不是 HTML: {content_type or 'unknown'}"
        return msg, 502, {"Content-Type": "text/plain; charset=utf-8"}

    text = _decode_text_body(payload, headers)
    if final_url and final_url != live_url:
        text = f"<!-- final_url: {final_url} -->\n{text}"
    return text, status, {"Content-Type": "text/plain; charset=utf-8"}


@app.route("/api/info")
def api_info():
    with _state_lock:
        return jsonify({
            "columns": [
                {
                    "idx": idx,
                    "uid": col.uid,
                    "path": col.path,
                    "field": col.field,
                    "label": col.label,
                    "total_in_file": col.total_in_file,
                    "loaded": col.loaded,
                }
                for idx, col in enumerate(_columns)
            ],
            "row_count": len(_row_ids),
        })


@app.route("/api/rows")
def api_rows():
    page = max(0, int(request.args.get("page", 0)))
    size = max(1, min(100, int(request.args.get("size", 10))))

    with _state_lock:
        start = page * size
        page_ids = _row_ids[start:start + size]
        rows = []
        for abs_row_idx, rec_id in enumerate(page_ids, start=start):
            row_meta: dict = {}
            for col in _columns:
                if rec_id in col.meta:
                    row_meta = col.meta[rec_id]
                    break

            cells = []
            for col in _columns:
                if rec_id in col.html_data:
                    meta = col.meta.get(rec_id, {})
                    cells.append({
                        "has_html": True,
                        "html_len": len(col.html_data[rec_id]),
                        "model": meta.get("model", ""),
                        "prompt_tokens": meta.get("prompt_tokens"),
                        "completion_tokens": meta.get("completion_tokens"),
                        "finish_reason": meta.get("finish_reason", ""),
                    })
                else:
                    cells.append({"has_html": False})

            rows.append({
                "row_idx": abs_row_idx,
                "id": rec_id,
                "meta": row_meta,
                "cells": cells,
            })

    return jsonify({
        "page": page,
        "size": size,
        "total_rows": len(_row_ids),
        "rows": rows,
    })


@app.route("/api/columns", methods=["POST"])
def api_add_column():
    body = request.get_json(force=True) or {}
    path = (body.get("path") or "").strip()
    if not path:
        return jsonify({"error": "path required"}), 400

    field = body.get("field", FIELD_OUTPUT)
    if field not in (FIELD_PREPROCESSED, FIELD_OUTPUT):
        return jsonify({"error": "field 必须是 preprocessed_html 或 output_html"}), 400

    sample_n = int(body.get("sample_n", 50))
    random_sample = bool(body.get("random", False))
    label = (body.get("label") or "").strip()
    if not label:
        stem = Path(path).stem
        label = f"{stem} [{'preprocessed' if 'preprocessed' in field else 'output'}]"

    try:
        with _state_lock:
            reference_ids = list(_columns[0].id_order) if _columns else None
        col = load_column(path, field, label, sample_n, random_sample, reference_ids)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    with _state_lock:
        _assign_uid(col)
        _columns.append(col)
        _recompute_row_ids()

    return jsonify({"ok": True, "col_idx": len(_columns) - 1, "uid": col.uid})


@app.route("/api/columns/<int:idx>", methods=["DELETE"])
def api_remove_column(idx: int):
    with _state_lock:
        if idx < 0 or idx >= len(_columns):
            return jsonify({"error": "index out of range"}), 400
        _columns.pop(idx)
        _recompute_row_ids()
    return jsonify({"ok": True})


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HTML 改写结果多列对比可视化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  # 单文件（生成 preprocessed + output 两列）
  python viewer.py --jsonl output.jsonl --sample-n 20

  # 对比两个模型配置（首文件生成 preprocessed + output，第二文件只生成 output）
  python viewer.py --jsonl output_a.jsonl output_b.jsonl --sample-n 20

  # 只看预处理结果（Stage 1 产出）
  python viewer.py --jsonl preprocessed.jsonl --sample-n 50

  # WSL 下从 Windows 浏览器访问
  python viewer.py --jsonl output.jsonl --host 0.0.0.0 --port 7862
""",
    )
    parser.add_argument("--jsonl", nargs="*", default=[], metavar="FILE")
    parser.add_argument("--sample-n", type=int, default=20, help="加载条数，-1=全部")
    parser.add_argument("--random-sample", action="store_true")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    is_first_file = True
    for path in args.jsonl or []:
        fields = detect_fields(path)
        if not fields:
            logger.warning(f"未找到可用 HTML 字段 (preprocessed_html/output_html): {path}")
            continue

        show_fields = fields if is_first_file else [f for f in fields if f == FIELD_OUTPUT] or fields
        stem = Path(path).stem

        for field in show_fields:
            field_short = "preprocessed" if "preprocessed" in field else "output"
            label = f"{stem} [{field_short}]"
            try:
                with _state_lock:
                    reference_ids = list(_columns[0].id_order) if _columns else None
                col = load_column(path, field, label, args.sample_n, args.random_sample, reference_ids)
                with _state_lock:
                    _assign_uid(col)
                    _columns.append(col)
                    _recompute_row_ids()
            except Exception as exc:
                logger.error(f"加载失败 {path} field={field}: {exc}")
                sys.exit(1)

        is_first_file = False

    url = f"http://localhost:{args.port}"
    logger.info(f"启动 → {url}")
    print(f"\n  HTML 改写对比工具已就绪: {url}\n")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
