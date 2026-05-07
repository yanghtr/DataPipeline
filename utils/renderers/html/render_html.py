#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import base64
import hashlib
import argparse
import traceback
import sys
import time
import threading
import mimetypes
import urllib.request
import urllib.error
import ssl
from multiprocessing import Pool, cpu_count, Manager
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from PIL import Image
from tqdm import tqdm

try:
    from preprocess_html import preprocess_html_string as _preprocess_html_string
except ImportError:
    def _preprocess_html_string(html: str) -> str:  # type: ignore[misc]
        return html


# =========================================================
# ID 清洗 / 文件格式加载
# =========================================================
_SANITIZE_RE = re.compile(r'[^\w\-.]')

# 这些域的资源（字体、图标等）加载失败不影响核心渲染质量，降为 WARN
_DECORATIVE_DOMAINS = frozenset([
    "fonts.googleapis.com",
    "fonts.gstatic.com",
    "use.fontawesome.com",
    "ka-f.fontawesome.com",
])


def _is_decorative_url(url: str) -> bool:
    try:
        from urllib.parse import urlparse
        host = urlparse(url).netloc.lower()
        return any(host == d or host.endswith("." + d) for d in _DECORATIVE_DOMAINS)
    except Exception:
        return False


def _sanitize_id(raw_id) -> str:
    """将任意字符串转为合法文件名（保留字母数字 - . _，其余换成 _）。"""
    return _SANITIZE_RE.sub('_', str(raw_id))[:200]


def _load_records(path: Path) -> list[dict]:
    """从 JSON 数组(.json)或 JSONL(.jsonl) 文件加载记录列表。"""
    if path.suffix.lower() == ".jsonl":
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# 统一日志接口
# =========================================================
def _log(level, category, log_id, msg="", **fields):
    parts = [f"[{level}]", f"[{category}]", f"id={log_id}"]
    for k, v in fields.items():
        parts.append(f"{k}={v}")
    if msg:
        parts.append(f"msg={msg}")
    line = " | ".join(parts)
    try:
        sys.stderr.write(line + "\n")
        sys.stderr.flush()
    except Exception:
        pass


def _log_warn(category, log_id, msg="", **fields):
    _log("WARN", category, log_id, msg, **fields)


def _log_error(category, log_id, msg="", **fields):
    _log("ERROR", category, log_id, msg, **fields)


# =========================================================
# 运行时检测:页面无限拉长
# ---------------------------------------------------------
# 思路:
#   在页面上注入一个高度采样器,每 ~200ms 记录一次 scrollHeight。
#   Python 侧在等待阶段周期性读取样本,用一个滑动窗口判定:
#     - 窗口时长 >= min_window_ms
#     - 窗口内高度单调不减(允许 monotonic_tolerance_px 抖动)
#     - 窗口内净增长 >= growth_threshold_px
#     - 末尾仍在增长(排除"先涨后平"的合法首屏布局)
#   命中就 WARN 并通知调用方提前跳出等待。
#
# 为什么比静态 HTML 扫描更鲁棒:
#   - 不用去枚举各种触发模式(Chart.js maintainAspectRatio:false、
#     canvas height 属性、ECharts 同类 bug、ResizeObserver 死循环、
#     死循环 appendChild 等)
#   - 直接看"症状"(页面一直涨),对未来新的拉长来源自然有效
# =========================================================
_GROWTH_MONITOR_INIT = r"""
(() => {
    if (window.__growthMonitor) return;
    const samples = [];
    const startedAt = Date.now();

    function sample() {
        try {
            const h = Math.max(
                document.documentElement.scrollHeight || 0,
                document.body ? document.body.scrollHeight : 0
            );
            samples.push({ t: Date.now() - startedAt, h: h });
            if (samples.length > 600) samples.shift();
        } catch (e) {}
    }

    sample();
    const timer = setInterval(sample, 200);

    window.__growthMonitor = {
        samples: samples,
        stop: () => clearInterval(timer),
        snapshot: () => samples.slice(),
    };
})();
"""


def _install_growth_monitor(page):
    try:
        page.evaluate(_GROWTH_MONITOR_INIT)
        return True
    except Exception:
        return False


def _read_growth_samples(page):
    try:
        return page.evaluate(
            "() => (window.__growthMonitor && window.__growthMonitor.snapshot()) || []"
        )
    except Exception:
        return []


def _analyze_growth(samples,
                    min_window_ms=3000,
                    growth_threshold_px=2000,
                    monotonic_tolerance_px=2):
    """
    返回 (is_growing, info_dict)。命中时 info 含窗口时长、净增长、首尾高度等。
    """
    if not samples or len(samples) < 3:
        return False, None

    now_t = samples[-1]['t']
    window = [s for s in samples
              if now_t - s['t'] <= max(min_window_ms * 2, min_window_ms + 1000)]
    if len(window) < 3:
        return False, None

    window_ms = window[-1]['t'] - window[0]['t']
    if window_ms < min_window_ms:
        return False, None

    # 判断是否为单调递增
    for i in range(1, len(window)):
        if window[i]['h'] < window[i - 1]['h'] - monotonic_tolerance_px:
            return False, None

    delta = window[-1]['h'] - window[0]['h']
    if delta < growth_threshold_px:
        return False, None

    if len(window) >= 2 and window[-1]['h'] <= window[-2]['h']:
        return False, None

    return True, {
        'window_ms': window_ms,
        'delta_px': delta,
        'start_h': window[0]['h'],
        'end_h': window[-1]['h'],
        'samples': len(window),
    }


def _check_growth_and_warn(page, log_id, warned_flag,
                           min_window_ms, growth_threshold_px):
    """
    等待期间周期性调用。命中一次就 WARN,并把 warned_flag['hit'] 置 True。
    返回是否"已命中过"(包含本次之前的命中),供调用方据此决定是否提前跳出。
    """
    if warned_flag.get('hit'):
        return True
    samples = _read_growth_samples(page)
    is_growing, info = _analyze_growth(
        samples,
        min_window_ms=min_window_ms,
        growth_threshold_px=growth_threshold_px,
    )
    if not is_growing:
        return False
    _log_warn(
        "INFINITE_GROWTH",
        log_id,
        window_ms=info['window_ms'],
        delta_px=info['delta_px'],
        start_h=info['start_h'],
        end_h=info['end_h'],
        samples=info['samples'],
        msg="page height grew monotonically -> likely infinite stretch",
    )
    warned_flag['hit'] = True
    return True


# =========================================================
# 系统负载自适应
# =========================================================
def _system_load_factor():
    try:
        load1, _, _ = os.getloadavg()
        ncpu = os.cpu_count() or 1
        ratio = load1 / ncpu
        if ratio < 0.8:
            return 1.0
        elif ratio < 1.5:
            return 2.0
        elif ratio < 2.5:
            return 3.5
        else:
            return 5.0
    except Exception:
        return 1.0


# =========================================================
# CDN 缓存系统
# =========================================================
_URL_PATTERNS = [
    re.compile(r"""<script\b[^>]*\bsrc\s*=\s*["']([^"']+)["']""", re.IGNORECASE),
    re.compile(r"""<link\b[^>]*\bhref\s*=\s*["']([^"']+)["']""", re.IGNORECASE),
    re.compile(r"""@import\s+(?:url\()?\s*["']([^"']+)["']""", re.IGNORECASE),
]


def _url_to_cache_path(url, cache_dir):
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()
    ext = ""
    path_part = url.split("?", 1)[0].split("#", 1)[0]
    m = re.search(r"\.([a-zA-Z0-9]{1,5})$", path_part)
    if m:
        ext = "." + m.group(1).lower()
    return os.path.join(cache_dir, h + ext)


def _extract_urls_from_html(html):
    urls = set()
    if not html:
        return urls
    for pat in _URL_PATTERNS:
        for m in pat.findall(html):
            if m.startswith("http://") or m.startswith("https://"):
                urls.add(m.strip())
    return urls


def _collect_all_urls(json_files, json_dir, html_field="html", preprocess=True):
    all_urls = set()
    for fname in json_files:
        fpath = json_dir / fname
        if not fpath.exists():
            continue
        try:
            records = _load_records(fpath)
        except Exception as e:
            print(f"[WARN] Failed to read {fpath}: {e}")
            continue
        for data in tqdm(records[:100]):
            html = data.get(html_field, "")
            if preprocess and html:
                html = _preprocess_html_string(html)
            all_urls.update(_extract_urls_from_html(html))
    return all_urls


def _download_one(url, cache_dir, timeout=30):
    path = _url_to_cache_path(url, cache_dir)
    meta_path = path + ".meta"
    if os.path.exists(path) and os.path.exists(meta_path):
        return (url, True, "cached")

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Linux; Playwright) AppleWebKit/537.36",
            "Accept": "*/*",
        })
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            content = resp.read()
            content_type = resp.headers.get("Content-Type", "")
        tmp_path = path + ".tmp"
        with open(tmp_path, "wb") as f:
            f.write(content)
        os.replace(tmp_path, path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"url": url, "content_type": content_type}, f)
        return (url, True, "downloaded")
    except urllib.error.HTTPError as e:
        return (url, False, f"HTTP {e.code}")
    except Exception as e:
        return (url, False, f"{type(e).__name__}: {e}")


def prepare_cdn_cache(json_files, json_dir, cache_dir, concurrency=16,
                      html_field="html", preprocess=True):
    os.makedirs(cache_dir, exist_ok=True)

    print(f"\n[CDN] Scanning JSON files for external URLs...")
    all_urls = _collect_all_urls(json_files, json_dir,
                                 html_field=html_field, preprocess=preprocess)
    print(f"[CDN] Found {len(all_urls)} unique external URLs")

    if not all_urls:
        return set(), {}

    succeeded = set()
    failed = {}

    print(f"[CDN] Downloading (concurrency={concurrency})...")
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {ex.submit(_download_one, url, cache_dir): url for url in all_urls}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="CDN download", file=sys.stdout, dynamic_ncols=True):
            url, ok, reason = fut.result()
            if ok:
                succeeded.add(url)
            else:
                failed[url] = reason

    print(f"[CDN] Cache ready: {len(succeeded)} succeeded, {len(failed)} failed")
    if failed:
        for url, reason in list(failed.items())[:10]:
            print(f"[CDN] FAILED: {reason} | {url}")
        if len(failed) > 10:
            print(f"[CDN]   ... and {len(failed) - 10} more failures")
    return succeeded, failed


_MIME_MAP = {
    ".js": "application/javascript",
    ".mjs": "application/javascript",
    ".css": "text/css",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".ttf": "font/ttf",
    ".otf": "font/otf",
    ".html": "text/html",
}


def _get_mime_from_path(path, fallback="application/octet-stream"):
    ext = os.path.splitext(path)[1].lower()
    return _MIME_MAP.get(ext, mimetypes.guess_type(path)[0] or fallback)


class _BodyCache:
    def __init__(self, max_entries=64, max_total_bytes=64 * 1024 * 1024):
        self.d = OrderedDict()
        self.max_entries = max_entries
        self.max_total_bytes = max_total_bytes
        self.total = 0

    def get(self, key):
        v = self.d.get(key)
        if v is not None:
            self.d.move_to_end(key)
        return v

    def put(self, key, body, content_type):
        size = len(body)
        if size > self.max_total_bytes // 2:
            return
        if key in self.d:
            old_body, _ = self.d[key]
            self.total -= len(old_body)
            del self.d[key]
        self.d[key] = (body, content_type)
        self.total += size
        while (len(self.d) > self.max_entries or
               self.total > self.max_total_bytes):
            _, (old_body, _) = self.d.popitem(last=False)
            self.total -= len(old_body)


def make_cdn_interceptor(cache_dir, body_cache):
    def handler(route):
        request = route.request
        url = request.url
        if not (url.startswith("http://") or url.startswith("https://")):
            route.continue_()
            return

        cache_path = _url_to_cache_path(url, cache_dir)

        cached = body_cache.get(cache_path)
        if cached is not None:
            body, content_type = cached
            route.fulfill(status=200, body=body, headers={
                "Content-Type": content_type,
                "Access-Control-Allow-Origin": "*",
                "Cache-Control": "public, max-age=31536000",
            })
            return

        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    body = f.read()
                content_type = None
                meta_path = cache_path + ".meta"
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, "r", encoding="utf-8") as mf:
                            meta = json.load(mf)
                        content_type = meta.get("content_type") or None
                    except Exception:
                        pass
                if not content_type:
                    content_type = _get_mime_from_path(cache_path)

                body_cache.put(cache_path, body, content_type)
                route.fulfill(status=200, body=body, headers={
                    "Content-Type": content_type,
                    "Access-Control-Allow-Origin": "*",
                    "Cache-Control": "public, max-age=31536000",
                })
                return
            except Exception:
                pass

        route.continue_()

    return handler


# =========================================================
# 运行时检测:非父子关系文本元素相互叠压
# ---------------------------------------------------------
# 思路:
#   在 networkidle 之后、截图之前,向页面注入一段 JS,枚举所有可见的
#   文本叶节点(p/h1-h6/span/li/td/th/label/a/button),逐对检测其
#   getBoundingClientRect 是否有显著重叠(排除祖先-后代关系)。
#   命中时记录 WARN,不阻断截图流程。
#
# 为什么排除祖先-后代关系:
#   嵌套元素(如 <h1><span>…</span></h1>)的边界框天然重叠,
#   但那是合法的 DOM 结构,不是渲染错误。
#
# 触发阈值:
#   - 重叠面积 / 较小元素面积 > overlap_ratio_threshold (默认 0.15)
#   - 重叠宽度和高度均 > 5px
# =========================================================
_TEXT_OVERLAP_CHECK_JS = r"""
(threshold) => {
    'use strict';
    function hasDirectText(el) {
        for (const c of el.childNodes)
            if (c.nodeType === Node.TEXT_NODE && c.textContent.trim().length > 1)
                return true;
        return false;
    }
    const transformCache = new WeakMap();
    function hasTransformedAncestor(el) {
        let cur = el.parentElement;
        while (cur && cur.tagName !== 'BODY') {
            if (transformCache.has(cur)) return transformCache.get(cur);
            const t = window.getComputedStyle(cur).transform;
            if (t && t !== 'none') { transformCache.set(cur, true); return true; }
            cur = cur.parentElement;
        }
        return false;
    }
    const allEls = Array.from(document.querySelectorAll(
        'p,h1,h2,h3,h4,h5,h6,span,li,td,th,label,a,button'
    ));
    const rects = [];
    for (const el of allEls) {
        if (!hasDirectText(el)) continue;
        const st = window.getComputedStyle(el);
        if (st.display === 'none' || st.visibility === 'hidden'
                || parseFloat(st.opacity) < 0.1) continue;
        if (hasTransformedAncestor(el)) continue;
        const r = el.getBoundingClientRect();
        if (r.width < 5 || r.height < 5) continue;
        rects.push({
            el,
            tag: el.tagName,
            text: el.textContent.trim().substring(0, 60),
            top:    r.top    + window.scrollY,
            left:   r.left,
            bottom: r.bottom + window.scrollY,
            right:  r.right,
            pos:    st.position,
            parent: el.parentElement,
        });
    }
    const normalFlow = p => p === 'static' || p === 'relative';
    const overlaps = [];
    const LIMIT = Math.min(rects.length, 300);
    for (let i = 0; i < LIMIT && overlaps.length < 20; i++) {
        for (let j = i + 1; j < LIMIT && overlaps.length < 20; j++) {
            const a = rects[i], b = rects[j];
            if (a.el.contains(b.el) || b.el.contains(a.el)) continue;
            if (a.parent === b.parent && normalFlow(a.pos) && normalFlow(b.pos)) continue;
            const oh = Math.min(a.right, b.right) - Math.max(a.left, b.left);
            const ov = Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top);
            if (oh <= 5 || ov <= 5) continue;
            const aArea = (a.right - a.left) * (a.bottom - a.top);
            const bArea = (b.right - b.left) * (b.bottom - b.top);
            if (aArea < 1 || bArea < 1) continue;
            const ratio = (oh * ov) / Math.min(aArea, bArea);
            if (ratio > threshold) {
                overlaps.push({
                    a_tag: a.tag, a_text: a.text,
                    b_tag: b.tag, b_text: b.text,
                    overlap_w: Math.round(oh),
                    overlap_h: Math.round(ov),
                    ratio: Math.round(ratio * 100),
                });
            }
        }
    }
    return overlaps;
}
"""


def _check_text_overlap_and_warn(page, log_id, overlap_ratio_threshold=0.20):
    """
    检测页面内非父子关系的文本元素是否相互叠压。
    命中时记录 WARN,返回 True。
    """
    try:
        overlaps = page.evaluate(_TEXT_OVERLAP_CHECK_JS, overlap_ratio_threshold)
    except Exception:
        return False
    if not overlaps:
        return False
    worst = max(overlaps, key=lambda x: x['ratio'])
    _log_warn(
        "TEXT_OVERLAP",
        log_id,
        pairs=len(overlaps),
        worst_ratio=worst['ratio'],
        worst_a=f"{worst['a_tag']}:{worst['a_text'][:30]}",
        worst_b=f"{worst['b_tag']}:{worst['b_text'][:30]}",
        msg="overlapping text elements detected",
    )
    return True


# =========================================================
# 注入脚本:全局禁用常见图表库的动画
# =========================================================
_INIT_SCRIPT = r"""
(() => {
    let _Chart;
    Object.defineProperty(window, 'Chart', {
        configurable: true,
        get() { return _Chart; },
        set(v) {
            _Chart = v;
            try {
                if (v && v.defaults) {
                    v.defaults.animation = false;
                    v.defaults.animations = false;
                    if (v.defaults.transitions) {
                        Object.keys(v.defaults.transitions).forEach(k => {
                            v.defaults.transitions[k] = { animation: { duration: 0 } };
                        });
                    }
                }
            } catch (e) {}
        }
    });

    let _echarts;
    Object.defineProperty(window, 'echarts', {
        configurable: true,
        get() { return _echarts; },
        set(v) {
            _echarts = v;
            try {
                if (v && typeof v.init === 'function') {
                    const origInit = v.init;
                    v.init = function(...args) {
                        const inst = origInit.apply(this, args);
                        const origSetOption = inst.setOption;
                        inst.setOption = function(opt, ...rest) {
                            if (opt && typeof opt === 'object') {
                                opt.animation = false;
                            }
                            return origSetOption.call(this, opt, ...rest);
                        };
                        return inst;
                    };
                }
            } catch (e) {}
        }
    });
})();
"""


# =========================================================
# 单页渲染
# =========================================================
def _render_one(task, context, body_cache,
                max_wait_ms, idle_timeout_ms,
                stable_frames, page_total_timeout_ms, factor,
                enable_growth_check=True,
                growth_min_window_ms=3000,
                growth_threshold_px=2000,
                enable_overlap_check=True,
                overlap_ratio_threshold=0.15,
                tall_ratio_threshold=3.0):
    item_id = task["id"]
    log_id = task["log_id"]
    html = task["html"]
    output_dir = task["output_dir"]
    cache_dir = task["cache_dir"]

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{item_id}.png")

    if os.path.exists(out_path):
        return out_path

    if not html or not html.strip():
        _log_error("BAD_DATA", log_id, msg="empty html")
        Image.new("RGB", (1280, 960), color="white").save(out_path)
        return out_path

    data_url = "data:text/html;base64," + base64.b64encode(html.encode("utf-8")).decode("utf-8")

    failed_requests = []
    http_errors = []
    page_errors = []
    console_errors = []
    pending_requests = {}
    growth_flag = {'hit': False}

    page = None
    page_start = time.time()

    def _remaining_ms():
        return max(1000, page_total_timeout_ms - int((time.time() - page_start) * 1000))

    try:
        page = context.new_page()

        if cache_dir:
            page.route("**/*", make_cdn_interceptor(cache_dir, body_cache))

        # -------- 事件监听 --------
        def _on_request(req):
            pending_requests[req.url] = req.resource_type

        def _on_request_finished(req):
            pending_requests.pop(req.url, None)

        def _on_request_failed(req):
            pending_requests.pop(req.url, None)
            failed_requests.append((req.url, req.resource_type, str(req.failure)))

        def _on_response(res):
            try:
                if res.status >= 400:
                    http_errors.append((res.url, res.status, res.request.resource_type))
            except Exception:
                pass

        def _on_page_error(err):
            page_errors.append(str(err))

        def _on_console(msg):
            if msg.type == "error":
                console_errors.append(msg.text)

        page.on("request", _on_request)
        page.on("requestfinished", _on_request_finished)
        page.on("requestfailed", _on_request_failed)
        page.on("response", _on_response)
        page.on("pageerror", _on_page_error)
        page.on("console", _on_console)

        # -------- 尽早装上页面高度监测器(init script,比 domcontentloaded 还早) --------
        # 这样即使 CDN 全部缓存、networkidle 瞬间触发,也能在 goto 期间积累足够样本。
        if enable_growth_check:
            try:
                page.add_init_script(_GROWTH_MONITOR_INIT)
            except Exception:
                pass

        # -------- 导航 --------
        try:
            page.goto(data_url, timeout=min(30000, _remaining_ms()),
                      wait_until="domcontentloaded")
        except PlaywrightTimeoutError as e:
            _log_error("TIMEOUT", log_id, stage="goto", msg=str(e) or "goto timeout")
            for url, rtype in list(pending_requests.items()):
                _log_warn("PENDING_REQUEST", log_id, type=rtype, url=url)
            raise

        # -------- 补装监测器(幂等,应对 add_init_script 未生效的情况) --------
        if enable_growth_check:
            _install_growth_monitor(page)

        # -------- 滚动触发 paint --------
        try:
            page.evaluate("""
                async () => {
                    const distance = 300;
                    const delay = 50;
                    const height = document.body.scrollHeight;
                    for (let y = 0; y < height; y += distance) {
                        window.scrollTo(0, y);
                        await new Promise(r => setTimeout(r, delay));
                    }
                    window.scrollTo(0, 0);
                }
            """)
        except Exception as e:
            _log_warn("RENDER_ERROR", log_id, stage="scroll", msg=str(e) or repr(e))

        # -------- 渲染完成等待 --------
        # 原实现是 page.wait_for_load_state('networkidle', timeout=cur_max_wait),
        # 一次阻塞 cur_max_wait 毫秒。为了能在等待期间做拉长检测,
        # 这里改成切片等待:每次 wait 一小段,中间插检测。
        # 单次切片用较小的 timeout,networkidle 命中就正常返回;不命中就抛超时,
        # 我们忽略超时继续下一片,累计总等待时长 >= cur_max_wait 再宣布放弃。
        cur_max_wait = min(max_wait_ms, _remaining_ms())
        slice_ms = 500
        wait_ok = False

        for attempt in range(2):
            if _remaining_ms() < 1500:
                break

            elapsed = 0
            while elapsed < cur_max_wait:
                step = min(slice_ms, cur_max_wait - elapsed, _remaining_ms())
                if step <= 0:
                    break
                try:
                    page.wait_for_load_state("networkidle", timeout=step)
                    wait_ok = True
                    break
                except PlaywrightTimeoutError:
                    pass
                elapsed += step

                # 周期性检测拉长。命中后停止继续等待 —— 再等也是白等,还会让 pbar 卡很久。
                if enable_growth_check and _check_growth_and_warn(
                    page, log_id, growth_flag,
                    growth_min_window_ms, growth_threshold_px,
                ):
                    break

            if wait_ok or growth_flag['hit']:
                break
            _log_warn("WAIT_TIMEOUT", log_id,
                      attempt=attempt, max_wait=cur_max_wait, factor=factor,
                      msg="networkidle timeout")
            cur_max_wait = min(int(cur_max_wait * 1.5), _remaining_ms())

        if not wait_ok and not growth_flag['hit']:
            _log_warn("WAIT_FINAL_TIMEOUT", log_id,
                      msg="proceeding to screenshot without stable signal")

        # 截图前确保监测器已积累足够时长的样本,再做最终判定。
        # 背景:当 CDN 全部命中缓存时 networkidle 几乎瞬间触发,wait 循环里根本来不及
        # 积累够 growth_min_window_ms 的样本——导致漏判。此处在不超过剩余时间的前提下,
        # 补等到样本窗口满 growth_min_window_ms,每 200 ms 检测一次。
        if enable_growth_check and not growth_flag['hit']:
            samples = _read_growth_samples(page)
            if samples:
                monitor_span_ms = samples[-1]['t'] - samples[0]['t']
                needed_ms = growth_min_window_ms + 500
                if monitor_span_ms < needed_ms:
                    extra_budget = min(needed_ms - monitor_span_ms,
                                       _remaining_ms() - 1000)
                    extra_step_ms = 200
                    extra_elapsed = 0
                    while extra_elapsed < extra_budget and not growth_flag['hit']:
                        time.sleep(extra_step_ms / 1000.0)
                        extra_elapsed += extra_step_ms
                        if _check_growth_and_warn(
                            page, log_id, growth_flag,
                            growth_min_window_ms, growth_threshold_px,
                        ):
                            break

        if enable_growth_check:
            _check_growth_and_warn(
                page, log_id, growth_flag,
                growth_min_window_ms, growth_threshold_px,
            )

        # -------- 文字叠压检测 --------
        if enable_overlap_check:
            _check_text_overlap_and_warn(page, log_id, overlap_ratio_threshold)

        # -------- 截图 --------
        # 注意:即使检测到无限拉长,这里仍然 full_page 截图,会拍出一张巨长的图。
        # 这是有意为之 —— WARN 已经出了,是否丢弃由下游根据 log 决定。
        # 如果想直接丢弃,可以在这里按 growth_flag['hit'] 置白图。
        try:
            page.screenshot(path=out_path, full_page=True,
                            animations="disabled",
                            timeout=max(10000, _remaining_ms()))
        except PlaywrightTimeoutError as e:
            _log_error("TIMEOUT", log_id, stage="screenshot",
                       msg=str(e) or "screenshot timeout")
            raise
        except Exception as e:
            _log_error("RENDER_ERROR", log_id, stage="screenshot",
                       msg=str(e) or repr(e))
            raise

    except Exception as e:
        if not isinstance(e, PlaywrightTimeoutError):
            _log_error("RENDER_ERROR", log_id, msg=str(e) or repr(e))
        try:
            if not os.path.exists(out_path):
                Image.new("RGB", (1280, 960), color="white").save(out_path)
        except Exception:
            pass
    finally:
        if page is not None:
            try:
                page.close()
            except Exception:
                pass

    # -------- 汇总异常 --------
    # 字体/装饰性 CDN 失败不影响核心渲染，降为 WARN 避免污染 issues
    for url, rtype, reason in failed_requests:
        if rtype == "font" or _is_decorative_url(url):
            _log_warn("REQUEST_FAILED", log_id, type=rtype, url=url, reason=reason)
        else:
            _log_error("REQUEST_FAILED", log_id, type=rtype, url=url, reason=reason)

    failed_urls = {u for u, _, _ in failed_requests}
    for url, status, rtype in http_errors:
        if url in failed_urls:
            continue
        if rtype == "font" or _is_decorative_url(url):
            _log_warn("HTTP_ERROR", log_id, type=rtype, status=status, url=url)
        else:
            _log_error("HTTP_ERROR", log_id, type=rtype, status=status, url=url)

    for err in page_errors:
        _log_error("PAGE_ERROR", log_id, msg=err)

    for err in console_errors:
        _log_warn("CONSOLE_ERROR", log_id, msg=err)

    try:
        if os.path.exists(out_path):
            size = os.path.getsize(out_path)
            if size < 1024:
                _log_error("EMPTY_OUTPUT", log_id, size=size, path=out_path)
        else:
            _log_error("EMPTY_OUTPUT", log_id, msg="file not created", path=out_path)
    except Exception:
        pass

    # -------- 宽高比检测 --------
    if tall_ratio_threshold > 0:
        try:
            if os.path.exists(out_path):
                with Image.open(out_path) as img:
                    w, h = img.size
                if w > 0 and h / w > tall_ratio_threshold:
                    _log_warn("TALL_PAGE", log_id,
                              ratio=round(h / w, 1), width=w, height=h)
        except Exception:
            pass

    return out_path


# =========================================================
# Worker 主循环
# =========================================================
def _worker_main(args):
    worker_id, tasks, cfg, progress_queue = args

    base_max_wait = cfg["base_max_wait"]
    base_idle_timeout = cfg["base_idle_timeout"]
    stable_frames = cfg["stable_frames"]
    page_total_timeout_ms = cfg["page_total_timeout_ms"]
    recycle_every = cfg["recycle_every"]
    enable_growth_check = cfg.get("enable_growth_check", True)
    growth_min_window_ms = cfg.get("growth_min_window_ms", 3000)
    growth_threshold_px = cfg.get("growth_threshold_px", 2000)
    enable_overlap_check = cfg.get("enable_overlap_check", True)
    overlap_ratio_threshold = cfg.get("overlap_ratio_threshold", 0.15)
    tall_ratio_threshold = cfg.get("tall_ratio_threshold", 3.0)

    factor = _system_load_factor()
    max_wait_ms = int(base_max_wait * factor)
    idle_timeout_ms = int(base_idle_timeout * factor)

    body_cache = _BodyCache()

    results = []
    pw = None
    browser = None
    context = None
    rendered = 0

    def _launch():
        nonlocal pw, browser, context
        if pw is None:
            pw = sync_playwright().start()
        browser = pw.chromium.launch(
            args=["--disable-web-security", "--ignore-certificate-errors",
                  "--disable-dev-shm-usage"]
        )
        context = browser.new_context(
            ignore_https_errors=True,
            viewport={"width": 1280, "height": 800},
        )
        context.add_init_script(_INIT_SCRIPT)

    def _shutdown():
        nonlocal browser, context
        try:
            if context is not None:
                context.close()
        except Exception:
            pass
        try:
            if browser is not None:
                browser.close()
        except Exception:
            pass
        browser = None
        context = None

    try:
        _launch()
        for task in tasks:
            out_path = _render_one(
                task, context, body_cache,
                max_wait_ms, idle_timeout_ms,
                stable_frames, page_total_timeout_ms,
                factor,
                enable_growth_check=enable_growth_check,
                growth_min_window_ms=growth_min_window_ms,
                growth_threshold_px=growth_threshold_px,
                enable_overlap_check=enable_overlap_check,
                overlap_ratio_threshold=overlap_ratio_threshold,
                tall_ratio_threshold=tall_ratio_threshold,
            )
            results.append(out_path)
            rendered += 1

            if progress_queue is not None:
                try:
                    progress_queue.put_nowait(1)
                except Exception:
                    pass

            if recycle_every > 0 and rendered % recycle_every == 0:
                _shutdown()
                _launch()
    finally:
        _shutdown()
        try:
            if pw is not None:
                pw.stop()
        except Exception:
            pass

    return results


# =========================================================
# 主入口
# =========================================================
def _output_stem(json_path: Path, json_dir: Path) -> str:
    """
    从输入文件路径派生输出子目录名，处理子目录文件名冲突。
    part_01/output.jsonl → part_01_output
    data.json            → data
    """
    try:
        rel = json_path.relative_to(json_dir)
    except ValueError:
        rel = Path(json_path.name)
    parts = list(rel.with_suffix("").parts)
    return "_".join(parts)


def process_file(json_path, images_root, workers, cache_dir, cfg,
                 html_field="html", id_field="id", json_dir=None):
    json_path = Path(json_path)
    base_dir = Path(json_dir) if json_dir else json_path.parent
    stem = _output_stem(json_path, base_dir)
    out_dir = os.path.join(images_root, stem)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n[LOAD] {json_path}")

    try:
        data_array = _load_records(json_path)
    except Exception as e:
        print(f"[ERROR] Failed to load {json_path}: {e}")
        return

    print(f"[TASKS] Loaded {len(data_array)} items from {json_path}")

    print(f"[SCAN] Listing existing outputs in {out_dir} ...")
    t0 = time.time()
    try:
        existing = set()
        with os.scandir(out_dir) as it:
            for entry in it:
                name = entry.name
                if name.endswith(".png"):
                    existing.add(name[:-4])
        print(f"[SCAN] Found {len(existing)} existing outputs in {time.time()-t0:.1f}s")
    except FileNotFoundError:
        existing = set()
        print(f"[SCAN] Output dir not found, assuming empty")

    preprocess_html = cfg.get("preprocess_html", True)
    all_tasks = []
    for data in data_array:
        raw_id = data.get(id_field, data.get("id", ""))
        item_id = _sanitize_id(raw_id)
        if item_id in existing:
            continue
        html = data.get(html_field, "")
        if preprocess_html and html:
            html = _preprocess_html_string(html)
        all_tasks.append({
            "json_name": json_path.name,
            "id": item_id,
            "log_id": f"{stem}:{item_id}",
            "html": html,
            "output_dir": out_dir,
            "cache_dir": cache_dir,
        })

    n_skipped = len(data_array) - len(all_tasks)
    if n_skipped > 0:
        print(
            f"[RESUME] {n_skipped}/{len(data_array)} items skipped "
            f"(PNG already exists). Their WARN/ERROR from previous runs "
            f"are NOT captured in this run's log.",
            flush=True,
        )
    if not all_tasks:
        print(f"[DONE] {json_path} (all {n_skipped} items already rendered)\n")
        return

    n = min(workers, len(all_tasks))
    chunks = [[] for _ in range(n)]
    for i, t in enumerate(all_tasks):
        chunks[i % n].append(t)

    total = len(all_tasks)
    pbar = tqdm(total=total, desc=f"Rendering {json_path.name}",
                file=sys.stdout, dynamic_ncols=True)

    manager = Manager()
    progress_queue = manager.Queue()
    stop_flag = threading.Event()

    def _progress_consumer():
        while not stop_flag.is_set():
            try:
                progress_queue.get(timeout=0.5)
                pbar.update(1)
            except Exception:
                continue

    consumer = threading.Thread(target=_progress_consumer, daemon=True)
    consumer.start()

    worker_args = [(i, chunks[i], cfg, progress_queue) for i in range(n)]

    with Pool(n) as pool:
        try:
            for result_list in pool.imap_unordered(_worker_main, worker_args):
                pass
        except Exception as e:
            _log_error("RENDER_ERROR", f"{json_path.stem}:pool", msg=str(e) or repr(e))
            try:
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
            except Exception:
                pass
            pool.terminate()

    time.sleep(1.0)
    stop_flag.set()
    consumer.join(timeout=2.0)
    pbar.close()
    print(f"[DONE] {json_path}\n")


# =========================================================
# CLI
# =========================================================
def main(argv=None):
    parser = argparse.ArgumentParser(description="Render HTML from JSON array using Playwright")
    parser.add_argument("--json_dir", type=str, required=True)
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--workers", type=int, default=max(cpu_count() - 1, 1))
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--cache_concurrency", type=int, default=16)
    parser.add_argument("--skip_cache_download", action="store_true")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--base_max_wait", type=int, default=6000)
    parser.add_argument("--base_idle_timeout", type=int, default=1500)
    parser.add_argument("--stable_frames", type=int, default=3)
    parser.add_argument("--page_total_timeout", type=int, default=60000)
    parser.add_argument("--recycle_every", type=int, default=200)

    # 运行时拉长检测
    parser.add_argument("--no_growth_check", action="store_true",
                        help="禁用运行时页面无限拉长检测")
    parser.add_argument("--growth_min_window_ms", type=int, default=1000,
                        help="拉长判定的最小窗口时长(毫秒)")
    parser.add_argument("--growth_threshold_px", type=int, default=500,
                        help="判定窗口内至少累计增长多少像素才算异常")

    # 运行时文字叠压检测
    parser.add_argument("--no_overlap_check", action="store_true",
                        help="禁用运行时文字叠压检测")
    parser.add_argument("--overlap_ratio_threshold", type=float, default=0.20,
                        help="叠压判定阈值:重叠面积/较小元素面积超过此比例才报警(默认 0.20)")

    # HTML 预处理
    parser.add_argument("--no_preprocess", action="store_true",
                        help="禁用 HTML 预处理（保留 Google Fonts 等外部字体引用）")

    # 图像宽高比检测
    parser.add_argument("--tall_ratio_threshold", type=float, default=3.0,
                        help="高宽比超过此值时记录 WARN/TALL_PAGE（默认 3.0，0 禁用）")

    # 数据字段映射
    parser.add_argument("--html_field", default="html",
                        help="JSON 记录中包含 HTML 内容的字段名（默认 html）")
    parser.add_argument("--id_field", default="id",
                        help="JSON 记录中用作截图文件名的 ID 字段名（默认 id）")
    args = parser.parse_args(argv)

    json_dir = Path(args.json_dir)
    images_dir = args.images_dir

    if args.no_cache:
        cache_dir = None
    else:
        cache_dir = args.cache_dir or os.path.join(
            os.path.dirname(os.path.abspath(images_dir)), "_cdn_cache"
        )
        os.makedirs(cache_dir, exist_ok=True)
        print(f"[CDN] Cache directory: {cache_dir}")

    if cache_dir and not args.skip_cache_download:
        prepare_cdn_cache(args.files, json_dir, cache_dir,
                          concurrency=args.cache_concurrency,
                          html_field=args.html_field,
                          preprocess=not args.no_preprocess)

    cfg = {
        "base_max_wait": args.base_max_wait,
        "base_idle_timeout": args.base_idle_timeout,
        "stable_frames": args.stable_frames,
        "page_total_timeout_ms": args.page_total_timeout,
        "recycle_every": args.recycle_every,
        "enable_growth_check": not args.no_growth_check,
        "growth_min_window_ms": args.growth_min_window_ms,
        "growth_threshold_px": args.growth_threshold_px,
        "enable_overlap_check": not args.no_overlap_check,
        "overlap_ratio_threshold": args.overlap_ratio_threshold,
        "preprocess_html": not args.no_preprocess,
        "tall_ratio_threshold": args.tall_ratio_threshold,
    }

    for filename in args.files:
        process_file(
            json_dir / filename, images_dir, args.workers, cache_dir, cfg,
            html_field=args.html_field,
            id_field=args.id_field,
            json_dir=json_dir,
        )


if __name__ == "__main__":
    main()
