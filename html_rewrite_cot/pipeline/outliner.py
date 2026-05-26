"""HTML outline 提取：静态解析（BeautifulSoup）+ 渲染提取（Playwright）。"""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Any

from bs4 import BeautifulSoup, NavigableString, Tag
from loguru import logger

if TYPE_CHECKING:
    from html_rewrite_cot.config import RenderConfig

# ── 结构关键词 ─────────────────────────────────────────────────────────────────

_KW_LAYOUT = frozenset("container wrapper inner outer layout grid row rows column columns col flex stack section block region area content main article body page".split())
_KW_HEADER = frozenset("header head top topbar masthead brand branding logo nav navbar navigation menu menubar primary-nav secondary-nav main-nav subnav breadcrumb breadcrumbs tabs tabbar".split())
_KW_HERO = frozenset("hero banner jumbotron cover splash headline intro landing promo promotion showcase lead".split())
_KW_SIDEBAR = frozenset("sidebar side-bar aside rail right-rail left-rail drawer widget panel summary info-box infobox help-box related recommend filter filters".split())
_KW_CARD = frozenset("card cards box tile panel module widget feature features item list-item callout notice alert profile stat stats pricing plan testimonial review".split())
_KW_FORM = frozenset("form field input textarea select dropdown search searchbox query contact newsletter subscribe signup sign-up login cta action button btn submit request help call".split())
_KW_MEDIA = frozenset("image img photo picture media thumbnail thumb avatar icon icons logo brand placeholder figure gallery carousel slider video map background bg".split())
_KW_FOOTER = frozenset("footer foot bottom copyright legal privacy terms policy disclaimer social share follow sponsor partners credits attribution".split())
_KW_TABLE = frozenset("table thead tbody row cell data chart graph stats metrics comparison schedule calendar".split())

_ALL_KW = _KW_LAYOUT | _KW_HEADER | _KW_HERO | _KW_SIDEBAR | _KW_CARD | _KW_FORM | _KW_MEDIA | _KW_FOOTER | _KW_TABLE

_ALWAYS_KEEP = frozenset(
    "header nav main article aside section footer "
    "form table thead tbody tr th td "
    "ul ol li "
    "h1 h2 h3 h4 h5 h6 "
    "button input textarea select label "
    "img picture source svg "
    "a".split()
)
_SKIP_TAGS = frozenset("script noscript template meta link style head".split())


# ── 静态解析辅助 ───────────────────────────────────────────────────────────────


def _tag_kws(tag: Tag) -> frozenset[str]:
    tokens: list[str] = []
    cls = tag.get("class", [])
    if isinstance(cls, list):
        for c in cls:
            tokens.extend(re.split(r"[-_\s]+", c.lower()))
    tag_id = tag.get("id", "")
    if tag_id:
        tokens.extend(re.split(r"[-_\s]+", tag_id.lower()))
    return frozenset(tokens)


def _is_hidden(tag: Tag) -> bool:
    if tag.has_attr("hidden"):
        return True
    if tag.get("aria-hidden") == "true":
        return True
    style = tag.get("style", "").replace(" ", "").lower()
    return "display:none" in style or "visibility:hidden" in style


def _short_text(tag: Tag, max_len: int = 80) -> str | None:
    text = tag.get_text(separator=" ", strip=True)
    if not text:
        return None
    return text if len(text) <= max_len else text[:max_len] + "…"


def _role_hint(tag: Tag) -> str:
    kws = _tag_kws(tag)
    name = tag.name
    if name == "header" or kws & _KW_HEADER:
        if kws & {"nav", "navbar", "navigation", "menu"}:
            return "navigation"
        return "header"
    if name == "nav" or kws & {"nav", "navbar", "navigation", "menu", "menubar"}:
        return "navigation"
    if name in ("main", "article") or kws & {"main", "article", "content"}:
        return "main_content"
    if name == "aside" or kws & _KW_SIDEBAR:
        return "sidebar"
    if name == "footer" or kws & _KW_FOOTER:
        return "footer"
    if name == "form" or kws & {"form", "login", "search", "subscribe"}:
        return "form"
    if kws & _KW_HERO:
        return "hero"
    if kws & _KW_CARD:
        return "card"
    if kws & _KW_MEDIA:
        return "media"
    if kws & _KW_LAYOUT:
        return "container"
    return "div"


def _selector(tag: Tag) -> str:
    cls = tag.get("class", [])
    cls_str = "." + ".".join(cls) if cls else ""
    id_str = "#" + tag.get("id", "") if tag.get("id") else ""
    return f"{tag.name}{id_str}{cls_str}"[:100]


def _keep_div(tag: Tag) -> bool:
    kws = _tag_kws(tag)
    if kws & _ALL_KW:
        return True
    direct_text = "".join(
        str(c) for c in tag.children if isinstance(c, NavigableString)
    ).strip()
    if direct_text and len(direct_text) <= 80:
        return True
    structural = {"h1", "h2", "h3", "h4", "h5", "h6", "nav", "form", "button", "img", "svg", "ul", "ol", "table"}
    return any(tag.find(t) for t in structural)


def _build_dom_outline(tag: Tag, depth: int = 0, max_depth: int = 8) -> list[dict]:
    if depth > max_depth:
        return []
    result: list[dict] = []
    for child in tag.children:
        if not isinstance(child, Tag):
            continue
        if child.name in _SKIP_TAGS or _is_hidden(child):
            continue
        if child.name in _ALWAYS_KEEP:
            keep = True
        elif child.name in ("div", "span", "section"):
            keep = _keep_div(child)
        else:
            keep = False

        if keep:
            node: dict[str, Any] = {
                "tag": child.name,
                "selector": _selector(child),
                "role_hint": _role_hint(child),
                "text": _short_text(child),
                "children": _build_dom_outline(child, depth + 1, max_depth),
            }
            result.append(node)
        else:
            result.extend(_build_dom_outline(child, depth, max_depth))
    return result


def _extract_title(soup: BeautifulSoup) -> str | None:
    t = soup.find("title")
    if t and t.get_text(strip=True):
        return t.get_text(strip=True)
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        return og["content"]
    mt = soup.find("meta", attrs={"name": "title"})
    if mt and mt.get("content"):
        return mt["content"]
    return None


def _extract_structural_text(soup: BeautifulSoup) -> dict:
    def t(tag: Tag) -> str:
        return tag.get_text(separator=" ", strip=True)

    headings = [
        {"tag": h.name, "text": t(h)}
        for h in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
        if not _is_hidden(h) and t(h)
    ]

    nav_containers = list(soup.find_all(["nav", "header"]))
    nav_class_containers = [
        el for el in soup.find_all(["div", "ul", "section"])
        if _tag_kws(el) & {"nav", "menu", "navbar", "topbar", "main-nav", "navigation"}
    ]
    seen_nav: set[str] = set()
    nav_links: list[dict] = []
    for container in nav_containers + nav_class_containers:
        for a in container.find_all("a", recursive=True):
            if not _is_hidden(a) and t(a) and t(a) not in seen_nav:
                seen_nav.add(t(a))
                nav_links.append({"text": t(a), "href": a.get("href", "")})
    nav_links = nav_links[:20]

    buttons: list[dict] = []
    for btn in soup.find_all(["button", "input"]):
        if _is_hidden(btn):
            continue
        if btn.name == "button" and t(btn):
            buttons.append({"type": "button", "text": t(btn)})
        elif btn.name == "input" and btn.get("type") in ("button", "submit", "reset"):
            val = btn.get("value") or btn.get("placeholder", "")
            if val:
                buttons.append({"type": btn.get("type"), "text": val})
    for a in soup.find_all("a"):
        if not _is_hidden(a) and (_tag_kws(a) & {"btn", "button", "cta", "action"}) and t(a):
            buttons.append({"type": "link-button", "text": t(a)})
    buttons = buttons[:15]

    forms: list[dict] = []
    for form in soup.find_all("form"):
        if _is_hidden(form):
            continue
        fields = [
            {
                "tag": inp.name,
                "type": inp.get("type", ""),
                "name": inp.get("name", ""),
                "placeholder": inp.get("placeholder", ""),
            }
            for inp in form.find_all(["input", "textarea", "select"])
            if not _is_hidden(inp)
        ][:8]
        forms.append({"action": form.get("action", ""), "fields": fields})

    sidebar_containers = [
        el for el in soup.find_all(True)
        if _tag_kws(el) & (_KW_SIDEBAR | _KW_CARD)
    ]
    sidebar_titles: list[str] = []
    seen_st: set[str] = set()
    for container in sidebar_containers:
        if _is_hidden(container):
            continue
        for h in container.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "strong"]):
            txt = t(h)
            if txt and txt not in seen_st:
                seen_st.add(txt)
                sidebar_titles.append(txt)
    sidebar_titles = sidebar_titles[:10]

    footer_links: list[dict] = []
    seen_ft: set[str] = set()
    for container in list(soup.find_all("footer")) + [
        el for el in soup.find_all(True) if _tag_kws(el) & _KW_FOOTER
    ]:
        for a in container.find_all("a"):
            if not _is_hidden(a) and t(a) and t(a) not in seen_ft:
                seen_ft.add(t(a))
                footer_links.append({"text": t(a), "href": a.get("href", "")})
    footer_links = footer_links[:20]

    lists: list[dict] = []
    for lst in soup.find_all(["ul", "ol"])[:6]:
        if _is_hidden(lst):
            continue
        items = [t(li) for li in lst.find_all("li", recursive=False) if t(li)][:5]
        if items:
            lists.append({"type": lst.name, "items": items})

    tables: list[dict] = []
    for tbl in soup.find_all("table")[:3]:
        if _is_hidden(tbl):
            continue
        headers = [t(th) for th in tbl.find_all("th")[:8] if t(th)]
        rows = tbl.find_all("tr")
        sample_rows = [
            [t(td) for td in row.find_all("td") if t(td)][:6]
            for row in rows[1:3]
            if row.find("td")
        ]
        tables.append({"headers": headers, "row_count": len(rows), "sample_rows": sample_rows})

    return {
        "headings": headings,
        "navigation_links": nav_links,
        "buttons": buttons,
        "forms": forms,
        "sidebar_or_card_titles": sidebar_titles,
        "footer_links": footer_links,
        "lists": lists,
        "tables": tables,
    }


def _build_major_structure(soup: BeautifulSoup) -> list[str]:
    body = soup.find("body") or soup
    sections: list[str] = []

    def _sub_desc(container: Tag) -> list[str]:
        sub = []
        if container.find(["img", "picture", "svg"]):
            sub.append("image/visual block")
        if container.find("nav") or any(
            _tag_kws(el) & {"nav", "menu", "navbar"} for el in container.find_all(True)
        ):
            sub.append("navigation links")
        if container.find(["h1", "h2"]):
            sub.append("headings")
        if container.find("form") or container.find("input"):
            sub.append("form/input")
        if container.find(["ul", "ol"]):
            sub.append("list")
        if container.find("table"):
            sub.append("table")
        if container.find("button"):
            sub.append("buttons")
        return sub[:4]

    def _classify(tag: Tag) -> str | None:
        kws = _tag_kws(tag)
        name = tag.name
        if name == "header" or (kws & _KW_HEADER and name in ("div", "section")):
            return "header"
        if name == "nav" or kws & {"nav", "navbar", "navigation"}:
            return "navigation bar"
        if name == "footer" or (kws & _KW_FOOTER and name in ("div", "section")):
            return "footer"
        if name in ("main", "article") or kws & {"main", "content", "article"}:
            return "main area"
        if name == "aside" or kws & _KW_SIDEBAR:
            return "sidebar"
        if kws & _KW_HERO:
            return "hero section"
        return None

    def _walk(candidates, depth: int = 0) -> None:
        if depth > 2:
            return
        for child in candidates:
            if not isinstance(child, Tag) or child.name in _SKIP_TAGS or _is_hidden(child):
                continue
            region = _classify(child)
            if region:
                sub = _sub_desc(child)
                desc = region
                if sub:
                    desc += " with " + ", ".join(sub)
                sections.append(desc)
            elif depth < 2:
                # recurse into generic wrapper (e.g. <div class="wrapper">)
                _walk(child.children, depth + 1)

    _walk(body.children)
    return sections


def _static_extract(html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    return {
        "title": _extract_title(soup),
        "major_structure": _build_major_structure(soup),
        "dom_outline": _build_dom_outline(soup),
        "structural_text": _extract_structural_text(soup),
    }


# ── Playwright 渲染提取 ────────────────────────────────────────────────────────

_PLAYWRIGHT_JS = """
() => {
    const SEMANTIC_SEL = 'body,header,nav,main,article,aside,section,footer,form,table,thead,tbody,tr,th,td,ul,ol,li,h1,h2,h3,h4,h5,h6,button,input,textarea,select,label,img,picture,svg,a';
    const STRUCT_RE = /container|wrapper|layout|grid|row|col|sidebar|hero|banner|card|panel|nav|header|footer|main|content|column|block|section/i;

    function rgb2hex(v) {
        const m = v && v.match(/rgba?\\((\\d+),\\s*(\\d+),\\s*(\\d+)/);
        return m ? '#' + [m[1],m[2],m[3]].map(n => (+n).toString(16).padStart(2,'0')).join('') : null;
    }
    function isClear(v) {
        return !v || v === 'transparent' || /rgba\\(0,\\s*0,\\s*0,\\s*0\\)/.test(v);
    }
    function getSel(el) {
        const id = el.id ? '#' + el.id.slice(0,30) : '';
        const cls = (el.className && typeof el.className === 'string')
            ? '.' + el.className.trim().split(/\\s+/).slice(0,3).join('.').slice(0,60) : '';
        return (el.tagName.toLowerCase() + id + cls).slice(0, 100);
    }
    function getInfo(el) {
        const s = getComputedStyle(el);
        const r = el.getBoundingClientRect();
        const txt = (el.textContent || '').trim();
        const hasBg = !isClear(s.backgroundColor);
        const hasBgImg = s.backgroundImage && s.backgroundImage !== 'none';
        const hasBorder = s.border && s.borderStyle !== 'none' && !s.border.startsWith('0px');
        const isVB = r.width*r.height > 400 && txt.length < 15 && (hasBg || hasBgImg || hasBorder);
        return {
            sel: getSel(el),
            tag: el.tagName.toLowerCase(),
            bbox: {x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height)},
            display: s.display,
            position: s.position,
            flexDir: s.flexDirection,
            gridCols: s.gridTemplateColumns !== 'none' ? s.gridTemplateColumns.slice(0,80) : null,
            bgColor: hasBg ? s.backgroundColor : null,
            bgHex: hasBg ? rgb2hex(s.backgroundColor) : null,
            color: s.color,
            colorHex: rgb2hex(s.color),
            fontFamily: s.fontFamily.split(',')[0].trim().replace(/["']/g,'').slice(0,40),
            fontSize: s.fontSize,
            fontWeight: s.fontWeight,
            border: hasBorder ? s.border.slice(0,60) : null,
            borderRadius: s.borderRadius !== '0px' ? s.borderRadius : null,
            bgImage: hasBgImg ? s.backgroundImage.slice(0,80) : null,
            isVB: isVB,
            shortText: txt.length < 60 ? txt : txt.slice(0,57) + '...',
        };
    }

    const seen = new WeakSet();
    const elements = [];
    document.querySelectorAll(SEMANTIC_SEL).forEach(el => {
        if (!seen.has(el)) { seen.add(el); elements.push(getInfo(el)); }
    });
    document.querySelectorAll('div,span').forEach(el => {
        if (!seen.has(el)) {
            const cls = (el.className && typeof el.className === 'string') ? el.className : '';
            if (STRUCT_RE.test(cls) || STRUCT_RE.test(el.id || '')) {
                seen.add(el); elements.push(getInfo(el));
            }
        }
    });

    // pseudo-elements (limit to structural tags)
    const pseudos = [];
    document.querySelectorAll('header,nav,section,div,a,button,li,footer,aside').forEach(el => {
        ['::before','::after'].forEach(ps => {
            const s = getComputedStyle(el, ps);
            if (s.content && s.content !== 'none' && s.content !== '""') {
                const r = el.getBoundingClientRect();
                if (r.width > 5 && r.height > 5 && !isClear(s.backgroundColor)) {
                    pseudos.push({
                        sel: getSel(el) + ps,
                        bgColor: s.backgroundColor,
                        bgHex: rgb2hex(s.backgroundColor),
                        content: s.content.slice(0, 30),
                    });
                }
            }
        });
    });

    return { elements, pseudos };
}
"""


def _rgb2hex(v: str) -> str | None:
    m = re.match(r"rgba?\((\d+),\s*(\d+),\s*(\d+)", v or "")
    if not m:
        return None
    return "#{:02x}{:02x}{:02x}".format(int(m.group(1)), int(m.group(2)), int(m.group(3)))


def _is_clear_color(v: str | None) -> bool:
    if not v:
        return True
    return v in ("transparent", "rgba(0, 0, 0, 0)") or re.search(r"rgba\(0,\s*0,\s*0,\s*0\)", v) is not None


def _parse_rendered(data: dict, viewport_w: int, viewport_h: int) -> dict:
    elements: list[dict] = data.get("elements", [])
    pseudos: list[dict] = data.get("pseudos", [])

    layout_hints: list[str] = []
    style_hints: list[str] = []
    colors: list[dict] = []
    fonts: set[str] = set()
    images: list[dict] = []
    svgs: list[dict] = []
    visual_blocks: list[dict] = []
    css_backgrounds: list[dict] = []
    seen_colors: set[str] = set()

    for el in elements:
        sel = el.get("sel", "")
        tag = el.get("tag", "")
        bbox = el.get("bbox", {})
        w = bbox.get("w", 0)
        h = bbox.get("h", 0)
        display = el.get("display", "")
        flex_dir = el.get("flexDir", "")
        grid_cols = el.get("gridCols")
        bg_hex = el.get("bgHex")
        color_hex = el.get("colorHex")
        font = el.get("fontFamily", "")

        # Layout hints
        if "flex" in display:
            if flex_dir in ("column", "column-reverse"):
                layout_hints.append(f"{sel}: vertical flex column")
            else:
                layout_hints.append(f"{sel}: horizontal flex row")
        if "grid" in display and grid_cols:
            col_count = len(grid_cols.split())
            layout_hints.append(f"{sel}: grid layout ({col_count} col-tracks)")

        # Style hints
        if w >= viewport_w * 0.85 and 0 < h < 120:
            style_hints.append(f"{sel}: full-width bar (likely nav/header)")
        br = el.get("borderRadius")
        border = el.get("border")
        bg_color = el.get("bgColor")
        if br and border and bg_color and w > 80 and h > 40:
            style_hints.append(f"{sel}: card-like box (border + rounded corners)")

        # Colors
        if bg_hex and bg_hex not in seen_colors:
            seen_colors.add(bg_hex)
            colors.append({"selector": sel, "property": "background-color", "value": el.get("bgColor", ""), "hex": bg_hex})
        if tag in ("h1", "h2", "h3", "h4", "nav", "header", "footer", "button") and color_hex and color_hex not in seen_colors:
            seen_colors.add(color_hex)
            colors.append({"selector": sel, "property": "color", "value": el.get("color", ""), "hex": color_hex})

        # Fonts
        if font and font not in ("", "inherit", "initial"):
            fonts.add(font)

        # Assets
        if tag in ("img", "picture"):
            images.append({"selector": sel, "alt": el.get("shortText", ""), "src": "", "bbox": bbox})
        elif tag == "svg":
            svgs.append({"selector": sel, "bbox": bbox})

        # CSS backgrounds
        bg_image = el.get("bgImage")
        if bg_image:
            css_backgrounds.append({"selector": sel, "value": bg_image})

        # Visual blocks
        if el.get("isVB"):
            desc_parts = []
            if bg_hex:
                desc_parts.append(f"bg {bg_hex}")
            if br:
                desc_parts.append("rounded")
            if w and h:
                desc_parts.append(f"{w}×{h}px")
            visual_blocks.append({
                "selector": sel,
                "bbox": bbox,
                "description": " ".join(desc_parts),
            })

    pseudo_visual_blocks = [
        {"selector": p["sel"], "bg_hex": p.get("bgHex"), "content": p.get("content", "")}
        for p in pseudos
    ]

    # Deduplicate layout hints (keep first 20)
    seen_lh: set[str] = set()
    dedup_layout: list[str] = []
    for h in layout_hints:
        if h not in seen_lh:
            seen_lh.add(h)
            dedup_layout.append(h)

    return {
        "layout_hints": dedup_layout[:20],
        "style_hints": list(dict.fromkeys(style_hints))[:10],
        "colors": colors[:20],
        "fonts": sorted(fonts)[:5],
        "images": images[:10],
        "svg_elements": svgs[:10],
        "visual_blocks": visual_blocks[:10],
        "css_backgrounds": css_backgrounds[:10],
        "pseudo_visual_blocks": pseudo_visual_blocks[:5],
    }


async def _playwright_extract(
    html: str,
    width: int,
    height: int,
    config: "RenderConfig",
    browser,
) -> tuple[dict, list[str], bool]:
    """
    使用共享 browser 实例创建新 page，渲染 HTML 并提取计算样式。
    返回 (rendered_dict, warnings, timed_out)。
    失败时返回空 dict 和 warnings，不抛出异常。
    """
    warnings: list[str] = []
    timed_out = False
    try:
        ctx = await browser.new_context(viewport={"width": width, "height": height})
        page = await ctx.new_page()

        if config.block_external:
            async def _block(route):
                url = route.request.url
                if url.startswith("http://") or url.startswith("https://"):
                    await route.abort()
                else:
                    await route.continue_()
            await page.route("**/*", _block)

        try:
            await page.set_content(html, wait_until="domcontentloaded", timeout=config.content_timeout_ms)
        except Exception as e:
            warnings.append(f"set_content timeout/error: {e}")
            timed_out = True
            await ctx.close()
            return {}, warnings, timed_out

        await asyncio.sleep(0.2)

        try:
            raw = await page.evaluate(_PLAYWRIGHT_JS, timeout=config.js_timeout_ms)
        except Exception as e:
            warnings.append(f"JS evaluate timeout/error: {e}")
            timed_out = True
            await ctx.close()
            return {}, warnings, timed_out

        await ctx.close()
        return raw, warnings, False

    except Exception as e:
        warnings.append(f"playwright fatal error: {e}")
        return {}, warnings, True


# ── 主入口（async）────────────────────────────────────────────────────────────


async def extract_outline(
    html: str,
    image_width: int,
    image_height: int,
    config: "RenderConfig",
    browser,
) -> dict:
    """
    提取完整的 html_outline_json。

    流程：
    1. BeautifulSoup 静态解析
    2. Playwright 渲染（失败时降级，记录 warning）
    3. 合并为 html_outline_json 结构
    """
    static = _static_extract(html)

    # 确定 viewport 尺寸
    if config.viewport_strategy == "match_image" and image_width > 0 and image_height > 0:
        vw, vh = image_width, image_height
        vp_source = "match_image"
    elif config.viewport_strategy == "user_config" and config.viewport_width and config.viewport_height:
        vw, vh = config.viewport_width, config.viewport_height
        vp_source = "user_config"
    else:
        vw, vh = config.fallback_width, config.fallback_height
        vp_source = "fallback"
        if config.viewport_strategy == "match_image":
            logger.warning(f"match_image: image dimensions 0x0, falling back to {vw}x{vh}")

    warnings: list[str] = []
    rendered_raw, pw_warnings, timed_out = await _playwright_extract(html, vw, vh, config, browser)
    warnings.extend(pw_warnings)
    if timed_out:
        warnings.insert(0, f"playwright extraction timed out for viewport {vw}x{vh}")

    rendered = _parse_rendered(rendered_raw, vw, vh) if rendered_raw else {
        "layout_hints": [], "style_hints": [], "colors": [], "fonts": [],
        "images": [], "svg_elements": [], "visual_blocks": [], "css_backgrounds": [],
        "pseudo_visual_blocks": [],
    }

    st = static["structural_text"]
    return {
        "meta": {
            "title": static["title"],
            "viewport": {"width": vw, "height": vh, "source": vp_source},
            "warnings": warnings,
        },
        "structure": {
            "major_structure": static["major_structure"],
            "dom_outline": static["dom_outline"],
        },
        "text": {
            "headings": st["headings"],
            "navigation_links": st["navigation_links"],
            "buttons": st["buttons"],
            "forms": st["forms"],
            "sidebar_or_card_titles": st["sidebar_or_card_titles"],
            "footer_links": st["footer_links"],
            "lists": st["lists"],
            "tables": st["tables"],
        },
        "layout_style": {
            "layout_hints": rendered["layout_hints"],
            "style_hints": rendered["style_hints"],
            "colors": rendered["colors"],
            "fonts": rendered["fonts"],
        },
        "assets": {
            "images": rendered["images"],
            "svg_elements": rendered["svg_elements"],
            "css_backgrounds": rendered["css_backgrounds"],
            "visual_blocks": rendered["visual_blocks"],
            "pseudo_visual_blocks": rendered["pseudo_visual_blocks"],
        },
    }
