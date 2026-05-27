"""将 html_outline_json 渲染为紧凑文本，供 VLM prompt 使用。"""

from __future__ import annotations

import re


def _none_str(val: object) -> str:
    return "None extracted" if not val else str(val)


def _list_lines(items: list, fmt) -> str:
    if not items:
        return "None extracted"
    return "\n".join(fmt(x) for x in items)


def render_outline_text(outline: dict) -> str:
    meta = outline.get("meta", {})
    structure = outline.get("structure", {})
    text = outline.get("text", {})
    ls = outline.get("layout_style", {})
    assets = outline.get("assets", {})

    title = meta.get("title") or "None extracted"

    major = structure.get("major_structure", [])
    major_str = _list_lines(major, lambda x: f"- {x}")

    headings = text.get("headings", [])
    headings_str = _list_lines(headings, lambda h: f'{h["tag"]}: {h["text"]}')

    nav_links = text.get("navigation_links", [])
    nav_str = _list_lines(nav_links, lambda n: f'- {n["text"]}')

    buttons = text.get("buttons", [])
    btns_str = _list_lines(buttons, lambda b: f'[{b["type"]}] {b["text"]}')

    forms = text.get("forms", [])
    def _fmt_form(f: dict) -> str:
        fields = ", ".join(
            f'{fd["tag"]}[{fd.get("type","")}]' + (f'="{fd["placeholder"]}"' if fd.get("placeholder") else "")
            for fd in f.get("fields", [])[:6]
        )
        return f'form(action={f.get("action","")!r}): {fields or "no fields"}'
    forms_str = _list_lines(forms, _fmt_form)

    sidebar_titles = text.get("sidebar_or_card_titles", [])
    sidebar_str = _list_lines(sidebar_titles, lambda s: f"- {s}")

    footer_links = text.get("footer_links", [])
    footer_str = _list_lines(footer_links, lambda n: f'- {n["text"]}')

    lists = text.get("lists", [])
    def _fmt_list(l: dict) -> str:
        items = ", ".join(l.get("items", [])[:4])
        return f'{l["type"]}: {items}'
    lists_str = _list_lines(lists, _fmt_list)

    tables = text.get("tables", [])
    def _fmt_table(t: dict) -> str:
        hdrs = ", ".join(t.get("headers", [])[:6])
        return f'table({t.get("row_count", 0)} rows), headers: {hdrs or "none"}'
    tables_str = _list_lines(tables, _fmt_table)

    # Layout and style
    layout_parts: list[str] = []
    for hint in ls.get("layout_hints", []):
        layout_parts.append(hint)
    for hint in ls.get("style_hints", []):
        layout_parts.append(hint)
    for c in ls.get("colors", []):
        # Skip background colors for placeholder elements — they are training-data
        # artifacts (not actual page colors) that the model would copy verbatim.
        if "placeholder" in c.get("selector", "").lower():
            continue
        hex_or_val = c.get("hex") or c.get("value", "")
        layout_parts.append(f'{c["selector"]}: {c["property"]} {hex_or_val}')
    for font in ls.get("fonts", [])[:3]:
        layout_parts.append(f"font: {font}")
    layout_str = _list_lines(layout_parts, lambda x: f"- {x}")

    # Assets
    asset_parts: list[str] = []
    for img in assets.get("images", [])[:5]:
        label = img.get("alt") or img.get("src", "")[:40]
        bbox = img.get("bbox", {})
        bstr = f' ({bbox.get("w",0)}×{bbox.get("h",0)})' if bbox else ""
        asset_parts.append(f"img: {label}{bstr}")
    svgs = assets.get("svg_elements", [])
    if svgs:
        asset_parts.append(f"{len(svgs)} SVG element(s)")
    # Deduplicate visual_blocks by description, then prioritize placeholder elements
    # so they are never cut off by the limit even when other blocks are numerous.
    _vb_seen: set[str] = set()
    _vb_placeholders: list[dict] = []
    _vb_others: list[dict] = []
    for vb in assets.get("visual_blocks", []):
        key = vb.get("description", "")
        if key in _vb_seen:
            continue
        _vb_seen.add(key)
        if "placeholder" in vb.get("selector", "").lower():
            _vb_placeholders.append(vb)
        else:
            _vb_others.append(vb)
    for vb in (_vb_placeholders + _vb_others)[:8]:
        sel = vb.get("selector", "")[:30]
        desc = vb.get("description", "")
        if "placeholder" in sel.lower():
            # Strip background color — placeholder fill colors are training-data
            # artifacts; expose only size/shape so the model uses layout properties.
            desc = re.sub(r"\bbg #[0-9a-fA-F]{3,6}\b\s*", "", desc).strip()
            asset_parts.append(
                f'media region: {desc} ({sel})'
                f' [placeholder — describe by visual role, size, and aspect ratio only]'
            )
        else:
            asset_parts.append(f'visual block: {desc} ({sel})')
    css_bgs = assets.get("css_backgrounds", [])
    if css_bgs:
        asset_parts.append(f"{len(css_bgs)} CSS background(s)")
    pseudo = assets.get("pseudo_visual_blocks", [])
    if pseudo:
        asset_parts.append(f"{len(pseudo)} pseudo-element visual block(s)")
    assets_str = _list_lines(asset_parts, lambda x: f"- {x}")

    warnings = meta.get("warnings", [])
    warnings_str = _list_lines(warnings, lambda w: f"- {w}")

    lines = [
        "HTML outline",
        "",
        "Title:",
        title,
        "",
        "Major structure:",
        major_str,
        "",
        "Structural text:",
        "Headings:",
        headings_str,
        "Navigation links:",
        nav_str,
        "Buttons:",
        btns_str,
        "Forms:",
        forms_str,
        "Sidebar/card titles:",
        sidebar_str,
        "Footer links:",
        footer_str,
        "Lists:",
        lists_str,
        "Tables:",
        tables_str,
        "",
        "Layout and style:",
        layout_str,
        "",
        "Assets and visual blocks:",
        assets_str,
        "",
        "Extraction warnings:",
        warnings_str,
    ]
    return "\n".join(lines)
