"""VLM 调用封装：基于现有 utils/api_client，构造 image-to-HTML CoT 请求。"""

from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup

from utils.api_client import call_chat_completion, image_text_content
from html_rewrite_cot.config import VLMConfig

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert frontend engineer and visual layout annotator.

Given a webpage screenshot, a rule-extracted HTML outline, and the original target HTML, \
write the reasoning text that should appear before the final HTML code in an \
image-to-HTML training answer.

Use a structured visual implementation analysis style. Explain the page layout, visible \
regions, important components, colors, visual style, assets, and CSS implementation choices \
needed to recreate the page as a self-contained HTML file with embedded CSS.

Visual media regions: For any region in the screenshot that carries image or media content \
— whether it appears as a real photograph, an illustration, a logo, a banner, or a \
placeholder box — describe it using layout-relevant properties only: its visual role in \
the page hierarchy (e.g. "brand logo", "hero banner", "sidebar ad slot"), its approximate \
size relative to surrounding elements, and its aspect ratio. These three attributes \
describe the region equally well for a real screenshot or a placeholder.

Strictly forbidden — do not write any of the following:
- Fill/background color of a placeholder (e.g. "light gray background", "gray fill", \
"#e9ecef background")
- Border style of a placeholder ("dashed border", "dotted outline")
- Text labels found inside placeholder elements (e.g. "IpLiveCams — Logo", \
"Sidebar area (ads / widgets removed)", "Image Placeholder", filename labels)
- Phrases that reveal training-data construction ("placeholder box", "styled placeholder \
div", "ads removed", "widgets removed", "placeholder text")

In the implementation plan, choose and explain the appropriate technique for each media \
region based on the actual HTML structure — the correct choice may be an <img> element, \
a CSS background-image container, a video embed, an SVG, or a styled placeholder div.

Do not output HTML or CSS code.\
"""

_USER_TEMPLATE = """\
You are given:
1. A webpage screenshot.
2. A rule-extracted HTML outline.
3. The original target HTML.

Use the screenshot for visual appearance.
Use the HTML outline for structure, visible text, layout, style, assets, and visual blocks.
Use the original target HTML for exact implemented elements.

Write only the reasoning text. A post-processing script will append the final HTML code \
block after your reasoning.

Required reasoning structure:
1. Start with one brief task framing sentence, such as:
   "The user wants me to convert the webpage screenshot into a single self-contained HTML \
file with embedded CSS. Let me analyze the image carefully."

2. Include a "Layout Analysis:" section.
   Describe the overall page layout from top to bottom, including major regions and \
column/grid relationships.

3. Add relevant region-specific sections according to the actual screenshot.
   Use descriptive headings such as Header Section, Navigation Section, Hero Section, \
Main Content Area, Sidebar, Summary/Card Section, Forms and Tables, Footer Section, or \
Assets and Visual Blocks when they fit the page.
   Do not force a section that is not present.

   For each image or media region visible in the screenshot — including hero banners, \
logos, product thumbnails, avatars, inline illustrations, and decorative images — describe:
   (a) its visual role in the layout (e.g., "full-width hero banner at the top of the \
main column", "square 80px avatar left of the heading", "4:3 thumbnail in a card grid"),
   (b) its approximate size relative to the containing section, and
   (c) its aspect ratio, since this determines the CSS sizing approach.
   Describe what the region IS and WHERE it sits — not how the placeholder box looks.
   The outline marks certain elements as "[placeholder — describe by visual role, size, \
and aspect ratio only]". Treat these exactly like real images: use only (a)(b)(c) above.

   Good example: "A 240×40px brand logo at roughly 6:1 aspect ratio, positioned at the \
top of the sidebar."
   Bad example: "A light gray (#e9ecef) rectangle with a dashed border labeled \
'IpLiveCams — Logo'." — Do not write descriptions like this.

4. Include a "Colors Observed:" section.
   Summarize dominant colors and approximate color roles. Use approximate hex values only \
when helpful.

5. End with a "Structure and Implementation Plan:" section.
   Describe the HTML/CSS implementation plan. Mention embedded CSS, flexbox/grid, \
multi-column layout, cards, forms, tables, image placeholders, CSS-drawn blocks, and \
centered containers when relevant.
   If a layout relationship is ambiguous, briefly state the final implementation choice.
   For each visual media region identified in your analysis, state the chosen \
implementation technique and how its dimensions or aspect ratio will be maintained.

6. After the "Structure and Implementation Plan:" section, close with a single natural \
transition sentence that bridges your analysis to the HTML output. Examples (vary the wording):
   "I'll now write the complete self-contained HTML file with embedded CSS to match this layout."
   "Now I'll implement the full page as a self-contained HTML file with embedded CSS."
   Use your own natural phrasing — do not copy an example verbatim.

Content requirements:
- Focus on concrete visual and coding-relevant details.
- Mention important visible components and text groups without copying long paragraph text.
- Do not include HTML or CSS code.
- Do not include markdown code fences.

HTML outline:
<outline>
{outline_text}
</outline>

Original target HTML:
<target_html>
{raw_html}
</target_html>\
"""


# ── HTML 预处理 ───────────────────────────────────────────────────────────────

_PLACEHOLDER_CLASS_RE = re.compile(r"\bplaceholder\b", re.IGNORECASE)

# Strips bg/border from CSS rules like ".logo-placeholder { background: ...; border: ... }"
_PLACEHOLDER_CSS_RULE_RE = re.compile(
    r"(\.[a-zA-Z0-9_-]*placeholder[a-zA-Z0-9_-]*\s*\{)([^}]+)(\})",
    re.IGNORECASE,
)
_PLACEHOLDER_PROP_RE = re.compile(
    r"\b(?:background(?:-color)?|border(?:-color|-style|-width)?)\s*:[^;]+;",
    re.IGNORECASE,
)


def _clean_placeholder_text(html: str) -> str:
    """Remove placeholder-specific content from raw_html before sending to VLM.

    Two operations:
    1. Clear direct text content from elements whose class contains 'placeholder'
       (prevents model from reading labels like "Sidebar area (ads / widgets removed)").
    2. Strip background-color and border CSS properties from placeholder CSS rules
       (prevents model from reading exact placeholder fill/border colors from <style>).
    Nested HTML child elements and non-placeholder CSS rules are preserved.
    """
    # Step 1: remove CSS background/border from placeholder class rules
    def _strip_placeholder_css_props(m: re.Match) -> str:
        body = _PLACEHOLDER_PROP_RE.sub("", m.group(2))
        return m.group(1) + body + m.group(3)

    html = _PLACEHOLDER_CSS_RULE_RE.sub(_strip_placeholder_css_props, html)

    # Step 2: remove text content from placeholder elements
    soup = BeautifulSoup(html, "html.parser")
    for el in soup.find_all(class_=_PLACEHOLDER_CLASS_RE):
        for text_node in el.find_all(string=True, recursive=False):
            text_node.replace_with("")
    return str(soup)


# ── 调用接口 ──────────────────────────────────────────────────────────────────


def call_vlm(
    image_path: str,
    outline_text: str,
    raw_html: str,
    image_format: str,
    config: VLMConfig,
    call_log_path: str | None = None,
) -> str:
    """
    调用 VLM，返回原始 reasoning 文本。
    user content 由 utils.api_client.image_text_content 构造（base64 编码由其负责）。
    generation_params 直接透传，与 html_rewrite 流水线保持一致。
    调用失败时抛出异常，由调用方捕获并记录。
    """
    cleaned_html = _clean_placeholder_text(raw_html)
    user_text = _USER_TEMPLATE.format(outline_text=outline_text, raw_html=cleaned_html)
    user_content = image_text_content(
        image_path=image_path,
        text=user_text,
        image_first=True,
        image_format=image_format,
    )

    log_path = Path(call_log_path) if call_log_path else None

    resp = call_chat_completion(
        url=config.url,
        api_key=config.api_key,
        model=config.model,
        user_content=user_content,
        system=SYSTEM_PROMPT,
        timeout=config.timeout,
        max_retries=config.max_retries,
        ssl_verify=config.ssl_verify,
        log_user=config.log_user,
        result_log_path=log_path,
        extra_params=config.generation_params or {},
    )
    return resp["choices"][0]["message"]["content"]
