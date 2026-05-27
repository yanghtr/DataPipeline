"""VLM 调用封装：基于现有 utils/api_client，构造 image-to-HTML CoT 请求。"""

from __future__ import annotations

from pathlib import Path

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

4. Include a "Colors Observed:" section.
   Summarize dominant colors and approximate color roles. Use approximate hex values only \
when helpful.

5. End with a "Structure and Implementation Plan:" section.
   Describe the HTML/CSS implementation plan. Mention embedded CSS, flexbox/grid, \
multi-column layout, cards, forms, tables, image placeholders, CSS-drawn blocks, and \
centered containers when relevant.
   If a layout relationship is ambiguous, briefly state the final implementation choice.

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
    user_text = _USER_TEMPLATE.format(outline_text=outline_text, raw_html=raw_html)
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
