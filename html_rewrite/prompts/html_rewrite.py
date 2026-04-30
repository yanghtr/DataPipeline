"""HTML 改写 prompt 模块。接口与 distillation/prompts/svg.py 一致，供动态加载。"""

from __future__ import annotations

from utils.api_client import text_content

SYSTEM_PROMPT: str = (
    "You are an expert web developer. "
    "Your task is to rewrite a preprocessed dirty HTML into a clean, self-contained, single-file HTML "
    "that renders correctly in a browser. "
    "The input HTML has media resources replaced with placeholder paths "
    "(format: __MEDIA_PLACEHOLDER__/media__width{W}__height{H}.ext). "
    "Keep the page structure, layout, navigation, sidebar, footer, and content intact. "
    "Replace placeholder paths with reasonable placeholder image/video/audio URLs or simple inline SVG. "
    "Remove excessive boilerplate scripts and styles while preserving layout-critical CSS. "
    "Output exactly one complete HTML document. "
    "Do not include any explanation or markdown — output only the raw HTML."
)


def build_user_content(preprocessed_html: str) -> list[dict]:
    prompt = (
        "Rewrite the following preprocessed HTML into a clean, self-contained, single-file HTML "
        "that renders correctly in a modern browser.\n\n"
        "Requirements:\n"
        "- Preserve the overall page structure, layout, and content.\n"
        "- Keep navigation, sidebar, footer, and main content areas.\n"
        "- Replace __MEDIA_PLACEHOLDER__ paths with appropriate placeholder resources.\n"
        "- Inline all CSS needed for layout (remove external stylesheet links).\n"
        "- Remove or stub out JavaScript (it is not needed for static rendering).\n"
        "- The output must be a complete HTML document starting with <!DOCTYPE html>.\n"
        "- Do not wrap the output in a markdown code block. Output raw HTML only.\n\n"
        f"Input HTML:\n{preprocessed_html}"
    )
    return text_content(prompt)
