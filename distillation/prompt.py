"""SVG 蒸馏 prompt 模板。"""

from __future__ import annotations

from utils.api_client import text_content

SVG_SYSTEM_PROMPT: str = (
    "You are an expert SVG code generator. "
    "Generate clean, valid SVG based on the user's description. "
    "Return ONLY the SVG code with no explanation, no markdown fences."
)


def build_svg_user_content(instruction: str) -> list[dict]:
    """
    将种子 instruction 构造为 user 消息的 content 列表。

    返回格式符合 OpenAI content item 规范（type=text）。
    """
    prompt = (
        f"Generate an SVG that depicts the following:\n\n"
        f"{instruction}\n\n"
        "Requirements:\n"
        "- Return only valid SVG code\n"
        "- Start with <svg and end with </svg>\n"
        "- No markdown code blocks, no explanation"
    )
    return text_content(prompt)
