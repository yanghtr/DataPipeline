"""SVG 蒸馏 prompt 模板。

接口约定（distill.py 动态加载时依赖）：
  SYSTEM_PROMPT: str
  build_user_content(instruction: str) -> list[dict]
"""

from __future__ import annotations

from utils.api_client import text_content

SYSTEM_PROMPT: str = (
    "You are an expert SVG generator. "
    "Generate clean, valid, self-contained, renderable SVG that matches the user's instruction. "
    "Focus on semantic accuracy and standards-compliant output."
)


def build_user_content(instruction: str) -> list[dict]:
    """
    将种子 instruction 构造为 user 消息的 content 列表。

    返回格式符合 OpenAI content item 规范（type=text）。
    """
    prompt = (
        "Design a scalable vector graphic (SVG) for the following instruction:\n\n"
        f"{instruction}\n\n"
        "Constraints:\n"
        "- The SVG must be self-contained and directly renderable.\n"
        "- Use standard XML/SVG syntax and include an explicit viewBox.\n"
        "- Include the root <svg> element with xmlns=\"http://www.w3.org/2000/svg\".\n"
        "- Do not use script, foreignObject, external images, external fonts, external stylesheets, or external references.\n"
        "- Return exactly one complete SVG inside a single ```svg ... ``` code block and no other text.\n"
        "- The SVG inside the code block must start with <svg and end with </svg>."
    )
    return text_content(prompt)
