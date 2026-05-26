"""最终答案拼接：reasoning_text + fenced raw_html。"""

from __future__ import annotations


def assemble_final_answer(reasoning_text: str, raw_html: str) -> str:
    return f"{reasoning_text}\n\n```html\n{raw_html}\n```"
