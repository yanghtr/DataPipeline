"""
App 1：单条调用 debug 脚本。

修改下方参数后直接运行：
    python -m distillation.demo
    # 或
    python distillation/demo.py

适用场景：快速验证 API 连通性、prompt 效果，无需配置文件。
"""

from pathlib import Path

from utils.api_client import call_chat_completion, image_text_content, text_content
from distillation.prompt import SVG_SYSTEM_PROMPT, build_svg_user_content

# ── 修改以下参数 ──────────────────────────────────────────────────────────────
URL     = "http://localhost:8000/v1/chat/completions"  # 完整 endpoint
API_KEY = "your-api-key"
MODEL   = "your-model-name"

# 纯文本 SVG 请求（最常用）：
TEXT  = "Draw a simple red circle with a blue border"
IMAGE = None  # 如需图文请求，填本地图片路径，如 "/path/to/ref.png"

# 调用结果保存路径（None 则不保存到文件）
RESULT_LOG = Path("logs/demo_calls.jsonl")
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if IMAGE:
        # 图文请求：图片 + 文字描述
        user_content = image_text_content(IMAGE, TEXT)
    else:
        # 纯文本 SVG 蒸馏请求（使用标准 SVG prompt 模板）
        user_content = build_svg_user_content(TEXT)

    result = call_chat_completion(
        url=URL,
        api_key=API_KEY,
        model=MODEL,
        user_content=user_content,
        system=SVG_SYSTEM_PROMPT,
        result_log_path=RESULT_LOG,
    )
    print(result)
