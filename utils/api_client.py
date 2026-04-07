"""
通用 OpenAI-compatible Chat Completions 调用工具。

使用方式：
    from utils.api_client import call_chat_completion, text_content, image_text_content

    resp = call_chat_completion(
        url="http://localhost:8000/v1/chat/completions",
        api_key="token",
        model="your-model",
        user_content=text_content("Draw a red circle"),
        system="You are an SVG expert.",
    )
    content = resp["choices"][0]["message"]["content"]
"""

from __future__ import annotations

import base64
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from loguru import logger

# log_completion 是内部代码，未提供时自动降级为 no-op
try:
    from utils.local_api_logger import log_completion as _ext_log_completion  # type: ignore
except ImportError:
    _ext_log_completion = None

# 可重试的 HTTP 状态码
_RETRYABLE_STATUS: frozenset[int] = frozenset({429, 500, 502, 503, 504})
# 不重试的状态码（参数错误，重试无意义）
_NON_RETRYABLE_STATUS: frozenset[int] = frozenset({400, 401, 403, 404})


def call_chat_completion(
    url: str,
    api_key: str,
    model: str,
    user_content: list[dict],
    system: str | None = None,
    timeout: float = 120.0,
    max_retries: int = 3,
    ssl_verify: bool = True,
    log_user: str = "distill",
    result_log_path: Path | None = Path("logs/api_calls.jsonl"),
    extra_params: dict | None = None,
) -> dict:
    """
    调用 OpenAI-compatible chat completions，返回完整的 API 响应 dict。

    返回值即 API 原始 JSON，调用方自行提取所需字段，例如：
        resp = call_chat_completion(...)
        content = resp["choices"][0]["message"]["content"]
        usage   = resp.get("usage", {})

    Args:
        url:             完整 endpoint，如 "http://host/v1/chat/completions"
        api_key:         Bearer token
        model:           模型名称
        user_content:    user 消息的 content 列表（用 text_content / image_text_content 构造）
        system:          可选 system prompt
        timeout:         单次请求超时秒数
        max_retries:     最大重试次数（429/5xx/Timeout/ConnectionError）
        ssl_verify:      False 用于自签名证书的本地 vLLM
        log_user:        传给 log_completion 的 user 字段
        result_log_path: 每次调用结果（含输入 query 和完整响应）追加到此文件，None 则不记录
        extra_params:    透传到 payload 的额外参数，如 {"temperature": 0.2, "max_tokens": 8192}
                         会直接 merge 进请求 body，可覆盖或扩展任何 API 支持的字段

    Returns:
        API 原始响应 dict（OpenAI chat completions 格式）。

    Raises:
        requests.exceptions.HTTPError:       非 2xx 且不可重试时
        requests.exceptions.Timeout:         重试耗尽后仍超时
        requests.exceptions.ConnectionError: 重试耗尽后仍连接失败
    """
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_content})

    payload: dict = {"model": model, "messages": messages}
    if extra_params:
        payload.update(extra_params)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        resp = None
        try:
            resp = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout,
                verify=ssl_verify,
            )

            # 可重试的 HTTP 错误：先 sleep 再继续
            if resp.status_code in _RETRYABLE_STATUS and attempt < max_retries:
                delay = float(resp.headers.get("Retry-After", 2 ** attempt))
                logger.warning(
                    f"[api] HTTP {resp.status_code}, retry {attempt + 1}/{max_retries}"
                    f" after {delay:.0f}s"
                )
                time.sleep(delay)
                continue

            resp.raise_for_status()
            data: dict = resp.json()

            # 外部调用日志（失败不影响主流程）
            if _ext_log_completion:
                try:
                    _ext_log_completion(
                        model=model,
                        request_data=payload,
                        response_data=data,
                        user=log_user,
                    )
                except Exception as log_err:
                    logger.debug(f"[api] log_completion failed (ignored): {log_err}")

            _append_call_log(
                result_log_path,
                url=url,
                model=model,
                status="ok",
                user_content=user_content,
                system=system,
                response_data=data,
                error=None,
            )
            return data

        except requests.exceptions.Timeout as exc:
            last_exc = exc
            if attempt < max_retries:
                delay = 2 ** attempt
                logger.warning(f"[api] Timeout, retry {attempt + 1}/{max_retries} after {delay}s")
                time.sleep(delay)

        except requests.exceptions.ConnectionError as exc:
            last_exc = exc
            if attempt < max_retries:
                delay = 2 ** attempt
                logger.warning(
                    f"[api] ConnectionError, retry {attempt + 1}/{max_retries} after {delay}s"
                )
                time.sleep(delay)

        except requests.exceptions.HTTPError as exc:
            last_exc = exc
            status_code = resp.status_code if resp is not None else 0
            # 不可重试的客户端错误，立即放弃
            if status_code in _NON_RETRYABLE_STATUS or attempt == max_retries:
                _append_call_log(
                    result_log_path,
                    url=url,
                    model=model,
                    status="error",
                    user_content=user_content,
                    system=system,
                    response_data=None,
                    error=str(exc),
                )
                raise
            delay = 2 ** attempt
            logger.warning(
                f"[api] HTTPError {status_code}, retry {attempt + 1}/{max_retries} after {delay}s"
            )
            time.sleep(delay)

    # 重试全部耗尽
    _append_call_log(
        result_log_path,
        url=url,
        model=model,
        status="error",
        user_content=user_content,
        system=system,
        response_data=None,
        error=str(last_exc),
    )
    raise last_exc  # type: ignore[misc]


# ── Content builders ──────────────────────────────────────────────────────────


def text_content(text: str) -> list[dict]:
    """构造纯文本 content item。"""
    return [{"type": "text", "text": text}]


def image_text_content(
    image_path: str,
    text: str,
    image_first: bool = True,
    image_format: str = "image/png",
) -> list[dict]:
    """
    构造图文混合 content item 列表。

    Args:
        image_path:   本地图片路径
        text:         文字内容
        image_first:  True = 图片在前；False = 文字在前
        image_format: MIME 类型，"image/png" 或 "image/jpeg"
    """
    b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
    image_item: dict = {
        "type": "image_url",
        "image_url": {"url": f"data:{image_format};base64,{b64}"},
    }
    text_item: dict = {"type": "text", "text": text}
    return [image_item, text_item] if image_first else [text_item, image_item]


# ── 内部工具 ──────────────────────────────────────────────────────────────────


def _extract_log_query(user_content: list[dict]) -> tuple[str, list[dict]]:
    """
    从 content items 中提取可读信息用于日志记录。

    Returns:
        query_text: 所有 text item 拼接的文本
        images:     每个 image item 的摘要（记录 format，不记录 base64 数据）
    """
    texts: list[str] = []
    images: list[dict] = []
    for item in user_content:
        if item.get("type") == "text":
            texts.append(item.get("text", ""))
        elif item.get("type") == "image_url":
            url_str: str = item.get("image_url", {}).get("url", "")
            # 提取 MIME type，不记录 base64 数据本身
            fmt = url_str.split(";")[0].replace("data:", "") if url_str.startswith("data:") else "unknown"
            images.append({"type": "image", "format": fmt})
    return "\n".join(texts), images


def _append_call_log(
    path: Path | None,
    url: str,
    model: str,
    status: str,
    user_content: list[dict],
    system: str | None,
    response_data: dict | None,
    error: str | None,
) -> None:
    """
    将单次调用的完整信息追加到 JSONL 日志文件（失败时静默忽略）。

    记录内容：
      - ts / url / model / status
      - query: user 消息的文本内容
      - images: image item 摘要（format，无 base64）
      - system: system prompt（如有）
      - response: 完整 API 响应 dict（status=ok 时）
      - error: 错误信息字符串（status=error 时）
    """
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        query_text, images = _extract_log_query(user_content)
        record: dict = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "url": url,
            "model": model,
            "status": status,
            "system": system,
            "query": query_text,
            "images": images or None,       # None 表示纯文本请求
            "response": response_data,      # 完整响应（含 choices / usage / id）
            "error": error,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.debug(f"[api] call log write failed (ignored): {e}")
