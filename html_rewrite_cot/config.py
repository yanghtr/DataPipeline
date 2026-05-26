"""html_rewrite_cot 流水线配置：dataclass 定义 + YAML 加载。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class VLMConfig:
    # 字段命名与 html_rewrite.config.HtmlRewriteConfig 对齐
    url: str                  # 完整 endpoint，如 http://host/v1/chat/completions
    api_key: str
    model: str
    max_retries: int = 3
    timeout: float = 120.0
    ssl_verify: bool = True
    log_user: str = "html_rewrite_cot"
    # generation_params 透传到 API payload，与 html_rewrite 保持一致
    # 常用字段：temperature, top_p, max_tokens, stop
    # 注意：stop 中避免使用 "<html" / "<!DOCTYPE html>"——这类模式会干扰
    # 后续任何需要在 reasoning 里分析 HTML 标签的场景；只用 "```html" 即可。
    generation_params: dict = field(default_factory=lambda: {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 2048,
        "stop": ["```html"],
    })


@dataclass
class RuntimeConfig:
    num_workers: int = 4
    playwright_concurrency: int = 2
    resume: bool = True
    run_dir: str = "./run"


@dataclass
class RenderConfig:
    viewport_strategy: str = "match_image"
    viewport_width: Optional[int] = None
    viewport_height: Optional[int] = None
    fallback_width: int = 1280
    fallback_height: int = 900
    content_timeout_ms: int = 10000
    js_timeout_ms: int = 5000
    block_external: bool = True


@dataclass
class InputConfig:
    image_root: str
    jsonl_files: list[str] = field(default_factory=list)


@dataclass
class OutputConfig:
    output_dir: str
    debug_dir: Optional[str] = None


@dataclass
class PipelineConfig:
    input: InputConfig
    output: OutputConfig
    vlm: VLMConfig
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    max_input_chars: int = 262144


def load_config(path: Path) -> PipelineConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return PipelineConfig(
        input=InputConfig(**raw["input"]),
        output=OutputConfig(**raw["output"]),
        vlm=VLMConfig(**raw["vlm"]),
        runtime=RuntimeConfig(**raw.get("runtime", {})),
        render=RenderConfig(**raw.get("render", {})),
        max_input_chars=raw.get("max_input_chars", 262144),
    )
