"""html_rewrite 流水线配置：dataclass 定义 + YAML 加载。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class HtmlRewriteConfig:
    # ── API ──────────────────────────────────────────────────────────────────
    url: str
    api_key: str
    model: str
    timeout: float = 120.0
    max_retries: int = 3
    ssl_verify: bool = True
    log_user: str = "html_rewrite"

    # ── 路径 ─────────────────────────────────────────────────────────────────
    input_path: str = ""                                    # 原始 JSONL（Stage 1 输入）
    preprocessed_path: str = "preprocessed.jsonl"          # Stage 1 输出 / Stage 2 输入
    output_path: str = "html_rewrite_output.jsonl"         # Stage 2 最终输出
    call_log_path: str = "logs/api_calls.jsonl"            # API 调用原始记录
    stats_log_path: str = "logs/preprocess_stats.jsonl"    # 逐条预处理统计
    reject_log_path: str = "logs/preprocess_rejects.jsonl" # Stage 1 reject 样本
    summary_log_path: str = "logs/preprocess_summary.json" # Stage 1 聚合统计
    stats_plot_dir: str = "logs/preprocess_plots"          # Stage 1 分布图目录

    # ── 生成参数（透传到 API payload）────────────────────────────────────────
    generation_params: dict = field(default_factory=dict)

    # ── Prompt ───────────────────────────────────────────────────────────────
    prompt_module: str = "html_rewrite"

    # ── 预处理阈值 ────────────────────────────────────────────────────────────
    inline_script_max_chars: int = 4096
    json_payload_max_chars: int = 4096
    hidden_input_max_chars: int = 4096
    html_comment_max_chars: int = 1024
    inline_style_max_chars: int = 32768
    min_preprocessed_chars: int = 1024   # 过空 gate
    max_preprocessed_chars: int = 65536  # 超长 gate
    fetch_media_size: bool = False      # 是否尝试下载图片头部以获取尺寸（默认关闭）
    enable_language_filter: bool = True
    allowed_languages: list[str] = field(default_factory=lambda: ["en"])
    language_detector: str = "langid"
    language_min_visible_text_chars: int = 200
    language_min_letter_chars: int = 100
    language_sample_max_chars: int = 12000
    language_min_latin_ratio: float = 0.6
    language_min_detector_margin: float = 3.0

    # ── 运行 ─────────────────────────────────────────────────────────────────
    num_workers: int = 16
    resume: bool = True


def load_config(path: Path) -> HtmlRewriteConfig:
    """从 YAML 文件加载配置，返回强类型 HtmlRewriteConfig。"""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return HtmlRewriteConfig(**raw)
