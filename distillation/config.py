"""蒸馏流水线配置：dataclass 定义 + YAML 加载。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class DistillConfig:
    # ── API ──────────────────────────────────────────────────────────────────
    url: str                          # 完整 endpoint，含路径，如 "http://host/v1/chat/completions"
    api_key: str
    model: str
    timeout: float = 120.0
    max_retries: int = 3
    ssl_verify: bool = True           # 本地 vLLM 自签名证书时设为 false
    log_user: str = "svg_distill"

    # ── 路径 ─────────────────────────────────────────────────────────────────
    input_path: str = ""              # 种子 JSONL（high_priority_pool.jsonl 等）
    output_path: str = "distillation_output.jsonl"   # 蒸馏结果写出路径
    call_log_path: str = "logs/api_calls.jsonl"      # 每次原始 API 调用记录

    # ── 运行 ─────────────────────────────────────────────────────────────────
    num_workers: int = 16             # 并发线程数（I/O bound，16–32 通常足够）
    resume: bool = True               # 断点续跑：跳过已有输出中的 id


def load_config(path: Path) -> DistillConfig:
    """从 YAML 文件加载配置，返回强类型 DistillConfig。"""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return DistillConfig(**raw)
