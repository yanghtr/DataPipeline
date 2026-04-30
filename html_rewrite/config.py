"""html_rewrite 流水线配置：dataclass 定义 + YAML 加载。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .run_layout import InputShardSpec


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
    input_paths: list[str] = field(default_factory=list)    # 多个原始 JSONL（按列表顺序拼接）
    input_dir: str = ""                                     # 输入目录（模板展开模式）
    input_filename_template: str = ""                       # 例如 part-{index:05d}.jsonl
    input_start_index: int | None = None                    # 包含
    input_end_index_exclusive: int | None = None            # 不包含
    run_root_dir: str = ""                                  # 分片输出根目录
    output_shard_name_template: str = "part-{index:05d}"    # 输出 shard 目录名模板
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

    def stage1_input_paths(self) -> list[Path]:
        """返回 Stage 1 输入文件列表，保持 YAML 中的顺序。"""
        _, shards = self.resolve_input_shards()
        return [Path(shard.input_path) for shard in shards]

    def use_sharded_run(self) -> bool:
        return bool(self.run_root_dir)

    def resolve_input_shards(self) -> tuple[str, list[InputShardSpec]]:
        """
        解析输入配置，返回 (input_mode, shard_specs)。

        input_mode:
          - single
          - input_paths
          - template
        """
        has_paths = bool(self.input_paths)
        has_single = bool(self.input_path)
        has_template = any(
            [
                self.input_dir,
                self.input_filename_template,
                self.input_start_index is not None,
                self.input_end_index_exclusive is not None,
            ]
        )

        if has_paths and has_template:
            raise ValueError("Use either `input_paths` or template-based input fields, not both.")
        if has_single and has_template:
            raise ValueError("Use either `input_path` or template-based input fields, not both.")
        if has_paths and has_single:
            raise ValueError("Use either `input_paths` or `input_path`, not both.")

        if has_paths:
            return "input_paths", [
                InputShardSpec(
                    shard_index=i,
                    shard_name=self.output_shard_name_template.format(index=i),
                    input_path=path,
                )
                for i, path in enumerate(self.input_paths)
            ]

        if has_template:
            if not self.input_dir or not self.input_filename_template:
                raise ValueError("Template input mode requires `input_dir` and `input_filename_template`.")
            if self.input_start_index is None or self.input_end_index_exclusive is None:
                raise ValueError(
                    "Template input mode requires both `input_start_index` and `input_end_index_exclusive`."
                )
            if self.input_end_index_exclusive <= self.input_start_index:
                raise ValueError("`input_end_index_exclusive` must be greater than `input_start_index`.")
            return "template", [
                InputShardSpec(
                    shard_index=index,
                    shard_name=self.output_shard_name_template.format(index=index),
                    input_path=str(Path(self.input_dir) / self.input_filename_template.format(index=index)),
                )
                for index in range(self.input_start_index, self.input_end_index_exclusive)
            ]

        if has_single:
            return "single", [
                InputShardSpec(
                    shard_index=0,
                    shard_name=self.output_shard_name_template.format(index=0),
                    input_path=self.input_path,
                )
            ]

        raise ValueError(
            "Stage 1 requires one of: `input_path`, `input_paths`, or template input fields "
            "(`input_dir` + `input_filename_template` + range)."
        )


def load_config(path: Path) -> HtmlRewriteConfig:
    """从 YAML 文件加载配置，返回强类型 HtmlRewriteConfig。"""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return HtmlRewriteConfig(**raw)
