"""流水线配置：dataclass 定义 + YAML 加载。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class EmbedConfig:
    model_path: str
    dimension: int = 256
    batch_size: int = 16
    device: str = "cpu"
    shard_size: int = 100_000
    num_devices: int = 1   # 使用的 GPU/NPU 卡数（CPU 模式固定为 1）


@dataclass
class NearDedupConfig:
    num_perm: int = 128
    char_ngram: int = 5
    thresholds: dict[str, float] = field(default_factory=lambda: {
        "stage1_icon": 0.8,
        "stage2_icon": 0.8,
        "stage2_illustration": 0.7,
    })


@dataclass
class ClusterConfig:
    k_per_bucket: dict[str, int] = field(default_factory=lambda: {
        "stage1_icon": 10_000,
        "stage2_icon": 2_000,
        "stage2_illustration": 3_000,
    })
    random_seed: int = 42
    minibatch_size: int = 50_000


@dataclass
class SamplingConfig:
    total_pool_size: int = 1_000_000
    anneal_pool_size: int = 900_000
    high_priority_pool_size: int = 100_000
    random_seed: int = 42


@dataclass
class PipelineConfig:
    input_paths: list[str]
    output_root: str
    embedding: EmbedConfig
    near_dedup: NearDedupConfig
    clustering: ClusterConfig
    sampling: SamplingConfig
    svg_filter_bottom_pct: float = 0.10
    num_workers: int = 4   # extract / dedup_near / cluster 的并行进程数


def load_config(path: Path) -> PipelineConfig:
    """从 YAML 文件加载配置，返回强类型 PipelineConfig。"""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    emb = raw.get("embedding", {})
    nd  = raw.get("near_dedup", {})
    cl  = raw.get("clustering", {})
    sa  = raw.get("sampling", {})

    return PipelineConfig(
        input_paths=raw["input_paths"],
        output_root=raw["output_root"],
        svg_filter_bottom_pct=raw.get("svg_filter_bottom_pct", 0.10),
        embedding=EmbedConfig(
            model_path=emb["model_path"],
            dimension=emb.get("dimension", 256),
            batch_size=emb.get("batch_size", 16),
            device=emb.get("device", "cpu"),
            shard_size=emb.get("shard_size", 100_000),
            num_devices=emb.get("num_devices", 1),
        ),
        near_dedup=NearDedupConfig(
            num_perm=nd.get("num_perm", 128),
            char_ngram=nd.get("char_ngram", 5),
            thresholds=nd.get("thresholds", {
                "stage1_icon": 0.8,
                "stage2_icon": 0.8,
                "stage2_illustration": 0.7,
            }),
        ),
        clustering=ClusterConfig(
            k_per_bucket=cl.get("k_per_bucket", {
                "stage1_icon": 10_000,
                "stage2_icon": 2_000,
                "stage2_illustration": 3_000,
            }),
            random_seed=cl.get("random_seed", 42),
            minibatch_size=cl.get("minibatch_size", 50_000),
        ),
        sampling=SamplingConfig(
            total_pool_size=sa.get("total_pool_size", 1_000_000),
            anneal_pool_size=sa.get("anneal_pool_size", 900_000),
            high_priority_pool_size=sa.get("high_priority_pool_size", 100_000),
            random_seed=sa.get("random_seed", 42),
        ),
        num_workers=raw.get("num_workers", 4),
    )
