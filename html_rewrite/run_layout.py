"""输入分片展开、run 目录布局与 manifest 管理。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from loguru import logger


@dataclass(frozen=True)
class InputShardSpec:
    shard_index: int
    shard_name: str
    input_path: str


@dataclass(frozen=True)
class Stage1ShardPaths:
    output_path: Path
    stats_log_path: Path
    reject_log_path: Path
    summary_log_path: Path
    stats_plot_dir: Path


@dataclass(frozen=True)
class Stage2ShardPaths:
    input_path: Path
    output_path: Path
    call_log_path: Path


def stage1_shard_paths(run_root_dir: Path, shard_name: str) -> Stage1ShardPaths:
    shard_dir = run_root_dir / "stage1" / shard_name
    return Stage1ShardPaths(
        output_path=shard_dir / "preprocessed.jsonl",
        stats_log_path=shard_dir / "stats.jsonl",
        reject_log_path=shard_dir / "rejects.jsonl",
        summary_log_path=shard_dir / "summary.json",
        stats_plot_dir=shard_dir / "plots",
    )


def stage2_shard_paths(run_root_dir: Path, shard_name: str) -> Stage2ShardPaths:
    stage1_dir = run_root_dir / "stage1" / shard_name
    stage2_dir = run_root_dir / "stage2" / shard_name
    return Stage2ShardPaths(
        input_path=stage1_dir / "preprocessed.jsonl",
        output_path=stage2_dir / "output.jsonl",
        call_log_path=stage2_dir / "api_calls.jsonl",
    )


def aggregate_dir(run_root_dir: Path) -> Path:
    return run_root_dir / "aggregate"


def ensure_manifest(run_root_dir: Path, manifest_payload: dict) -> dict:
    """创建或校验 run manifest，防止 resume 到不一致的输入布局。"""
    run_root_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_root_dir / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_manifest(existing, manifest_payload)
        return existing

    tmp_path = manifest_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(manifest_path)
    logger.info(f"[layout] manifest 已写出：{manifest_path}")
    return manifest_payload


def build_stage1_aggregate_summary(run_root_dir: Path, shard_specs: list[InputShardSpec]) -> Path:
    out_dir = aggregate_dir(run_root_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "stage1_summary.json"

    shard_summaries = []
    total_input = 0
    total_kept = 0
    total_rejected = 0
    reject_reasons: dict[str, int] = {}

    for shard in shard_specs:
        summary_path = stage1_shard_paths(run_root_dir, shard.shard_name).summary_log_path
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        shard_summaries.append(
            {
                "shard_name": shard.shard_name,
                "input_path": shard.input_path,
                "summary_path": str(summary_path),
                "total_input": summary.get("total_input", 0),
                "kept": summary.get("kept", 0),
                "rejected": summary.get("rejected", 0),
                "reject_reasons": summary.get("reject_reasons", {}),
            }
        )
        total_input += summary.get("total_input", 0)
        total_kept += summary.get("kept", 0)
        total_rejected += summary.get("rejected", 0)
        for reason, count in (summary.get("reject_reasons") or {}).items():
            reject_reasons[reason] = reject_reasons.get(reason, 0) + count

    payload = {
        "total_shards": len(shard_specs),
        "summarized_shards": len(shard_summaries),
        "total_input": total_input,
        "kept": total_kept,
        "rejected": total_rejected,
        "reject_reasons": dict(sorted(reject_reasons.items())),
        "shards": shard_summaries,
    }
    _atomic_write_json(out_path, payload)
    return out_path


def build_stage2_aggregate_summary(run_root_dir: Path, shard_specs: list[InputShardSpec]) -> Path:
    out_dir = aggregate_dir(run_root_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "stage2_summary.json"

    shard_summaries = []
    total_outputs = 0

    for shard in shard_specs:
        shard_paths = stage2_shard_paths(run_root_dir, shard.shard_name)
        if not shard_paths.output_path.exists():
            continue
        line_count = _count_lines(shard_paths.output_path)
        shard_summaries.append(
            {
                "shard_name": shard.shard_name,
                "input_path": shard.input_path,
                "output_path": str(shard_paths.output_path),
                "records": line_count,
            }
        )
        total_outputs += line_count

    payload = {
        "total_shards": len(shard_specs),
        "summarized_shards": len(shard_summaries),
        "total_outputs": total_outputs,
        "shards": shard_summaries,
    }
    _atomic_write_json(out_path, payload)
    return out_path


def build_manifest_payload(
    *,
    input_mode: str,
    input_dir: str,
    input_filename_template: str,
    input_start_index: int | None,
    input_end_index_exclusive: int | None,
    output_shard_name_template: str,
    shard_specs: list[InputShardSpec],
) -> dict:
    return {
        "schema_version": 1,
        "input_mode": input_mode,
        "input_dir": input_dir,
        "input_filename_template": input_filename_template,
        "input_start_index": input_start_index,
        "input_end_index_exclusive": input_end_index_exclusive,
        "output_shard_name_template": output_shard_name_template,
        "shards": [asdict(shard) for shard in shard_specs],
    }


def _validate_manifest(existing: dict, expected: dict) -> None:
    keys = (
        "schema_version",
        "input_mode",
        "input_dir",
        "input_filename_template",
        "input_start_index",
        "input_end_index_exclusive",
        "output_shard_name_template",
        "shards",
    )
    for key in keys:
        if existing.get(key) != expected.get(key):
            raise RuntimeError(
                f"Existing run manifest mismatch on `{key}`. "
                "Please use a new `run_root_dir` or clean the old run outputs."
            )


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _count_lines(path: Path) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)
