"""核心数据模型：SampleRecord + panguml I/O 辅助函数。"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from loguru import logger


# ── SampleRecord ──────────────────────────────────────────────────────────────


@dataclass
class SampleRecord:
    """一条样本在整个流水线中的完整状态。"""

    sample_id: str          # "{jsonl_stem}:{line_no}"
    jsonl_file: str
    line_no: int
    raw_sample: dict        # 原始 panguml 样本，全程不修改

    image_rel_path: str
    image_format: str       # image/jpeg | image/png
    image_width: int
    image_height: int
    raw_html: str

    # Phase 1
    html_outline_json: Optional[dict] = None
    outline_text: Optional[str] = None
    extraction_status: str = "pending"
    extraction_warnings: list[str] = field(default_factory=list)
    extraction_timeout: bool = False

    # Phase 2
    vlm_model: Optional[str] = None
    vlm_reasoning_raw: Optional[str] = None
    reasoning_text: Optional[str] = None
    final_answer: Optional[str] = None
    generation_status: str = "pending"
    generation_warnings: list[str] = field(default_factory=list)
    quality_metadata: Optional[dict] = None

    def to_panguml_output(self) -> dict:
        out = copy.deepcopy(self.raw_sample)
        if self.final_answer is not None:
            out["data"][1]["content"][0]["text"]["string"] = self.final_answer
        return out

    def to_debug_dict(self, api_base: str = "") -> dict:
        qm = self.quality_metadata or {}
        qm_full = {
            **qm,
            "extraction_warnings": self.extraction_warnings,
            "generation_warnings": self.generation_warnings,
        }
        return {
            "sample_id": self.sample_id,
            "jsonl_file": self.jsonl_file,
            "line_no": self.line_no,
            "image_rel_path": self.image_rel_path,
            "raw_html_len": len(self.raw_html),
            "html_outline_json": self.html_outline_json,
            "outline_text": self.outline_text,
            "vlm": {
                "model": self.vlm_model,
                "url": api_base,
            },
            "prompt_version": "image_to_html_cot_v8",
            "vlm_reasoning_raw": self.vlm_reasoning_raw,
            "reasoning_text": self.reasoning_text,
            "final_answer": self.final_answer,
            "status": {
                "extraction": self.extraction_status,
                "generation": self.generation_status,
            },
            "quality_metadata": qm_full,
        }

    def to_outline_cache_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "html_outline_json": self.html_outline_json,
            "outline_text": self.outline_text,
            "extraction_status": self.extraction_status,
            "extraction_warnings": self.extraction_warnings,
            "extraction_timeout": self.extraction_timeout,
        }

    def apply_outline_cache(self, cached: dict) -> None:
        self.html_outline_json = cached.get("html_outline_json")
        self.outline_text = cached.get("outline_text")
        self.extraction_status = cached.get("extraction_status", "ok")
        self.extraction_warnings = cached.get("extraction_warnings", [])
        self.extraction_timeout = cached.get("extraction_timeout", False)


# ── panguml I/O ───────────────────────────────────────────────────────────────


def iter_samples(jsonl_path: str | Path) -> Iterator[tuple[int, dict]]:
    """从 panguml JSONL 逐行 yield (line_no, sample)。line_no 从 0 计。"""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON 解析错误 {jsonl_path}:{line_no}: {e}")


def extract_panguml_fields(sample: dict) -> tuple[str, str, int, int, str]:
    """从 panguml 样本提取 (rel_path, format, width, height, html)。"""
    rel_path = img_fmt = ""
    width = height = 0
    for item in sample["data"][0]["content"]:
        if item["type"] == "image":
            img = item["image"]
            rel_path = img["relative_path"]
            img_fmt = img["format"]
            width = img.get("width", 0)
            height = img.get("height", 0)
            break
    html = ""
    for item in sample["data"][1]["content"]:
        if item["type"] == "text":
            html = item["text"]["string"]
            break
    return rel_path, img_fmt, width, height, html


def build_sample_records(jsonl_path: str | Path) -> list[SampleRecord]:
    """读取一个 panguml JSONL，返回 SampleRecord 列表。"""
    path = Path(jsonl_path)
    stem = path.stem
    records = []
    for line_no, sample in iter_samples(path):
        try:
            rel_path, img_fmt, width, height, html = extract_panguml_fields(sample)
        except (KeyError, IndexError) as e:
            logger.warning(f"样本字段缺失 {path}:{line_no}: {e}")
            continue
        sample_id = f"{stem}:{line_no}"
        records.append(
            SampleRecord(
                sample_id=sample_id,
                jsonl_file=str(path),
                line_no=line_no,
                raw_sample=sample,
                image_rel_path=rel_path,
                image_format=img_fmt,
                image_width=width,
                image_height=height,
                raw_html=html,
            )
        )
    return records


# ── 文件 I/O ──────────────────────────────────────────────────────────────────


def load_outline_cache(cache_path: str | Path) -> dict[str, dict]:
    """加载 Phase 1 outline 缓存，返回 {sample_id: cached_dict}。"""
    path = Path(cache_path)
    if not path.exists():
        return {}
    result: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                result[d["sample_id"]] = d
            except (json.JSONDecodeError, KeyError):
                pass
    return result


def load_done_ids(done_path: str | Path) -> set[str]:
    """从 done 记录文件读取已完成的 sample_id 集合（每行一个 id）。"""
    path = Path(done_path)
    if not path.exists():
        return set()
    ids: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            sid = line.strip()
            if sid:
                ids.add(sid)
    return ids


def append_jsonl(obj: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def append_done_id(sample_id: str, done_path: str | Path) -> None:
    p = Path(done_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(sample_id + "\n")
