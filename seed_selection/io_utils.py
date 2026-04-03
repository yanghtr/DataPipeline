"""JSONL 读写、ID 生成、source 优先级等共用工具。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

# img2svg 来源优先级高于 text2svg（数字越小越优先）
SOURCE_PRIORITY: dict[str, int] = {
    "img2svg": 0,
    "text2svg": 1,
}

DOMAINS = ("stage1_icon", "stage2_illustration")


def make_id(source_file: str, line_no: int) -> str:
    """生成全局唯一记录 ID。"""
    return f"{source_file}:{line_no}"


def get_meta(rec: dict) -> dict:
    """返回记录的 _meta 字典（不存在时返回空 dict）。"""
    return rec.get("_meta", {})


def update_meta(rec: dict, **kwargs) -> dict:
    """将 kwargs 合并进 rec["_meta"]，返回更新后的 rec。"""
    if "_meta" not in rec:
        rec["_meta"] = {}
    rec["_meta"].update(kwargs)
    return rec


def read_jsonl(path: Path) -> Iterator[dict]:
    """逐行读取 JSONL，跳过空行，解析错误时静默跳过。"""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def write_jsonl(records: Iterable[dict], path: Path) -> int:
    """将记录写入 JSONL 文件，返回写入行数。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def infer_domain(file_path: str) -> str:
    """
    从文件路径推断 domain（两桶分类）。

    - stage2/illustration → "stage2_illustration"
    - stage1/icon 或 stage2/icon → "stage1_icon"
      （stage2_icon 的 instruction 在 exact dedup 中已被 stage1_icon 全量覆盖，
       因此两者统一归为 stage1_icon 桶，不再单独建桶。）
    """
    p = file_path.replace("\\", "/")
    if "stage2/illustration" in p or "stage2\\illustration" in p:
        return "stage2_illustration"
    if "illustration" in p:
        return "stage2_illustration"
    return "stage1_icon"


def infer_source(file_path: str) -> str:
    """从文件路径推断 source（img2svg 或 text2svg）。"""
    p = file_path.replace("\\", "/")
    if "text2svg" in p:
        return "text2svg"
    return "img2svg"
