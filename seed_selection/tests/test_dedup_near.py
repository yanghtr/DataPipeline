import json
from pathlib import Path

import pytest

from seed_selection.dedup_near import run_dedup_near


def _write(tmp_path: Path, records: list[dict]) -> Path:
    p = tmp_path / "in.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")
    return p


def _read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _rec(id_: str, instruction: str, domain: str = "stage1_icon") -> dict:
    return {"id": id_, "instruction": instruction, "svg_len": 100, "domain": domain, "source": "img2svg"}


THRESHOLDS = {"stage1_icon": 0.8, "stage2_icon": 0.8, "stage2_illustration": 0.7}


def test_dedup_near_removes_near_dup(tmp_path):
    # 这两条 instruction 有大量重叠的 char 5-gram，应被判定为 near-dup
    near_dup_1 = "Generate a red five-pointed star on a white background canvas area."
    near_dup_2 = "Generate a red five-pointed star on a white background canvas area now."
    distinct   = "Draw a blue circle in the center of the image completely."

    records = [
        _rec("f:1", near_dup_1),
        _rec("f:2", near_dup_2),
        _rec("f:3", distinct),
    ]
    out = tmp_path / "out.jsonl"
    stats = run_dedup_near(_write(tmp_path, records), out, thresholds=THRESHOLDS)

    kept = _read(out)
    assert len(kept) == 2  # near_dup_2 removed
    kept_ids = {r["id"] for r in kept}
    assert "f:1" in kept_ids  # first one kept
    assert "f:3" in kept_ids


def test_dedup_near_keeps_distinct(tmp_path):
    records = [
        _rec("f:1", "Draw a red star on white background with five points."),
        _rec("f:2", "Create a blue circle centered on a gray canvas."),
        _rec("f:3", "Generate a yellow triangle with equal sides."),
    ]
    out = tmp_path / "out.jsonl"
    run_dedup_near(_write(tmp_path, records), out, thresholds=THRESHOLDS)
    kept = _read(out)
    assert len(kept) == 3  # all distinct


def test_dedup_near_cross_domain_no_interference(tmp_path):
    """不同 domain 的记录不互相去重。"""
    identical_text = "Draw a red five-pointed star with solid fill color."
    records = [
        _rec("f:1", identical_text, domain="stage1_icon"),
        _rec("f:2", identical_text, domain="stage2_illustration"),
    ]
    out = tmp_path / "out.jsonl"
    run_dedup_near(_write(tmp_path, records), out, thresholds=THRESHOLDS)
    kept = _read(out)
    assert len(kept) == 2  # 不同 domain 各保留一条
