import json
from pathlib import Path

from seed_selection.dedup_exact import run_dedup_exact


def _write(tmp_path: Path, records: list[dict]) -> Path:
    p = tmp_path / "in.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")
    return p


def _read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def test_dedup_exact_removes_duplicates(tmp_path):
    records = [
        {"id": "f:1", "instruction": "Draw a star.", "svg_len": 100, "domain": "stage1_icon", "source": "img2svg"},
        {"id": "f:2", "instruction": "Draw a star.", "svg_len": 100, "domain": "stage1_icon", "source": "text2svg"},
        {"id": "f:3", "instruction": "Draw a circle.", "svg_len": 80, "domain": "stage1_icon", "source": "img2svg"},
    ]
    out = tmp_path / "out.jsonl"
    stats = run_dedup_exact(_write(tmp_path, records), out)

    assert stats.kept == 2
    assert stats.removed == 1
    kept = _read(out)
    assert len(kept) == 2
    instructions = {r["instruction"] for r in kept}
    assert instructions == {"Draw a star.", "Draw a circle."}


def test_dedup_exact_img2svg_priority(tmp_path):
    """text2svg 先出现，但 img2svg 后出现时应替换成 img2svg 版本。"""
    records = [
        {"id": "f:1", "instruction": "Draw a star.", "svg_len": 100, "domain": "stage1_icon", "source": "text2svg"},
        {"id": "f:2", "instruction": "Draw a star.", "svg_len": 200, "domain": "stage1_icon", "source": "img2svg"},
    ]
    out = tmp_path / "out.jsonl"
    stats = run_dedup_exact(_write(tmp_path, records), out)

    kept = _read(out)
    assert len(kept) == 1
    assert kept[0]["source"] == "img2svg"
    assert stats.replaced == 1


def test_dedup_exact_same_priority_keeps_first(tmp_path):
    """相同优先级时保留先出现的。"""
    records = [
        {"id": "f:1", "instruction": "Draw a star.", "svg_len": 100, "domain": "stage1_icon", "source": "img2svg"},
        {"id": "f:2", "instruction": "Draw a star.", "svg_len": 150, "domain": "stage1_icon", "source": "img2svg"},
    ]
    out = tmp_path / "out.jsonl"
    run_dedup_exact(_write(tmp_path, records), out)
    kept = _read(out)
    assert len(kept) == 1
    assert kept[0]["id"] == "f:1"
