import json
from pathlib import Path

from seed_selection.dedup_exact import run_dedup_exact


def _rec(id_: str, instruction: str, source: str, domain: str = "stage1_icon",
         svg_len: int = 100) -> dict:
    return {
        "instruction": instruction,
        "_meta": {"id": id_, "domain": domain, "source": source, "svg_len": svg_len},
    }


def _write(tmp_path: Path, records: list[dict]) -> Path:
    p = tmp_path / "in.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")
    return p


def _read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def test_dedup_exact_removes_duplicates(tmp_path):
    records = [
        _rec("f:1", "Draw a star.", "img2svg"),
        _rec("f:2", "Draw a star.", "text2svg"),
        _rec("f:3", "Draw a circle.", "img2svg"),
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
        _rec("f:1", "Draw a star.", "text2svg", svg_len=100),
        _rec("f:2", "Draw a star.", "img2svg",  svg_len=200),
    ]
    out = tmp_path / "out.jsonl"
    stats = run_dedup_exact(_write(tmp_path, records), out)

    kept = _read(out)
    assert len(kept) == 1
    assert kept[0]["_meta"]["source"] == "img2svg"
    assert stats.replaced == 1


def test_dedup_exact_same_priority_keeps_first(tmp_path):
    """相同优先级时保留先出现的。"""
    records = [
        _rec("f:1", "Draw a star.", "img2svg", svg_len=100),
        _rec("f:2", "Draw a star.", "img2svg", svg_len=150),
    ]
    out = tmp_path / "out.jsonl"
    run_dedup_exact(_write(tmp_path, records), out)
    kept = _read(out)
    assert len(kept) == 1
    assert kept[0]["_meta"]["id"] == "f:1"
