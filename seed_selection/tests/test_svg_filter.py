import json
from pathlib import Path

from seed_selection.svg_filter import run_svg_filter


def _write(tmp_path: Path, records: list[dict]) -> Path:
    p = tmp_path / "in.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")
    return p


def _read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _rec(id_: str, instruction: str, svg_len: int, domain: str) -> dict:
    return {
        "instruction": instruction,
        "_meta": {"id": id_, "domain": domain, "source": "img2svg", "svg_len": svg_len},
    }


def test_svg_filter_removes_bottom_10pct(tmp_path):
    records = [
        _rec(f"f:{i}", f"Draw shape {i}.", i * 10, "stage1_icon")
        for i in range(1, 11)
    ]
    out = tmp_path / "out.jsonl"
    stats = run_svg_filter(_write(tmp_path, records), out, bottom_pct=0.10)

    kept = _read(out)
    assert len(kept) == 9
    svg_lens = [r["_meta"]["svg_len"] for r in kept]
    assert min(svg_lens) > 10


def test_svg_filter_keeps_illustration(tmp_path):
    records = [
        _rec("f:1", "Desc 1.", 5,  "stage2_illustration"),
        _rec("f:2", "Desc 2.", 10, "stage2_illustration"),
    ]
    out = tmp_path / "out.jsonl"
    stats = run_svg_filter(_write(tmp_path, records), out, bottom_pct=0.50)

    kept = _read(out)
    assert len(kept) == 2


def test_svg_filter_mixed_domains(tmp_path):
    records = [
        _rec("icon:1", "A.", 10,  "stage1_icon"),
        _rec("icon:2", "B.", 200, "stage1_icon"),
        _rec("ill:1",  "C.", 5,   "stage2_illustration"),
    ]
    out = tmp_path / "out.jsonl"
    run_svg_filter(_write(tmp_path, records), out, bottom_pct=0.50)
    kept = _read(out)
    kept_ids = {r["_meta"]["id"] for r in kept}
    assert "icon:2" in kept_ids
    assert "ill:1"  in kept_ids
    assert "icon:1" not in kept_ids
