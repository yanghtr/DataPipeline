import json
from pathlib import Path

from seed_selection.svg_filter import run_svg_filter


def _write(tmp_path: Path, records: list[dict]) -> Path:
    p = tmp_path / "in.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")
    return p


def _read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def test_svg_filter_removes_bottom_10pct(tmp_path):
    # 10 条 stage1_icon，svg_len 为 10,20,...,100
    records = [
        {"id": f"f:{i}", "instruction": f"Draw shape {i}.", "svg_len": i * 10,
         "domain": "stage1_icon", "source": "img2svg"}
        for i in range(1, 11)
    ]
    out = tmp_path / "out.jsonl"
    stats = run_svg_filter(_write(tmp_path, records), out, bottom_pct=0.10)

    kept = _read(out)
    # 10% of 10 = 1 removed（svg_len=10, cutoff=10）
    assert len(kept) == 9
    svg_lens = [r["svg_len"] for r in kept]
    assert min(svg_lens) > 10


def test_svg_filter_keeps_illustration(tmp_path):
    records = [
        {"id": "f:1", "instruction": "Desc 1.", "svg_len": 5,
         "domain": "stage2_illustration", "source": "img2svg"},
        {"id": "f:2", "instruction": "Desc 2.", "svg_len": 10,
         "domain": "stage2_illustration", "source": "img2svg"},
    ]
    out = tmp_path / "out.jsonl"
    stats = run_svg_filter(_write(tmp_path, records), out, bottom_pct=0.50)

    kept = _read(out)
    assert len(kept) == 2  # illustration 全保留


def test_svg_filter_mixed_domains(tmp_path):
    records = [
        {"id": "icon:1", "instruction": "A.", "svg_len": 10, "domain": "stage1_icon", "source": "img2svg"},
        {"id": "icon:2", "instruction": "B.", "svg_len": 200, "domain": "stage1_icon", "source": "img2svg"},
        {"id": "ill:1",  "instruction": "C.", "svg_len": 5,   "domain": "stage2_illustration", "source": "img2svg"},
    ]
    out = tmp_path / "out.jsonl"
    run_svg_filter(_write(tmp_path, records), out, bottom_pct=0.50)
    kept = _read(out)
    kept_ids = {r["id"] for r in kept}
    # icon:1 (svg_len=10, bottom 50%) removed; icon:2 kept; ill:1 always kept
    assert "icon:2" in kept_ids
    assert "ill:1"  in kept_ids
    assert "icon:1" not in kept_ids
