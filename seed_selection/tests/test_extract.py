import pytest
from pathlib import Path
from seed_selection.extract import run_extract


def test_extract_counts(tmp_path, all_input_paths):
    out = tmp_path / "raw.jsonl"
    stats = run_extract(all_input_paths, out)

    # fixtures: 5 + 2 + 2 + 3 = 12 total lines
    assert stats.total == 12
    assert stats.extracted > 0
    assert out.exists()


def test_extract_fields(tmp_path, all_input_paths):
    out = tmp_path / "raw.jsonl"
    run_extract(all_input_paths, out)

    import json
    records = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert len(records) > 0
    for rec in records:
        assert "id" in rec
        assert "instruction" in rec and isinstance(rec["instruction"], str)
        assert "svg_len" in rec and isinstance(rec["svg_len"], int) and rec["svg_len"] > 0
        assert "domain" in rec and rec["domain"] in ("stage1_icon", "stage2_icon", "stage2_illustration")
        assert "source" in rec and rec["source"] in ("img2svg", "text2svg")


def test_extract_img2svg_before_text2svg(tmp_path, all_input_paths):
    """img2svg 记录必须排在 text2svg 之前（保证 exact dedup 中 img2svg 优先）。"""
    out = tmp_path / "raw.jsonl"
    run_extract(all_input_paths, out)

    import json
    records = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    sources = [r["source"] for r in records]
    # 找到第一个 text2svg 的位置
    first_text = next((i for i, s in enumerate(sources) if s == "text2svg"), len(sources))
    # 其后不应有 img2svg
    assert all(s != "img2svg" for s in sources[first_text:])


def test_extract_dry_run(tmp_path, all_input_paths):
    out = tmp_path / "raw.jsonl"
    stats = run_extract(all_input_paths, out, dry_run_limit=2)
    # 每个文件最多读 2 行
    assert stats.total <= 2 * len(all_input_paths)
