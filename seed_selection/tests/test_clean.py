import json
from pathlib import Path

import pytest

from seed_selection.clean import clean_instruction, is_valid, run_clean


def test_clean_strips_whitespace():
    assert clean_instruction("  hello world  ") == "hello world"


def test_clean_nfc_normalization():
    # é as NFD (e + combining accent) → NFC é
    nfd = "e\u0301"
    result = clean_instruction(nfd)
    assert result == "\xe9"


def test_is_valid_short():
    assert not is_valid("ab")
    assert is_valid("abc")


def _make_raw(tmp_path: Path, records: list[dict]) -> Path:
    p = tmp_path / "raw.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")
    return p


def test_run_clean_filters_empty(tmp_path):
    raw = _make_raw(tmp_path, [
        {"id": "f:1", "instruction": "  ", "svg_len": 100, "domain": "stage1_icon", "source": "img2svg"},
        {"id": "f:2", "instruction": "Draw a red star.", "svg_len": 100, "domain": "stage1_icon", "source": "img2svg"},
    ])
    out = tmp_path / "clean.jsonl"
    stats = run_clean(raw, out)
    assert stats.kept == 1
    assert stats.skip_empty == 1


def test_run_clean_preserves_fields(tmp_path):
    raw = _make_raw(tmp_path, [
        {"id": "f:1", "instruction": " Draw a star. ", "svg_len": 50, "domain": "stage1_icon", "source": "img2svg"},
    ])
    out = tmp_path / "clean.jsonl"
    run_clean(raw, out)
    rec = json.loads(out.read_text().strip())
    assert rec["instruction"] == "Draw a star."
    assert rec["svg_len"] == 50
