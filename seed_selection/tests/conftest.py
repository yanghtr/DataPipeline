"""共用 fixtures 和工具函数。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> list[dict]:
    """读取 fixtures/ 下的 JSONL 文件，返回记录列表。"""
    return [json.loads(line) for line in (FIXTURES_DIR / name).read_text().splitlines() if line.strip()]


@pytest.fixture()
def stage1_icon_img2svg_path() -> Path:
    return FIXTURES_DIR / "stage1_icon_img2svg.jsonl"


@pytest.fixture()
def stage1_icon_text2svg_path() -> Path:
    return FIXTURES_DIR / "stage1_icon_text2svg.jsonl"


@pytest.fixture()
def stage2_icon_img2svg_path() -> Path:
    return FIXTURES_DIR / "stage2_icon_img2svg.jsonl"


@pytest.fixture()
def stage2_illustration_img2svg_path() -> Path:
    return FIXTURES_DIR / "stage2_illustration_img2svg.jsonl"


@pytest.fixture()
def all_input_paths(
    stage1_icon_img2svg_path,
    stage2_icon_img2svg_path,
    stage2_illustration_img2svg_path,
    stage1_icon_text2svg_path,
) -> list[Path]:
    """img2svg 在前，text2svg 在后（与 default.yaml 一致）。"""
    return [
        stage1_icon_img2svg_path,
        stage2_icon_img2svg_path,
        stage2_illustration_img2svg_path,
        stage1_icon_text2svg_path,
    ]
