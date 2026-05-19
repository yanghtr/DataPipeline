"""
Fix width/height fields in JSONL files to match actual image dimensions.

Each JSONL line has structure:
  data[0].content[0].image.{relative_path, width, height}

Images are resolved relative to an --images-dir (default: <jsonl_dir>/../images).

Usage:
  python fix_image_dimensions.py <jsonl_file_or_glob> [--images-dir PATH] [--inplace] [--suffix SUFFIX]

Examples:
  # Preview changes (dry-run, default):
  python fix_image_dimensions.py data_000000.jsonl

  # Fix in-place:
  python fix_image_dimensions.py data_000000.jsonl --inplace

  # Fix in-place with .bak backup:
  python fix_image_dimensions.py data_000000.jsonl --inplace --suffix .bak

  # Glob pattern:
  python fix_image_dimensions.py "C:/j00934267/.../jsonl_filtered/*.jsonl" --inplace
"""

import argparse
import glob
import json
import os
import shutil
import sys
from pathlib import Path

from PIL import Image


def get_actual_size(image_path: Path) -> tuple[int, int] | None:
    """Return (width, height) of image, or None if file not found/unreadable."""
    if not image_path.exists():
        return None
    try:
        with Image.open(image_path) as img:
            return img.width, img.height
    except Exception as e:
        print(f"  [WARN] Cannot read {image_path}: {e}", file=sys.stderr)
        return None


def fix_jsonl(jsonl_path: Path, images_dir: Path, inplace: bool, backup_suffix: str | None) -> dict:
    stats = {"total": 0, "fixed": 0, "missing": 0, "unchanged": 0}

    output_lines = []
    with open(jsonl_path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            raw = raw.rstrip("\n")
            if not raw.strip():
                output_lines.append(raw)
                continue

            stats["total"] += 1
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"  [WARN] Line {lineno}: JSON parse error: {e}", file=sys.stderr)
                output_lines.append(raw)
                continue

            # Navigate to image node
            try:
                image_node = obj["data"][0]["content"][0]["image"]
                rel_path = image_node["relative_path"]
            except (KeyError, IndexError, TypeError):
                output_lines.append(raw)
                continue

            image_path = images_dir / rel_path
            actual = get_actual_size(image_path)

            if actual is None:
                stats["missing"] += 1
                print(f"  [MISS] Line {lineno}: {rel_path}", file=sys.stderr)
                output_lines.append(raw)
                continue

            actual_w, actual_h = actual
            old_w = image_node.get("width")
            old_h = image_node.get("height")

            if old_w == actual_w and old_h == actual_h:
                stats["unchanged"] += 1
                output_lines.append(raw)
                continue

            image_node["width"] = actual_w
            image_node["height"] = actual_h
            stats["fixed"] += 1
            print(f"  [FIX]  Line {lineno}: {rel_path}  {old_w}x{old_h} -> {actual_w}x{actual_h}")
            output_lines.append(json.dumps(obj, ensure_ascii=False))

    if inplace:
        if backup_suffix:
            shutil.copy2(jsonl_path, str(jsonl_path) + backup_suffix)
        with open(jsonl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines))
            if output_lines:
                f.write("\n")
    else:
        # Dry-run: write to stdout summary only (changes already printed above)
        pass

    return stats


def main():
    parser = argparse.ArgumentParser(description="Fix image width/height in JSONL files.")
    parser.add_argument("pattern", help="JSONL file path or glob pattern")
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Base directory for resolving relative_path. "
             "Default: <jsonl_dir>/../images",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Write fixes back to the original file (default: dry-run)",
    )
    parser.add_argument(
        "--suffix",
        default=None,
        metavar="SUFFIX",
        help="Backup suffix when --inplace is set (e.g. .bak)",
    )
    args = parser.parse_args()

    files = sorted(glob.glob(args.pattern, recursive=True))
    if not files:
        print(f"No files matched: {args.pattern}", file=sys.stderr)
        sys.exit(1)

    mode = "IN-PLACE" if args.inplace else "DRY-RUN"
    print(f"Mode: {mode}  |  Files: {len(files)}\n")

    total_stats = {"total": 0, "fixed": 0, "missing": 0, "unchanged": 0}

    for fpath in files:
        jsonl_path = Path(fpath)
        if args.images_dir:
            images_dir = Path(args.images_dir)
        else:
            images_dir = jsonl_path.parent.parent / "images"

        print(f"Processing: {jsonl_path}")
        print(f"Images dir: {images_dir}")

        if not images_dir.exists():
            print(f"  [ERROR] Images directory not found: {images_dir}", file=sys.stderr)
            continue

        stats = fix_jsonl(jsonl_path, images_dir, args.inplace, args.suffix)
        for k in total_stats:
            total_stats[k] += stats[k]

        print(
            f"  => total={stats['total']}  fixed={stats['fixed']}  "
            f"unchanged={stats['unchanged']}  missing={stats['missing']}\n"
        )

    print(
        f"Grand total: total={total_stats['total']}  fixed={total_stats['fixed']}  "
        f"unchanged={total_stats['unchanged']}  missing={total_stats['missing']}"
    )
    if not args.inplace:
        print("\n[DRY-RUN] No files were modified. Re-run with --inplace to apply changes.")


if __name__ == "__main__":
    main()
