"""
Run Python code from each JSONL record and save the generated image to:
    <output_root>/<relative_path>

Usage:
    python run_code_save_images.py <jsonl_path> <output_root> [options]

Options:
    --workers N     Number of parallel workers (default: 1)
    --timeout N     Seconds before killing a single run (default: 30)
    --skip-existing Skip records whose output image already exists
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
# Matches the first positional argument (filename) of plt.savefig(...)
_SAVEFIG_RE = re.compile(
    r"(plt\.savefig\s*\(\s*)['\"]([^'\"]+)['\"]",
    re.MULTILINE,
)


def extract_code(assistant_content: list) -> str | None:
    """Return the first Python code block found in the assistant content list."""
    for item in assistant_content:
        if item.get("type") != "text":
            continue
        raw = item["text"]
        if isinstance(raw, dict):
            raw = raw.get("string", "")
        m = _CODE_FENCE_RE.search(raw)
        if m:
            return m.group(1)
    return None


def extract_relative_path(user_content: list) -> str | None:
    """Return the relative_path from the user image field."""
    for item in user_content:
        if item.get("type") != "image":
            continue
        img = item.get("image", {})
        rp = img.get("relative_path")
        if rp:
            return rp
    return None


def patch_savefig(code: str, output_path: str) -> str:
    """Replace the filename in the first plt.savefig() call with output_path."""
    escaped = output_path.replace("\\", "\\\\")
    patched, n = _SAVEFIG_RE.subn(
        lambda m: f"{m.group(1)}{repr(output_path)}",
        code,
        count=1,
    )
    if n == 0:
        # No savefig found — append one before any plt.show() / plt.close() / plt.clf()
        patched = re.sub(
            r"(plt\.(show|close|clf)\s*\()",
            f"plt.savefig({repr(output_path)}, bbox_inches='tight')\n\\1",
            code,
            count=1,
        )
        if patched == code:
            patched = code.rstrip() + f"\nplt.savefig({repr(output_path)}, bbox_inches='tight')\nplt.close()\n"
    return patched


def run_record(args):
    index, code, output_path, timeout = args
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            timeout=timeout,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return index, False, result.stderr.strip()
        if not os.path.exists(output_path):
            return index, False, "code ran but image was not created"
        return index, True, ""
    except subprocess.TimeoutExpired:
        return index, False, f"timeout after {timeout}s"
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run matplotlib code from JSONL and save images.")
    parser.add_argument("jsonl_path", help="Path to the .jsonl file")
    parser.add_argument("output_root", help="Root directory for saving images")
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker count")
    parser.add_argument("--timeout", type=int, default=30, help="Per-record timeout (seconds)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip already-saved images")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl_path)
    output_root = Path(args.output_root)

    if not jsonl_path.exists():
        sys.exit(f"JSONL file not found: {jsonl_path}")

    tasks = []
    skipped = 0

    with open(jsonl_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] line {i+1}: JSON parse error — {e}")
                continue

            data = record.get("data", [])
            user_content = next(
                (turn["content"] for turn in data if turn.get("role") == "user"), []
            )
            assistant_content = next(
                (turn["content"] for turn in data if turn.get("role") == "assistant"), []
            )

            relative_path = extract_relative_path(user_content)
            code = extract_code(assistant_content)

            if not relative_path:
                print(f"[WARN] line {i+1}: no relative_path found, skipping")
                continue
            if not code:
                print(f"[WARN] line {i+1}: no Python code block found, skipping")
                continue

            output_path = str(output_root / Path(relative_path))

            if args.skip_existing and os.path.exists(output_path):
                skipped += 1
                continue

            patched_code = patch_savefig(code, output_path)
            tasks.append((i + 1, patched_code, output_path, args.timeout))

    print(f"Records to process: {len(tasks)}  (skipped existing: {skipped})")

    ok_count = 0
    fail_count = 0

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_record, t): t[0] for t in tasks}
            for future in as_completed(futures):
                idx, success, msg = future.result()
                if success:
                    ok_count += 1
                else:
                    fail_count += 1
                    print(f"[FAIL] line {idx}: {msg}")
    else:
        for task in tasks:
            idx, success, msg = run_record(task)
            if success:
                ok_count += 1
            else:
                fail_count += 1
                print(f"[FAIL] line {idx}: {msg}")

    print(f"\nDone — success: {ok_count}, failed: {fail_count}")


if __name__ == "__main__":
    main()
