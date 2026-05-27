import argparse
import base64
import html
import io
import os
import random
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


MAX_INPUT_ROWS = 12
SAMPLE_NUM = -1

APP_CSS = """
.table-wrap {
  width: 100%;
  overflow: visible;
  border: 1px solid #d9d9d9;
  border-radius: 8px;
  background: #fff;
}
table {
  border-collapse: collapse;
  width: 100%;
  table-layout: fixed;
}
th, td {
  border: 1px solid #e8e8e8;
  vertical-align: top;
  padding: 8px;
  background: #fff;
  word-break: break-word;
}
.label-col {
  width: 180px;
  background: #fafafa;
  font-weight: 600;
}
.folder-title {
  font-weight: 700;
}
.folder-path {
  color: #666;
  font-size: 12px;
  word-break: break-all;
  margin-top: 4px;
}
.cell-image img {
  width: 100%;
  height: auto;
  max-height: 56vh;
  object-fit: contain;
  border: 1px solid #d4d4d4;
  border-radius: 6px;
  background: #f7f7f7;
}
.cell-html pre {
  margin: 0;
  background: #f7f7f7;
  border: 1px solid #e0e0e0;
  border-radius: 6px;
  padding: 8px;
  white-space: pre-wrap;
  word-break: break-all;
  font-size: 12px;
  line-height: 1.45;
}
.placeholder {
  border: 1px dashed #bbb;
  border-radius: 6px;
  padding: 24px 12px;
  text-align: center;
  color: #666;
  background: #fafafa;
  min-height: 220px;
  display: flex;
  align-items: center;
  justify-content: center;
}
.placeholder.error {
  border-color: #f1aeb5;
  color: #a22;
  background: #fff5f6;
}
.hint {
  padding: 18px;
  border: 1px dashed #bbb;
  border-radius: 8px;
  color: #555;
  background: #fafafa;
}
"""

FolderScan = Dict[str, Dict[str, Optional[str]]]


def normalize_folder_path(raw_path: str) -> Tuple[Optional[str], Optional[str]]:
    candidate = (raw_path or "").strip().strip('"').strip("'")
    if not candidate:
        return None, "empty path"
    resolved = os.path.abspath(os.path.expanduser(candidate))
    if not os.path.exists(resolved):
        return None, f"path does not exist: {resolved}"
    if not os.path.isdir(resolved):
        return None, f"not a folder: {resolved}"
    return resolved, None


def scan_set_folder(folder: str) -> FolderScan:
    png_map: Dict[str, str] = {}
    html_map_exact: Dict[str, str] = {}
    html_map_lower: Dict[str, str] = {}

    with os.scandir(folder) as entries:
        for entry in entries:
            if not entry.is_file():
                continue
            stem, ext = os.path.splitext(entry.name)
            ext = ext.lower()
            if ext == ".png" and stem not in png_map:
                png_map[stem] = entry.path
            elif ext == ".html" or ext == ".svg":
                if stem not in html_map_exact:
                    html_map_exact[stem] = entry.path
                lower_stem = stem.lower()
                if lower_stem not in html_map_lower:
                    html_map_lower[lower_stem] = entry.path

    result: FolderScan = {}
    for stem in sorted(png_map.keys(), key=lambda s: s.lower()):
        html_path = html_map_exact.get(stem) or html_map_lower.get(stem.lower())
        result[stem] = {"png_path": png_map[stem], "html_path": html_path}
    return result


def build_union_basenames(scans: Dict[str, FolderScan]) -> List[str]:
    names = set()
    for folder_data in scans.values():
        names.update(folder_data.keys())
    return sorted(names, key=lambda s: s.lower())


def apply_sampling(basenames: List[str], sample_num: int) -> List[str]:
    if sample_num < 0:
        return basenames
    if sample_num == 0:
        return []
    if sample_num >= len(basenames):
        return basenames
    sampled = random.sample(basenames, sample_num)
    return sorted(sampled, key=lambda s: s.lower())


def clamp_index(index: int, total: int) -> int:
    if total <= 0:
        return 0
    return max(0, min(index, total - 1))


@lru_cache(maxsize=2048)
def _read_html_cached(path: str, mtime: float) -> str:
    del mtime
    encodings = ("utf-8", "utf-8-sig", "gb18030", "latin-1")
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def read_html_text(path: Optional[str]) -> str:
    if not path:
        return "missing"
    try:
        mtime = os.path.getmtime(path)
        return _read_html_cached(path, mtime)
    except Exception as exc:
        return f"[error reading html: {exc}]"


@lru_cache(maxsize=1024)
def _load_image_data_uri(path: str, mtime: float) -> Tuple[str, str]:
    del mtime
    if Image is None:
        return "", "Pillow is not installed, cannot validate png."
    try:
        with Image.open(path) as img:
            img.load()
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGBA")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}", ""
    except Exception as exc:
        return "", str(exc)


def get_image_data_uri(path: Optional[str]) -> Tuple[str, str]:
    if not path:
        return "", "missing png"
    try:
        mtime = os.path.getmtime(path)
    except Exception as exc:
        return "", str(exc)
    return _load_image_data_uri(path, mtime)


def render_folder_list_markdown(folders: List[str]) -> str:
    if not folders:
        return "### Added folders\n- (empty)"
    lines = ["### Added folders"]
    for idx, folder in enumerate(folders, start=1):
        lines.append(f"- {idx}. `{folder}`")
    return "\n".join(lines)


def render_grid_html(
    folders: List[str],
    scans: Dict[str, FolderScan],
    basenames: List[str],
    index: int,
) -> str:
    if not folders:
        return "<div class='hint'>Add at least one set folder and click Confirm.</div>"
    if not basenames:
        return "<div class='hint'>No PNG files found in current folders.</div>"

    idx = clamp_index(index, len(basenames))
    base_name = basenames[idx]

    header_cells = ["<th class='label-col'>row type</th>"]
    image_row_cells = [
        "<td class='label-col'>"
        f"<div><code>{html.escape(base_name)}</code></div>"
        "<div>PNG</div>"
        "</td>"
    ]
    html_row_cells = ["<td class='label-col'>HTML</td>"]

    for folder in folders:
        folder_name = Path(folder).name or folder
        folder_name_safe = html.escape(folder_name)
        folder_safe = html.escape(folder)
        header_cells.append(
            "<th>"
            f"<div class='folder-title' title='{folder_safe}'>{folder_name_safe}</div>"
            f"<div class='folder-path'>{folder_safe}</div>"
            "</th>"
        )

        record = scans.get(folder, {}).get(base_name)
        if not record:
            image_block = "<div class='placeholder'>missing png</div>"
            html_text = "missing"
        else:
            img_uri, img_err = get_image_data_uri(record.get("png_path"))
            if img_uri:
                image_block = f"<img src='{img_uri}' alt='{html.escape(base_name)}' />"
            else:
                image_block = (
                    "<div class='placeholder error'>"
                    f"png open error: {html.escape(img_err)}"
                    "</div>"
                )
            html_text = read_html_text(record.get("html_path"))

        image_row_cells.append(
            "<td class='cell-image'>"
            f"{image_block}"
            "</td>"
        )
        html_row_cells.append(
            "<td class='cell-html'>"
            f"<pre>{html.escape(html_text)}</pre>"
            "</td>"
        )

    return (
        "<div class='table-wrap'>"
        "<table>"
        "<thead><tr>"
        + "".join(header_cells)
        + "</tr></thead>"
        "<tbody>"
        "<tr>"
        + "".join(image_row_cells)
        + "</tr>"
        "<tr>"
        + "".join(html_row_cells)
        + "</tr>"
        "</tbody>"
        "</table>"
        "</div>"
    )


def build_response(
    folders: List[str],
    scans: Dict[str, FolderScan],
    basenames: List[str],
    index: int,
    status: str = "",
):
    folders = list(folders or [])
    scans = dict(scans or {})
    basenames = list(basenames or [])
    index = clamp_index(int(index or 0), len(basenames))
    selected = basenames[index] if basenames else None

    folder_markdown = render_folder_list_markdown(folders)
    select_update = gr.update(choices=basenames, value=selected)

    if basenames:
        nav_text = f"Current: **{index + 1} / {len(basenames)}** | base name: `{selected}`"
    else:
        nav_text = "Current: **0 / 0**"

    grid_html = render_grid_html(folders, scans, basenames, index)
    if not status:
        status = "Ready."
        if folders:
            status = f"Loaded {len(folders)} folder(s)."

    return (
        folders,
        scans,
        basenames,
        index,
        folder_markdown,
        select_update,
        nav_text,
        grid_html,
        status,
    )


def collect_folders_from_inputs(
    path_values: List[str],
    active_rows: int,
) -> Tuple[List[str], Dict[str, FolderScan], List[str], str]:
    folders: List[str] = []
    scans: Dict[str, FolderScan] = {}
    errors: List[str] = []
    seen = set()

    for i in range(active_rows):
        raw_path = path_values[i] if i < len(path_values) else ""
        if not (raw_path or "").strip():
            continue

        folder, err = normalize_folder_path(raw_path)
        if err:
            errors.append(f"Row {i + 1}: {err}")
            continue

        folder_key = os.path.normcase(folder)
        if folder_key in seen:
            errors.append(f"Row {i + 1}: duplicate ignored: {folder}")
            continue

        try:
            folder_scan = scan_set_folder(folder)
        except Exception as exc:
            errors.append(f"Row {i + 1}: scan failed: {exc}")
            continue

        seen.add(folder_key)
        folders.append(folder)
        scans[folder] = folder_scan

    all_basenames = build_union_basenames(scans)
    sampled_basenames = apply_sampling(all_basenames, SAMPLE_NUM)

    if folders:
        status = f"Loaded {len(folders)} folder(s)."
    else:
        status = "No valid folder loaded."
    if SAMPLE_NUM >= 0:
        status += (
            f" Sampled {len(sampled_basenames)} / {len(all_basenames)} rows"
            f" (sample_num={SAMPLE_NUM})."
        )
    if errors:
        status += " " + " | ".join(errors)
    return folders, scans, sampled_basenames, status


def on_confirm_paths(row_count: int, index: int, *path_values: str):
    folders, scans, basenames, status = collect_folders_from_inputs(
        list(path_values), int(row_count)
    )
    return build_response(folders, scans, basenames, index, status)


def on_plus_row(row_count: int):
    current = int(row_count)
    new_count = min(MAX_INPUT_ROWS, current + 1)
    row_updates = [gr.update(visible=i < new_count) for i in range(MAX_INPUT_ROWS)]
    if new_count == current:
        status = f"Already at max input rows: {MAX_INPUT_ROWS}"
    else:
        status = f"Input rows: {new_count}"
    return (new_count, *row_updates, status)


def on_minus_row(row_count: int, index: int, *path_values: str):
    current = int(row_count)
    paths = list(path_values)
    if current <= 1:
        new_count = 1
        status_prefix = "At least one input row must remain."
    else:
        new_count = current - 1
        paths[current - 1] = ""
        status_prefix = f"Removed row {current}."

    row_updates = [gr.update(visible=i < new_count) for i in range(MAX_INPUT_ROWS)]
    input_updates = [gr.update(value=paths[i]) for i in range(MAX_INPUT_ROWS)]

    folders, scans, basenames, scan_status = collect_folders_from_inputs(paths, new_count)
    final_status = f"{status_prefix} {scan_status}".strip()
    main_updates = build_response(folders, scans, basenames, index, final_status)
    return (new_count, *row_updates, *input_updates, *main_updates)


def on_prev(
    folders: List[str],
    scans: Dict[str, FolderScan],
    basenames: List[str],
    index: int,
):
    return build_response(folders, scans, basenames, int(index) - 1)


def on_next(
    folders: List[str],
    scans: Dict[str, FolderScan],
    basenames: List[str],
    index: int,
):
    return build_response(folders, scans, basenames, int(index) + 1)


def on_select_basename(
    selected_name: Optional[str],
    folders: List[str],
    scans: Dict[str, FolderScan],
    basenames: List[str],
):
    if selected_name in basenames:
        new_index = basenames.index(selected_name)
    else:
        new_index = 0
    return build_response(folders, scans, basenames, new_index)


def parse_cli_args():
    parser = argparse.ArgumentParser(description="PNG + HTML alignment viewer")
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for Gradio server, e.g. 8060",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for Gradio server, default 127.0.0.1",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link",
    )
    parser.add_argument(
        "--inbrowser",
        action="store_true",
        help="Open app automatically in browser",
    )
    parser.add_argument(
        "--sample_num",
        type=int,
        default=-1,
        help=(
            "Random sample size of aligned rows. "
            "-1 means no sampling (use all rows)."
        ),
    )
    return parser.parse_args()


with gr.Blocks(title="PNG HTML Aligned Viewer") as demo:
    gr.HTML(f"<style>{APP_CSS}</style>")
    gr.Markdown("## PNG + HTML Alignment Viewer")
    gr.Markdown(
        "Input multiple set folder paths, click Confirm, then browse aligned rows by PNG base name union."
    )

    folders_state = gr.State([])
    scans_state = gr.State({})
    basenames_state = gr.State([])
    index_state = gr.State(0)
    row_count_state = gr.State(1)

    status_md = gr.Markdown("Ready. Input folder path(s), then click Confirm.")
    folder_list_md = gr.Markdown(render_folder_list_markdown([]))

    with gr.Row():
        plus_btn = gr.Button("+", min_width=80)
        minus_btn = gr.Button("-", min_width=80)
        confirm_btn = gr.Button("Confirm", variant="primary")

    folder_input_rows = []
    folder_inputs = []
    for i in range(MAX_INPUT_ROWS):
        with gr.Row(visible=(i == 0)) as row:
            text = gr.Textbox(
                label=f"Set folder path {i + 1}",
                placeholder=r"D:\path\to\set_folder",
            )
        folder_input_rows.append(row)
        folder_inputs.append(text)

    with gr.Row():
        prev_btn = gr.Button("Prev", min_width=90)
        next_btn = gr.Button("Next", min_width=90)
        basename_dd = gr.Dropdown(
            label="Select base name",
            choices=[],
            value=None,
            scale=5,
        )
        nav_info_md = gr.Markdown("Current: **0 / 0**")

    grid_html = gr.HTML(render_grid_html([], {}, [], 0))

    common_outputs = [
        folders_state,
        scans_state,
        basenames_state,
        index_state,
        folder_list_md,
        basename_dd,
        nav_info_md,
        grid_html,
        status_md,
    ]

    demo.load(
        fn=lambda folders, scans, basenames, index: build_response(
            folders,
            scans,
            basenames,
            index,
            "Ready. Input folder path(s), then click Confirm.",
        ),
        inputs=[folders_state, scans_state, basenames_state, index_state],
        outputs=common_outputs,
    )

    plus_btn.click(
        fn=on_plus_row,
        inputs=[row_count_state],
        outputs=[row_count_state, *folder_input_rows, status_md],
    )

    minus_btn.click(
        fn=on_minus_row,
        inputs=[row_count_state, index_state, *folder_inputs],
        outputs=[
            row_count_state,
            *folder_input_rows,
            *folder_inputs,
            *common_outputs,
        ],
    )

    confirm_btn.click(
        fn=on_confirm_paths,
        inputs=[row_count_state, index_state, *folder_inputs],
        outputs=common_outputs,
    )

    prev_btn.click(
        fn=on_prev,
        inputs=[folders_state, scans_state, basenames_state, index_state],
        outputs=common_outputs,
    )

    next_btn.click(
        fn=on_next,
        inputs=[folders_state, scans_state, basenames_state, index_state],
        outputs=common_outputs,
    )

    basename_dd.change(
        fn=on_select_basename,
        inputs=[basename_dd, folders_state, scans_state, basenames_state],
        outputs=common_outputs,
    )


if __name__ == "__main__":
    cli_args = parse_cli_args()
    SAMPLE_NUM = cli_args.sample_num
    demo.launch(
        server_name=cli_args.host,
        server_port=cli_args.port,
        share=cli_args.share,
        inbrowser=cli_args.inbrowser,
    )