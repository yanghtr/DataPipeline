#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据 parse_render_log.py 生成的 issues.json，从原数据 JSON/JSONL 中剔除有问题的 id，
将过滤后的副本写入指定输出目录。原数据文件不做任何修改。

用法:
    # 单个文件（JSON 数组或 JSONL 均可）
    python filter_by_issues.py --issues issues.json --input data/foo.jsonl \\
        --output data_clean/ --json_dir /data/json --id_field record_uid

    # 整个目录
    python filter_by_issues.py --issues issues.json --input data/ \\
        --output data_clean/ --json_dir /data/json --id_field record_uid

过滤模式 (--mode):
    all    (默认) 剔除所有在 issues.json 中出现的 id（含 WARN 和 ERROR）
    error  仅剔除有 ERROR 的 id
    warn   仅剔除有 WARN 的 id

--json_dir 说明:
    渲染时若文件位于子目录（如 subdir/output.jsonl），render_html.py 会将输出目录名
    记为 subdir_output。过滤时若不传 --json_dir，则以文件 stem（output）查找，会匹配不上。
    传入与渲染时相同的 --json_dir，脚本会自动计算正确的 stem。
"""

import json
import re
import random
import argparse
import sys
from pathlib import Path
from collections import defaultdict

try:
    from PIL import Image as _PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

# 与 render_html.py 保持一致：用于将任意 ID 转成合法文件名
_SANITIZE_RE = re.compile(r'[^\w\-.]')


def _sanitize_id(raw_id) -> str:
    return _SANITIZE_RE.sub('_', str(raw_id))[:200]


def _output_stem(src: Path, json_dir: Path | None) -> str:
    """
    计算与 render_html.py process_file() 相同的输出目录名（stem）。
    若 json_dir 未提供，退化为 src.stem。
    """
    if json_dir is None:
        return src.stem
    try:
        rel = src.relative_to(json_dir)
    except ValueError:
        return src.stem
    parts = list(rel.with_suffix("").parts)
    return "_".join(parts)


class _PangumlWriter:
    """
    将过滤后保留的记录逐条转为 panguml 格式并写入 JSONL 文件。
    user turn  = 随机 instruction 文字 + 渲染截图（image item）
    assistant  = 原始 HTML 代码（text item）
    """

    def __init__(self, path: str, images_dir: str, image_root: str,
                 html_field: str, id_field: str,
                 templates: list[str] | None = None) -> None:
        self._path = path
        self._f = open(path, "a", encoding="utf-8")
        self.images_dir = Path(images_dir)
        self.image_root = Path(image_root)
        self.html_field = html_field
        self.id_field = id_field
        self.templates = templates or []
        self.written = 0
        self.skipped = 0

    def emit(self, record: dict, sanitized_id: str, stem: str) -> None:
        png_path = self.images_dir / stem / f"{sanitized_id}.png"
        if not png_path.exists():
            self.skipped += 1
            return

        html = record.get(self.html_field, "")
        if not html:
            self.skipped += 1
            return

        try:
            rel_path = str(png_path.relative_to(self.image_root))
        except ValueError:
            rel_path = f"{stem}/{sanitized_id}.png"

        width, height = 0, 0
        if _PIL_AVAILABLE:
            try:
                with _PILImage.open(png_path) as img:
                    width, height = img.size
            except Exception:
                pass

        # user content: instruction text + screenshot image
        user_content: list[dict] = []
        if self.templates:
            instruction = random.choice(self.templates)
            user_content.append({
                "type": "text",
                "text": {
                    "type": "string",
                    "format": "utf-8",
                    "string": instruction,
                },
            })
        user_content.append({
            "type": "image",
            "image": {
                "type": "relative_path",
                "format": "image/png",
                "relative_path": rel_path,
                "width": width,
                "height": height,
            },
        })

        sample = {
            "meta_prompt": [""],
            "data": [
                {
                    "role": "user",
                    "content": user_content,
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": {
                                "type": "string",
                                "format": "utf-8",
                                "string": html,
                            },
                        }
                    ],
                },
            ],
        }
        self._f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        self.written += 1

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "_PangumlWriter":
        return self

    def __exit__(self, *_) -> None:
        self.close()


def load_issues(issues_path: str, mode: str) -> dict[str, set[str]]:
    """
    读取 issues.json，返回 {source_file_stem: set(item_id)} 映射。
    mode: 'all' | 'error' | 'warn'
    """
    with open(issues_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    ids_section: dict = report.get("ids", {})
    result: dict[str, set[str]] = defaultdict(set)

    for log_id, rec in ids_section.items():
        if mode == "error" and not rec.get("has_error", False):
            continue
        if mode == "warn" and not rec.get("has_warn", False):
            continue
        source_file = rec.get("source_file", "")
        item_id = rec.get("item_id", "")
        if source_file and item_id:
            result[source_file].add(item_id)

    return dict(result)


def collect_input_files(input_path: str) -> list[Path]:
    """接受单个文件或目录，返回所有待处理的 JSON/JSONL 文件路径列表。"""
    p = Path(input_path)
    if p.is_file():
        return [p]
    if p.is_dir():
        files = sorted(
            list(p.glob("*.json")) + list(p.glob("*.jsonl"))
        )
        if not files:
            print(f"[WARN] 目录中未找到 .json/.jsonl 文件: {p}", file=sys.stderr)
        return files
    print(f"[ERROR] 输入路径不存在: {p}", file=sys.stderr)
    sys.exit(1)


# ── JSON 数组过滤（保留原格式/编码/缩进） ───────────────────────────────────

def _detect_format(raw: bytes) -> tuple[str, bool, str, tuple[str, str]]:
    """探测 JSON 数组文件的编码/BOM/缩进/分隔符，dump 时尽量保持一致。"""
    if raw.startswith(b"\xef\xbb\xbf"):
        encoding, has_bom, body = "utf-8", True, raw[3:]
    else:
        encoding, has_bom, body = "utf-8", False, raw

    text = body.decode(encoding, errors="replace")
    lb = text.find("[")
    indent = ""
    item_sep = ","
    key_sep = ": "
    if lb != -1:
        i = lb + 1
        leading = []
        while i < len(text) and text[i] in " \t\r\n":
            leading.append(text[i])
            i += 1
        leading_s = "".join(leading)
        if "\n" in leading_s:
            after_nl = leading_s.rsplit("\n", 1)[-1]
            indent = after_nl if after_nl else "  "
        m = re.search(r'"\s*:\s*', text[lb:lb + 4000])
        if m:
            key_sep = m.group(0).split('"', 1)[-1]
            key_sep = ": " if " " in key_sep else ":"
    if indent == "":
        item_sep = ", " if re.search(r",\s", text[:4000]) else ","
    else:
        item_sep = ","
    return encoding, has_bom, indent, (item_sep, key_sep)


def _filter_json_array(src: Path, out_dir: Path,
                       blocked: set[str], id_field: str,
                       panguml_writer: "_PangumlWriter | None" = None,
                       stem: str | None = None) -> tuple[int, int]:
    """JSON 数组：剔除 blocked 中的 id，输出到 out_dir，保持原格式。"""
    raw = src.read_bytes()
    encoding, has_bom, indent, separators = _detect_format(raw)
    body = raw[3:] if has_bom else raw
    data = json.loads(body.decode(encoding))

    if not isinstance(data, list):
        print(f"[SKIP] {src.name}: 顶层不是数组，跳过", file=sys.stderr)
        return 0, 0

    original_count = len(data)
    filtered = []
    for item in data:
        item_id = _sanitize_id(item.get(id_field, item.get("id", "")))
        if blocked and item_id in blocked:
            pass
        else:
            filtered.append(item)
            if panguml_writer is not None and stem:
                panguml_writer.emit(item, item_id, stem)

    removed = original_count - len(filtered)

    out_path = out_dir / src.name
    dump_kwargs: dict = {"ensure_ascii": False, "separators": separators}
    if indent:
        dump_kwargs["indent"] = indent
        dump_kwargs["separators"] = (",", separators[1])

    with open(out_path, "wb") as f:
        if has_bom:
            f.write(b"\xef\xbb\xbf")
        f.write(json.dumps(filtered, **dump_kwargs).encode(encoding))
        if raw.endswith(b"\n"):
            f.write(b"\n")

    return original_count, removed


# ── JSONL 过滤（逐行处理，保留换行符） ──────────────────────────────────────

def _filter_jsonl(src: Path, out_dir: Path,
                  blocked: set[str], id_field: str,
                  panguml_writer: "_PangumlWriter | None" = None,
                  stem: str | None = None) -> tuple[int, int]:
    """JSONL：逐行读取，剔除 blocked 中的 id，输出到 out_dir。"""
    out_path = out_dir / src.name
    original_count = 0
    removed = 0
    with open(src, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            stripped = line.rstrip("\n\r")
            if not stripped:
                fout.write(line)
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError:
                fout.write(line)
                continue
            original_count += 1
            raw_id = item.get(id_field, item.get("id", ""))
            item_id = _sanitize_id(raw_id)
            if blocked and item_id in blocked:
                removed += 1
            else:
                fout.write(line)
                if panguml_writer is not None and stem:
                    panguml_writer.emit(item, item_id, stem)
    return original_count, removed


# ── 分发 ─────────────────────────────────────────────────────────────────────

def filter_file(src: Path, out_dir: Path,
                blocked: set[str], id_field: str,
                panguml_writer: "_PangumlWriter | None" = None,
                stem: str | None = None) -> tuple[int, int]:
    if src.suffix.lower() == ".jsonl":
        return _filter_jsonl(src, out_dir, blocked, id_field, panguml_writer, stem)
    return _filter_json_array(src, out_dir, blocked, id_field, panguml_writer, stem)


def _generate_filter_charts(
    file_results: list[tuple],   # [(name, original, removed, had_blocked)]
    stats_dir: Path,
) -> None:
    """
    生成堆叠柱状图：每列 = 保留（绿）+ 剔除（红），末列 = 全局总计（右轴独立刻度）。
    需要 matplotlib。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[STATS] matplotlib 未安装，跳过图表生成")
        return

    stats_dir.mkdir(parents=True, exist_ok=True)

    # 所有有数据的文件（含直通文件，让读者看到完整比例）
    all_data = [(n, o, r) for n, o, r, _hb in file_results if o > 0]
    if not all_data:
        return

    names     = [n for n, _, _ in all_data]
    originals = [o for _, o, _ in all_data]
    removed   = [r for _, _, r in all_data]
    kept      = [o - r for o, r in zip(originals, removed)]

    total_orig  = sum(originals)
    total_rm    = sum(removed)
    total_kept  = total_orig - total_rm
    total_pct   = round(total_rm / max(total_orig, 1) * 100, 1)

    n = len(all_data)
    xs = list(range(n))
    x_total = n          # 总计柱位置
    W = 0.65             # 柱宽

    fig, ax1 = plt.subplots(figsize=(max(7, (n + 2) * 1.5), 6))
    ax2 = ax1.twinx()    # 右轴：总计柱独立刻度

    # ── 每文件堆叠柱（左轴）──────────────────────────────────
    y1_max = max(originals) * 1.30
    ax1.set_ylim(0, y1_max)
    bk = ax1.bar(xs, kept,    color="#43A047", edgecolor="white", width=W, label="kept")
    br = ax1.bar(xs, removed, bottom=kept,
                 color="#E53935", edgecolor="white", width=W, label="removed")
    ax1.set_ylabel("Count (per file)", fontsize=10)

    # ── 总计柱（右轴）────────────────────────────────────────
    y2_max = total_orig * 1.30
    ax2.set_ylim(0, y2_max)
    ax2.bar(x_total, total_kept, color="#43A047", alpha=0.80, edgecolor="white", width=W)
    ax2.bar(x_total, total_rm,   bottom=total_kept,
            color="#E53935", alpha=0.80, edgecolor="white", width=W)
    ax2.set_ylabel("Count (total, right axis)", fontsize=10, color="#888888")
    ax2.tick_params(axis="y", labelcolor="#888888")

    # ── 每文件百分比标注 ──────────────────────────────────────
    for i, (o, r) in enumerate(zip(originals, removed)):
        pct = round(r / max(o, 1) * 100, 1)
        ax1.text(xs[i], o + y1_max * 0.012, f"{pct:.1f}%",
                 ha="center", va="bottom", fontsize=7, color="#B71C1C")

    # ── 总计标注 ──────────────────────────────────────────────
    ax2.text(x_total, total_orig + y2_max * 0.012,
             f"{total_pct:.1f}%\n({total_rm:,}/{total_orig:,})",
             ha="center", va="bottom", fontsize=8, color="#B71C1C", fontweight="bold")

    # ── X 轴标签 + 分隔线 ────────────────────────────────────
    ax1.set_xticks(xs + [x_total])
    ax1.set_xticklabels(names + ["[total]"], rotation=40, ha="right", fontsize=8)
    ax1.set_xlim(-0.6, n + 0.6)
    ax1.axvline(x=n - 0.5, color="#AAAAAA", linestyle="--", linewidth=1, alpha=0.7)

    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_title("Filter results: kept / removed  (last col = total, right axis)", fontsize=11)
    plt.tight_layout()

    out = stats_dir / "drop_ratio.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[STATS] 图表已保存: {out}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="根据 issues.json 过滤原数据 JSON/JSONL，输出副本到指定目录"
    )
    parser.add_argument("--issues", required=True,
                        help="parse_render_log.py 生成的 issues.json 路径")
    parser.add_argument("--input", required=True,
                        help="原数据文件路径，或包含多个 JSON/JSONL 文件的目录")
    parser.add_argument("--output", required=True,
                        help="过滤后副本的输出目录（不存在则自动创建）")
    parser.add_argument("--level", choices=["all", "error", "warn"], default="all",
                        help="按日志级别过滤：all=ERROR+WARN 都剔除, error=仅剔除有ERROR的id, "
                             "warn=仅剔除有WARN的id（默认 all）")
    parser.add_argument("--json_dir", default=None,
                        help="与渲染时相同的 --json_dir，用于正确计算 source_file stem；"
                             "文件在子目录时必填")
    parser.add_argument("--id_field", default="id",
                        help="原数据中的 ID 字段名，需与渲染时的 --id_field 一致（默认 id）")
    # panguml 导出
    parser.add_argument("--export_panguml", default=None,
                        help="将过滤后保留的记录导出为 panguml 格式 JSONL（指定输出文件路径）")
    parser.add_argument("--images_dir", default=None,
                        help="渲染截图根目录（export_panguml 时必填）")
    parser.add_argument("--image_root", default=None,
                        help="panguml relative_path 的参考根目录（默认等于 --images_dir）")
    parser.add_argument("--html_field", default="html",
                        help="原数据中 HTML 内容所在字段名（默认 html，用于 panguml assistant 内容）")
    parser.add_argument("--templates_file", default=None,
                        help="instruction 模板文件路径（每行一条，panguml user turn 随机采样；"
                             "不指定则 user turn 只含截图）")
    # 统计与图表
    parser.add_argument("--summary_json", default=None,
                        help="将本次过滤统计写入 JSON 文件（供外层脚本汇总）")
    parser.add_argument("--stats_dir", default=None,
                        help="生成统计图表并保存到此目录（需要 matplotlib）")
    args = parser.parse_args(argv)

    json_dir = Path(args.json_dir) if args.json_dir else None

    if args.export_panguml and not args.images_dir:
        parser.error("--export_panguml 需要同时指定 --images_dir")

    templates: list[str] = []
    if args.templates_file:
        try:
            with open(args.templates_file, "r", encoding="utf-8") as f:
                templates = [ln.strip() for ln in f if ln.strip()]
            print(f"[INFO] 已加载 {len(templates)} 条 instruction 模板: {args.templates_file}")
        except Exception as e:
            print(f"[WARN] 无法读取 templates_file: {e}", file=sys.stderr)

    blocked_by_file = load_issues(args.issues, args.level)
    total_blocked_ids = sum(len(s) for s in blocked_by_file.values())
    print(f"[LOAD] issues: {total_blocked_ids} 个待剔除 id  ({len(blocked_by_file)} 个 stem)  "
          f"level={args.level}")

    input_files = collect_input_files(args.input)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    panguml_writer: "_PangumlWriter | None" = None
    if args.export_panguml:
        image_root = args.image_root or args.images_dir
        panguml_writer = _PangumlWriter(
            args.export_panguml,
            args.images_dir,
            image_root,
            args.html_field,
            args.id_field,
            templates=templates,
        )

    # (name, original, removed, had_blocked)
    file_results: list[tuple[str, int, int, bool]] = []

    try:
        for src in input_files:
            stem = _output_stem(src, json_dir)
            blocked = blocked_by_file.get(stem, set())
            original, removed = filter_file(
                src, out_dir, blocked, args.id_field,
                panguml_writer=panguml_writer, stem=stem,
            )
            had_blocked = bool(blocked)
            file_results.append((src.name, original, removed, had_blocked))

            if not had_blocked:
                print(f"  {src.name:<45}  [直通]  {original} 条")
            else:
                pct = round(removed / max(original, 1) * 100, 1)
                kept = original - removed
                print(f"  {src.name:<45}  剔除 {removed}/{original} ({pct}%)  保留 {kept}")
    finally:
        if panguml_writer is not None:
            panguml_writer.close()

    # ── 分拆统计 ─────────────────────────────────────────────
    filtered_results  = [(n, o, r) for n, o, r, hb in file_results if hb]
    passthru_results  = [(n, o, r) for n, o, r, hb in file_results if not hb]

    filt_original = sum(o for _, o, _ in filtered_results)
    filt_removed  = sum(r for _, _, r in filtered_results)
    filt_kept     = filt_original - filt_removed
    pass_items    = sum(o for _, o, _ in passthru_results)

    # ── SUMMARY ──────────────────────────────────────────────
    pct_total = round(filt_removed / max(filt_original, 1) * 100, 1) if filt_original else 0.0
    print(f"\n[SUMMARY]")
    if filtered_results:
        names_str = ", ".join(n for n, _, _ in filtered_results)
        print(f"  过滤文件      : {len(filtered_results)} 个  ({names_str})")
    else:
        print(f"  过滤文件      : 0 个")
    print(f"  原始条目数    : {filt_original}")
    print(f"  剔除条目数    : {filt_removed} ({pct_total}%)")
    print(f"  保留条目数    : {filt_kept}")
    if passthru_results:
        pass_names = ", ".join(n for n, _, _ in passthru_results[:3])
        if len(passthru_results) > 3:
            pass_names += f" 等 {len(passthru_results)} 个"
        print(f"  直通文件      : {len(passthru_results)} 个  {pass_items} 条  ({pass_names})")
    print(f"  输出目录      : {out_dir.resolve()}")
    if panguml_writer is not None:
        print(f"  panguml 输出  : {Path(args.export_panguml).resolve()}")
        skipped_note = (f"  跳过: {panguml_writer.skipped} 条（PNG 不存在或 HTML 为空）"
                        if panguml_writer.skipped else "")
        print(f"  panguml 写入  : {panguml_writer.written} 条{skipped_note}")

    # ── 写出机器可读汇总（含 per-file 明细，供外层聚合成全局图表）──────
    if args.summary_json:
        summary_data = {
            "files": [
                {"name": n, "original": o, "removed": r, "had_blocked": hb}
                for n, o, r, hb in file_results
            ],
            "original":          filt_original,
            "removed":           filt_removed,
            "kept":              filt_kept,
            "passthrough_items": pass_items,
            "panguml_written":   panguml_writer.written if panguml_writer else 0,
            "panguml_skipped":   panguml_writer.skipped if panguml_writer else 0,
        }
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, ensure_ascii=False)

    # ── 单独调用时（不经 run.sh）生成本次子集的图表 ────────────────
    if args.stats_dir:
        _generate_filter_charts(file_results, Path(args.stats_dir))


if __name__ == "__main__":
    main()
