#!/usr/bin/env bash
# HTML 渲染管线一键脚本
# 用法: bash run.sh [--workers N] [--level error|warn|all] [--tall_ratio F]
#                   [--no-resume]
#
# 三阶段 + 统计:
#   1. render  → PNG 截图（默认 resume：已有 PNG 自动跳过；日志追加）
#   2. parse   → 解析日志生成 $ISSUES_FILE
#   3. filter  → 根据 issues 剔除有问题的记录，写到 $OUTPUT_DATA_DIR
#   4. stats   → 图片尺寸分布 + 综合汇总（需 matplotlib + PIL）
#
# resume 行为（默认开启）:
#   - render: 已存在 PNG 的条目自动跳过（render_html.py 内置）
#   - log:    追加到 $LOG_FILE，保留历史 WARN/ERROR
#   - --no-resume: 清空日志文件并从头记录（PNG 不会被删除）
set -euo pipefail

SEP="────────────────────────────────────────────────────────────"

# ── 路径配置 ──────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_DIR="/home/yanghaitao/Projects/Data/FineWebEdu/run/20260324_0_20"

# 待渲染的文件（相对 DATA_DIR 的路径）
INPUT_FILES=(
    "part2026-03-23-00000_01/output.jsonl"
    "part2026-03-23-00000_02/output.jsonl"
)

# 输出目录
IMAGES_DIR="$DATA_DIR/images"
LOG_FILE="$IMAGES_DIR/render.log"
ISSUES_FILE="$IMAGES_DIR/issues.json"
OUTPUT_DATA_DIR="$DATA_DIR/filtered"   # filter 阶段输出
PANGUML_FILE="$DATA_DIR/jsonl/data_000000.jsonl" # panguml 训练数据输出
STATS_DIR="$DATA_DIR/stats"            # 统计图表输出

# instruction 模板文件（panguml user turn 随机采样）
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TEMPLATES_FILE="$SCRIPT_ROOT/converters/templates/image_to_html_templates.txt"

# HTML / ID 字段名（针对当前数据集）
HTML_FIELD="output_html"
ID_FIELD="record_uid"

# ── 可调参数 ──────────────────────────────────────────────
WORKERS="${WORKERS:-$(( $(nproc) - 1 ))}"
WORKERS=$(( WORKERS < 1 ? 1 : WORKERS ))

FILTER_LEVEL="${FILTER_LEVEL:-all}"     # error | warn | all

# tall_ratio: 截图高宽比超过此值时记录 WARN/TALL_PAGE（0 禁用）
TALL_RATIO_THRESHOLD="${TALL_RATIO_THRESHOLD:-4.0}"

# resume: true=追加日志续跑（默认），false=清空日志重新开始
RESUME="${RESUME:-true}"

# ── 参数解析 ──────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --workers)     WORKERS="$2"; shift 2 ;;
        --level)       FILTER_LEVEL="$2"; shift 2 ;;
        --tall_ratio)  TALL_RATIO_THRESHOLD="$2"; shift 2 ;;
        --no-resume)   RESUME="false"; shift ;;
        *) echo "[ERROR] 未知参数: $1"; exit 1 ;;
    esac
done

# ── 准备 ─────────────────────────────────────────────────
mkdir -p "$IMAGES_DIR" "$STATS_DIR"
START_TS=$(date '+%Y-%m-%d %H:%M:%S')

echo "$SEP"
echo "  HTML 渲染管线  |  $START_TS"
echo "$SEP"
printf "  数据目录      : %s\n" "$DATA_DIR"
printf "  截图目录      : %s\n" "$IMAGES_DIR"
printf "  日志文件      : %s\n" "$LOG_FILE"
printf "  并行进程      : %s\n" "$WORKERS"
printf "  HTML 字段     : %s\n" "$HTML_FIELD"
printf "  ID 字段       : %s\n" "$ID_FIELD"
printf "  过滤级别      : %s\n" "$FILTER_LEVEL"
printf "  tall_ratio    : %s\n" "$TALL_RATIO_THRESHOLD"
printf "  resume 模式   : %s\n" "$RESUME"
printf "  panguml       : %s\n" "$PANGUML_FILE"
printf "  统计目录      : %s\n" "$STATS_DIR"
printf "  输入文件      :\n"
for f in "${INPUT_FILES[@]}"; do
    printf "    %s\n" "$f"
done
echo "$SEP"

# ── 阶段 1: 渲染 ─────────────────────────────────────────
echo ""
echo "┌─ STEP 1/3  渲染 HTML → PNG"
echo "│"

if [[ "$RESUME" == "true" ]]; then
    echo "│  [resume] 日志追加，已有 PNG 自动跳过"
else
    echo "│  [fresh]  清空日志，重新记录"
    : > "$LOG_FILE"
fi

python "$SCRIPT_DIR/html_render.py" render \
    --json_dir            "$DATA_DIR" \
    --files               "${INPUT_FILES[@]}" \
    --images_dir          "$IMAGES_DIR" \
    --html_field          "$HTML_FIELD" \
    --id_field            "$ID_FIELD" \
    --workers             "$WORKERS" \
    --tall_ratio_threshold "$TALL_RATIO_THRESHOLD" \
    2>>"$LOG_FILE"

RENDER_EXIT=$?
echo "│"
if [[ $RENDER_EXIT -ne 0 ]]; then
    echo "│  [WARN] 渲染进程退出码: $RENDER_EXIT（部分失败，继续后续步骤）"
fi

LOG_LINES=$(wc -l < "$LOG_FILE" 2>/dev/null || echo 0)
echo "│  日志条目数 : $LOG_LINES 行  →  $LOG_FILE"
echo "└─ STEP 1/3  完成"

# ── 阶段 2: 解析日志 ──────────────────────────────────────
echo ""
echo "┌─ STEP 2/3  解析渲染日志"
echo "│"

python "$SCRIPT_DIR/html_render.py" parse \
    "$LOG_FILE" \
    -o "$ISSUES_FILE" \
    | sed 's/^/│  /'

echo "│"
echo "└─ STEP 2/3  完成  →  $ISSUES_FILE"

# ── 阶段 3: 过滤原数据 ───────────────────────────────────
echo ""
echo "┌─ STEP 3/3  过滤原数据  (level=$FILTER_LEVEL)"
echo "│"

# 清空旧的 panguml 文件（多个子目录会 append 到同一文件）
rm -f "$PANGUML_FILE"
mkdir -p "$(dirname "$PANGUML_FILE")"

SUMMARY_JSONS=()

# 按子目录分别过滤，保持原目录结构；同时追加写入 panguml
for f in "${INPUT_FILES[@]}"; do
    src_dir="$DATA_DIR/$(dirname "$f")"
    sub="$(basename "$src_dir")"
    out_sub="$OUTPUT_DATA_DIR/$sub"
    mkdir -p "$out_sub"
    echo "│  ── $sub"
    python "$SCRIPT_DIR/html_render.py" filter \
        --issues          "$ISSUES_FILE" \
        --input           "$DATA_DIR/$f" \
        --output          "$out_sub" \
        --level           "$FILTER_LEVEL" \
        --json_dir        "$DATA_DIR" \
        --id_field        "$ID_FIELD" \
        --export_panguml  "$PANGUML_FILE" \
        --images_dir      "$IMAGES_DIR" \
        --html_field      "$HTML_FIELD" \
        --templates_file  "$TEMPLATES_FILE" \
        --summary_json    "$STATS_DIR/${sub}_filter.json" \
        | sed 's/^/│    /'
    SUMMARY_JSONS+=("$STATS_DIR/${sub}_filter.json")
    echo "│"
done

# ── 全局汇总（跨所有子目录）+ 全局图表 ───────────────────────
echo "│  ── 全局汇总"
python3 - "${SUMMARY_JSONS[@]}" "$STATS_DIR" <<'PYEOF' | sed 's/^/│    /'
import json, sys
from pathlib import Path

*summary_files, stats_dir_str = sys.argv[1:]
stats_dir = Path(stats_dir_str)
stats_dir.mkdir(parents=True, exist_ok=True)

total = {k: 0 for k in ("original", "removed", "kept",
                         "passthrough_items", "panguml_written", "panguml_skipped")}
# 收集所有子目录的 per-file 明细，每条 = (label, original, removed, had_blocked)
all_file_data = []

for sf in summary_files:
    try:
        d = json.load(open(sf))
        for k in total:
            total[k] += d.get(k, 0)
        subdir = Path(sf).stem.replace("_filter", "")
        for fstat in d.get("files", []):
            label = f"{subdir}/{fstat['name']}"
            all_file_data.append((label, fstat["original"],
                                   fstat["removed"], fstat["had_blocked"]))
    except Exception:
        pass

# ── 文字汇总 ─────────────────────────────────────────────
orig = total["original"]
rm   = total["removed"]
pct  = round(rm / max(orig, 1) * 100, 1) if orig else 0
print(f"[全局] 子目录数      : {len(summary_files)}")
print(f"[全局] 过滤前总条目  : {orig}")
print(f"[全局] 共剔除条目    : {rm} ({pct}%)")
print(f"[全局] 保留条目      : {total['kept']}")
if total["passthrough_items"]:
    print(f"[全局] 直通条目      : {total['passthrough_items']}")
if total["panguml_written"] or total["panguml_skipped"]:
    print(f"[全局] panguml 写入  : {total['panguml_written']} 条  "
          f"跳过: {total['panguml_skipped']} 条")

# ── 全局过滤图表（所有子目录的所有输入文件，每文件一根柱子）──
if not all_file_data:
    sys.exit(0)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("[STATS] matplotlib 未安装，跳过图表生成")
    sys.exit(0)

all_data  = [(n, o, r) for n, o, r, hb in all_file_data if o > 0]
n_files   = len(all_data)
if n_files == 0:
    sys.exit(0)

names     = [n for n, _, _ in all_data]
originals = [o for _, o, _ in all_data]
removed_  = [r for _, _, r in all_data]
kept_     = [o - r for o, r in zip(originals, removed_)]
total_o   = sum(originals)
total_r   = sum(removed_)
total_k   = total_o - total_r
total_pct = round(total_r / max(total_o, 1) * 100, 1)

xs      = list(range(n_files))
x_total = n_files
W       = 0.65

fig, ax1 = plt.subplots(figsize=(max(7, (n_files + 2) * 1.6), 6))
ax2 = ax1.twinx()

y1_max = max(originals) * 1.30
ax1.set_ylim(0, y1_max)
ax1.bar(xs, kept_,    color="#43A047", edgecolor="white", width=W, label="kept")
ax1.bar(xs, removed_, bottom=kept_,
        color="#E53935", edgecolor="white", width=W, label="removed")
ax1.set_ylabel("Count (per file)", fontsize=10)

y2_max = total_o * 1.30
ax2.set_ylim(0, y2_max)
ax2.bar(x_total, total_k, color="#43A047", alpha=0.80, edgecolor="white", width=W)
ax2.bar(x_total, total_r, bottom=total_k,
        color="#E53935", alpha=0.80, edgecolor="white", width=W)
ax2.set_ylabel("Count (total, right axis)", fontsize=10, color="#888888")
ax2.tick_params(axis="y", labelcolor="#888888")

for i, (o, r) in enumerate(zip(originals, removed_)):
    p = round(r / max(o, 1) * 100, 1)
    ax1.text(xs[i], o + y1_max * 0.012, f"{r}/{o}\n{p:.1f}%",
             ha="center", va="bottom", fontsize=7, color="#B71C1C")

ax2.text(x_total, total_o + y2_max * 0.012,
         f"{total_pct:.1f}%\n({total_r:,}/{total_o:,})",
         ha="center", va="bottom", fontsize=8, color="#B71C1C", fontweight="bold")

ax1.set_xticks(xs + [x_total])
ax1.set_xticklabels(names + ["[total]"], rotation=40, ha="right", fontsize=8)
ax1.set_xlim(-0.6, n_files + 0.6)
ax1.axvline(x=n_files - 0.5, color="#AAAAAA", linestyle="--", linewidth=1, alpha=0.7)
ax1.legend(loc="upper left", fontsize=9)
ax1.set_title("Filter results: kept / removed  (last col = total, right axis)", fontsize=11)
plt.tight_layout()

out = stats_dir / "drop_ratio.png"
fig.savefig(out, dpi=120)
plt.close(fig)
print(f"[STATS] 图表已保存: {out}")
PYEOF

echo "│"
echo "└─ STEP 3/3  完成  →  $OUTPUT_DATA_DIR"

# ── 阶段 4: 图片尺寸统计 ──────────────────────────────────
echo ""
echo "┌─ STEP 4/4  图片尺寸统计"
echo "│"

python3 - "$IMAGES_DIR" "$STATS_DIR" <<'PYEOF' | sed 's/^/│  /'
import sys
from pathlib import Path
images_dir = Path(sys.argv[1])
stats_dir  = Path(sys.argv[2])
stats_dir.mkdir(parents=True, exist_ok=True)

try:
    from PIL import Image as _PIL
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("[SKIP] PIL 或 matplotlib 未安装，跳过图片尺寸统计")
    sys.exit(0)

widths, heights = [], []
for png in images_dir.rglob("*.png"):
    try:
        with _PIL.open(png) as img:
            w, h = img.size
            widths.append(w)
            heights.append(h)
    except Exception:
        pass

if not widths:
    print("[SKIP] 未找到 PNG 文件")
    sys.exit(0)

n = len(widths)
print(f"扫描 {n} 张图片")

ratios = [h / w for w, h in zip(widths, heights) if w > 0]

# 宽度/高度分布（x 轴截到 99 百分位，避免少量极端值撑大空间）
w_max = float(np.percentile(widths, 99)) * 1.05
h_max = float(np.percentile(heights, 99)) * 1.05

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(widths,  bins=60, color="steelblue", edgecolor="white", range=(0, w_max))
axes[0].set_xlim(0, w_max)
axes[0].xaxis.set_major_locator(plt.MaxNLocator(nbins=15, integer=True))
axes[0].set_title(f"Width distribution  (n={n})")
axes[0].set_xlabel("Width (px)")
axes[0].set_ylabel("Count")

axes[1].hist(heights, bins=60, color="coral", edgecolor="white", range=(0, h_max))
axes[1].set_xlim(0, h_max)
axes[1].xaxis.set_major_locator(plt.MaxNLocator(nbins=15, integer=True))
axes[1].set_title(f"Height distribution  (n={n})")
axes[1].set_xlabel("Height (px)")

plt.tight_layout()
out = stats_dir / "image_dimensions.png"
fig.savefig(out, dpi=120)
plt.close(fig)
print(f"保存: {out}")

# 高宽比分布
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.hist(ratios, bins=60, color="mediumseagreen", edgecolor="white")
ax2.set_title(f"H/W ratio distribution  (n={n})")
ax2.set_xlabel("H/W ratio")
ax2.set_ylabel("Count")
plt.tight_layout()
out2 = stats_dir / "image_ratio.png"
fig2.savefig(out2, dpi=120)
plt.close(fig2)
print(f"保存: {out2}")

# 文字摘要
print(f"宽度  : min={min(widths)}  max={max(widths)}  mean={int(sum(widths)/n)}")
print(f"高度  : min={min(heights)}  max={max(heights)}  mean={int(sum(heights)/n)}")
if ratios:
    print(f"高宽比: min={min(ratios):.2f}  max={max(ratios):.2f}  mean={sum(ratios)/len(ratios):.2f}")
PYEOF

echo "│"
echo "└─ STEP 4/4  完成  →  $STATS_DIR"

# ── 汇总 ─────────────────────────────────────────────────
END_TS=$(date '+%Y-%m-%d %H:%M:%S')
echo ""
echo "$SEP"
echo "  全部完成  |  $END_TS"
echo "$SEP"
printf "  截图目录  : %s\n" "$IMAGES_DIR"
printf "  问题报告  : %s\n" "$ISSUES_FILE"
printf "  过滤数据  : %s\n" "$OUTPUT_DATA_DIR"
printf "  panguml   : %s\n" "$PANGUML_FILE"
printf "  统计图表  : %s\n" "$STATS_DIR"
echo "$SEP"
