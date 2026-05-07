#!/usr/bin/env bash
# HTML 渲染管线一键脚本
# 用法: bash run.sh [--workers N] [--level error|warn|all]
#
# 三阶段:
#   1. render  → PNG 截图，stderr 重定向到 $LOG_FILE（每次覆盖）
#   2. parse   → 解析日志生成 $ISSUES_FILE
#   3. filter  → 根据 issues 剔除有问题的记录，写到 $OUTPUT_DATA_DIR
#
# 注意：render 阶段默认 resume——已有 PNG 的条目会被跳过，
#       跳过条目的历史 WARN/ERROR 不会出现在当次日志里。
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

# instruction 模板文件（panguml user turn 随机采样）
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TEMPLATES_FILE="$SCRIPT_ROOT/converters/templates/image_to_html_templates.txt"

# HTML / ID 字段名（针对当前数据集）
HTML_FIELD="output_html"
ID_FIELD="record_uid"

# ── 可调参数 ──────────────────────────────────────────────
WORKERS="${WORKERS:-$(( $(nproc) - 1 ))}"
WORKERS=$(( WORKERS < 1 ? 1 : WORKERS ))

FILTER_LEVEL="${FILTER_LEVEL:-all}"   # error | warn | all

# ── 参数解析 ──────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --workers) WORKERS="$2"; shift 2 ;;
        --level)   FILTER_LEVEL="$2"; shift 2 ;;
        *) echo "[ERROR] 未知参数: $1"; exit 1 ;;
    esac
done

# ── 准备 ─────────────────────────────────────────────────
mkdir -p "$IMAGES_DIR"
START_TS=$(date '+%Y-%m-%d %H:%M:%S')

echo "$SEP"
echo "  HTML 渲染管线  |  $START_TS"
echo "$SEP"
printf "  数据目录  : %s\n" "$DATA_DIR"
printf "  截图目录  : %s\n" "$IMAGES_DIR"
printf "  日志文件  : %s  (每次覆盖，resume 的跳过条目不记入)\n" "$LOG_FILE"
printf "  并行进程  : %s\n" "$WORKERS"
printf "  HTML 字段 : %s\n" "$HTML_FIELD"
printf "  ID 字段   : %s\n" "$ID_FIELD"
printf "  过滤级别  : %s\n" "$FILTER_LEVEL"
printf "  panguml   : %s\n" "$PANGUML_FILE"
printf "  输入文件  :\n"
for f in "${INPUT_FILES[@]}"; do
    printf "    %s\n" "$f"
done
echo "$SEP"

# ── 阶段 1: 渲染 ─────────────────────────────────────────
echo ""
echo "┌─ STEP 1/3  渲染 HTML → PNG"
echo "│"

python "$SCRIPT_DIR/html_render.py" render \
    --json_dir   "$DATA_DIR" \
    --files      "${INPUT_FILES[@]}" \
    --images_dir "$IMAGES_DIR" \
    --html_field "$HTML_FIELD" \
    --id_field   "$ID_FIELD" \
    --workers    "$WORKERS" \
    2>"$LOG_FILE"

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

# 按子目录分别过滤，保持原目录结构；同时追加写入 panguml
for f in "${INPUT_FILES[@]}"; do
    src_dir="$DATA_DIR/$(dirname "$f")"
    sub="$(basename "$src_dir")"
    out_sub="$OUTPUT_DATA_DIR/$sub"
    mkdir -p "$out_sub"
    echo "│  ── $sub"
    python "$SCRIPT_DIR/html_render.py" filter \
        --issues          "$ISSUES_FILE" \
        --input           "$src_dir" \
        --output          "$out_sub" \
        --level           "$FILTER_LEVEL" \
        --json_dir        "$DATA_DIR" \
        --id_field        "$ID_FIELD" \
        --export_panguml  "$PANGUML_FILE" \
        --images_dir      "$IMAGES_DIR" \
        --html_field      "$HTML_FIELD" \
        --templates_file  "$TEMPLATES_FILE" \
        | sed 's/^/│    /'
    echo "│"
done

echo "└─ STEP 3/3  完成  →  $OUTPUT_DATA_DIR"

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
echo "$SEP"
