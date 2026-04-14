#!/usr/bin/env bash
# run_sagoge_distill.sh — 一键运行 SAgoge 蒸馏数据转换 + 过滤
#
# 所有数据集相关路径集中在这里配置，修改此处即可适配其他数据集
# （如 stage2_illustration 等）。
set -euo pipefail

# ─── 输入 ──────────────────────────────────────────────────────────────────────
INPUT="/home/yanghaitao/Projects/DataPipeline/outputs/distillation/SAgoge/svg_responses.jsonl"

# ─── 输出根目录 ────────────────────────────────────────────────────────────────
OUTPUT_ROOT="/home/yanghaitao/Projects/Data/processed/SAgoge_distillation"

# ─── 数据集特定路径配置 ────────────────────────────────────────────────────────
# 渲染后 PNG 图片存放目录
IMAGES_DIR="$OUTPUT_ROOT/stage1_icon/img2svg/images"

# canonical schema 中 relative_path 的计算基准：
#   relative_path = relpath(image_file, IMAGE_ROOT)
#   下游读取：full_path = IMAGE_ROOT / relative_path
IMAGE_ROOT="$OUTPUT_ROOT"

# 带 _meta 的中间 JSONL（convert 阶段产出）
INTER_JSONL="$OUTPUT_ROOT/stage1_icon/img2svg/intermediate/data_000000.jsonl"

# 过滤后最终 JSONL（不含 _meta）
FINAL_JSONL="$OUTPUT_ROOT/stage1_icon/img2svg/jsonl/data_000000.jsonl"

# ─── 其他参数 ──────────────────────────────────────────────────────────────────
TRAIN_MODE="sft"
WORKERS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$SCRIPT_DIR/convert_sagoge_distill.py"

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/sagoge_distill_${TIMESTAMP}.log"

# ─── 显示配置 ─────────────────────────────────────────────────────────────────
echo "========================================================"
echo "  SAgoge Distillation — Convert + Filter"
echo "========================================================"
echo "  INPUT:       $INPUT"
echo "  IMAGES_DIR:  $IMAGES_DIR"
echo "  IMAGE_ROOT:  $IMAGE_ROOT"
echo "  INTER_JSONL: $INTER_JSONL"
echo "  FINAL_JSONL: $FINAL_JSONL"
echo "  TRAIN_MODE:  $TRAIN_MODE"
echo "  WORKERS:     $WORKERS"
echo "  LOG:         $LOG_FILE"
echo "========================================================"
echo ""

# ─── Step 1: Convert ──────────────────────────────────────────────────────────
echo "[1/2] Converting: rendering SVGs in parallel ..."
python3 "$SCRIPT" convert \
    --input       "$INPUT" \
    --images-dir  "$IMAGES_DIR" \
    --image-root  "$IMAGE_ROOT" \
    --inter-jsonl "$INTER_JSONL" \
    --train-mode  "$TRAIN_MODE" \
    --workers     "$WORKERS" \
    --log-path    "$LOG_FILE"

echo ""

# ─── Step 2: Filter ───────────────────────────────────────────────────────────
echo "[2/2] Filtering: removing invalid / all-white samples ..."
python3 "$SCRIPT" filter \
    --input        "$INTER_JSONL" \
    --output-jsonl "$FINAL_JSONL" \
    --log-path     "$LOG_FILE"

echo ""
echo "========================================================"
echo "  Done!"
echo "  Intermediate JSONL : $INTER_JSONL"
echo "  Final JSONL        : $FINAL_JSONL"
echo "  Images             : $IMAGES_DIR"
echo "  Log                : $LOG_FILE"
echo "========================================================"
