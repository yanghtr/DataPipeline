#!/usr/bin/env bash
# run_sagoge_distill.sh — 一键运行 SAgoge 蒸馏数据转换 + 过滤
set -euo pipefail

# ─── 路径配置（按需修改）────────────────────────────────────────────────────────
INPUT="/home/yanghaitao/Projects/DataPipeline/outputs/distillation/SAgoge/svg_responses.jsonl"
OUTPUT="/home/yanghaitao/Projects/Data/processed/SAgoge_distillation"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$SCRIPT_DIR/convert_sagoge_distill.py"

TRAIN_MODE="sft"
WORKERS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/sagoge_distill_${TIMESTAMP}.log"

INTERMEDIATE_JSONL="$OUTPUT/stage1_icon/img2svg/intermediate/data_000000.jsonl"
FINAL_JSONL="$OUTPUT/stage1_icon/img2svg/jsonl/data_000000.jsonl"

# ─── 显示配置 ─────────────────────────────────────────────────────────────────
echo "========================================================"
echo "  SAgoge Distillation — Convert + Filter"
echo "========================================================"
echo "  INPUT:       $INPUT"
echo "  OUTPUT:      $OUTPUT"
echo "  TRAIN_MODE:  $TRAIN_MODE"
echo "  WORKERS:     $WORKERS"
echo "  LOG:         $LOG_FILE"
echo "========================================================"
echo ""

# ─── Step 1: Convert ──────────────────────────────────────────────────────────
echo "[1/2] Converting: rendering SVGs in parallel ..."
python3 "$SCRIPT" convert \
    --input      "$INPUT" \
    --output     "$OUTPUT" \
    --train-mode "$TRAIN_MODE" \
    --workers    "$WORKERS" \
    --log-path   "$LOG_FILE"

echo ""

# ─── Step 2: Filter ───────────────────────────────────────────────────────────
echo "[2/2] Filtering: removing invalid / all-white samples ..."
python3 "$SCRIPT" filter \
    --input    "$INTERMEDIATE_JSONL" \
    --output   "$OUTPUT" \
    --log-path "$LOG_FILE"

echo ""
echo "========================================================"
echo "  Done!"
echo "  Intermediate JSONL : $INTERMEDIATE_JSONL"
echo "  Final JSONL        : $FINAL_JSONL"
echo "  Images             : $OUTPUT/stage1_icon/img2svg/images/"
echo "  Log                : $LOG_FILE"
echo "========================================================"
