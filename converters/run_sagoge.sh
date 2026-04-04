#!/usr/bin/env bash
# 批量转换 SAgoge 数据集的全部 6 个子集
# 用法: bash converters/run_sagoge.sh
# 可在 DataPipeline 项目根目录执行

set -euo pipefail

RAW_ROOT="/cache/Data/SAgoge/raw/"
PROCESSED_ROOT="/cache/Data/SAgoge/processed/"
SCRIPT="$(dirname "$0")/convert_sagoge.py"
TRAIN_MODE="sft"
IMAGE_ROOT="$RAW_ROOT"

SUBSETS=(
  "stage1/icon/generation/img2svg/img2svg_alpaca.jsonl:stage1/icon/generation/img2svg/data_000000.jsonl"
  "stage1/icon/generation/text2svg/text2svg_alpaca.jsonl:stage1/icon/generation/text2svg/data_000000.jsonl"
  "stage2/icon/generation/img2svg/img2svg_alpaca.jsonl:stage2/icon/generation/img2svg/data_000000.jsonl"
  "stage2/icon/generation/text2svg/text2svg_alpaca.jsonl:stage2/icon/generation/text2svg/data_000000.jsonl"
  "stage2/illustration/img2svg/img2svg_alpaca.jsonl:stage2/illustration/img2svg/data_000000.jsonl"
  "stage2/illustration/text2svg/text2svg_alpaca.jsonl:stage2/illustration/text2svg/data_000000.jsonl"
)

for entry in "${SUBSETS[@]}"; do
  INPUT_REL="${entry%%:*}"
  OUTPUT_REL="${entry##*:}"
  INPUT="$RAW_ROOT/$INPUT_REL"
  OUTPUT="$PROCESSED_ROOT/$OUTPUT_REL"
  LOG="$PROCESSED_ROOT/logs/$(echo "$INPUT_REL" | tr '/' '_').log"

  echo "======================================================"
  echo "输入: $INPUT"
  echo "输出: $OUTPUT"
  echo "------------------------------------------------------"
  python3 "$SCRIPT" \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --train-mode "$TRAIN_MODE" \
    --image-root "$IMAGE_ROOT" \
    --log-path "$LOG"
done

echo "======================================================"
echo "所有子集转换完成。"
