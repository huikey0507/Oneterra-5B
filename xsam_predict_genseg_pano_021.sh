#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODE_DIR="$ROOT_DIR/xsam"
export PYTHONPATH="$CODE_DIR:${PYTHONPATH:-}"

CONFIG="./xsam/xsam/configs/xsam/phi3_mini_4k_instruct_siglip2_so400m_p14_384/s3_mixed_fineture_base/xsam_base_mixed_finetune_all.py"
DEFAULT_PTH_MODEL="./checkpoints/s3_mixed_fineture_base/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune_all_v1/pytorch_model.bin"

PTH_MODEL="${1:-$DEFAULT_PTH_MODEL}"
IMAGE_DIR="${2:-$ROOT_DIR/datas/pano/val/images}"
OUT_DIR="${3:-$ROOT_DIR/predict_results_genseg_pano_021_0528}"
PANO_JSON="${4:-$ROOT_DIR/assets/annotations_val.json}"
MAX_IMAGES="${5:--1}"

if [ ! -f "$PTH_MODEL" ] && [ ! -d "$PTH_MODEL" ]; then
    echo "错误: checkpoint路径不存在: $PTH_MODEL"
    exit 1
fi

if [ ! -d "$IMAGE_DIR" ]; then
    echo "错误: 图片目录不存在: $IMAGE_DIR"
    exit 1
fi

if [ ! -f "$PANO_JSON" ]; then
    echo "错误: pano定义文件不存在: $PANO_JSON"
    exit 1
fi

echo "=========================================="
echo "X-SAM Genseg Pano Predict 脚本"
echo "=========================================="
echo "配置文件: $CONFIG"
echo "Checkpoint: $PTH_MODEL"
echo "输入图片目录: $IMAGE_DIR"
echo "输出目录: $OUT_DIR"
echo "Pano定义文件: $PANO_JSON"
echo "最大图片数: $MAX_IMAGES"
echo "=========================================="
echo ""

cd "$ROOT_DIR"
python xsam/xsam/tools/predict_genseg_pano.py \
    "$CONFIG" \
    --pth_model "$PTH_MODEL" \
    --image-dir "$IMAGE_DIR" \
    --output-dir "$OUT_DIR" \
    --pano-categories-json "$PANO_JSON" \
    --max-images "$MAX_IMAGES"
