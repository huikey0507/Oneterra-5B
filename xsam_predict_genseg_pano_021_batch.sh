#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODE_DIR="$ROOT_DIR/xsam"
export PYTHONPATH="$CODE_DIR:${PYTHONPATH:-}"

CONFIG="./xsam/xsam/configs/xsam/phi3_mini_4k_instruct_siglip2_so400m_p14_384/s3_mixed_fineture_base/xsam_base_mixed_finetune_all.py"
DEFAULT_PTH_MODEL="./checkpoints/s3_mixed_fineture_base/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune_all_v1/pytorch_model.bin"

PTH_MODEL="${1:-$DEFAULT_PTH_MODEL}"
IMAGE_DIR="${2:-$ROOT_DIR/test_images}"
OUT_DIR="${3:-$ROOT_DIR/predict_results_genseg_pano_021}"
PANO_JSON="${4:-$ROOT_DIR/assets/annotations_val.json}"
MAX_IMAGES="${5:--1}"
# Number of GPUs to use (default: auto-detect via nvidia-smi, fallback to 1)
NGPU="${NGPU:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 1)}"
# Batch size per GPU
BATCH_SIZE="${BATCH_SIZE:-1}"
# DataLoader workers per GPU
NUM_WORKERS="${NUM_WORKERS:-4}"

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
echo "配置文件:       $CONFIG"
echo "Checkpoint:     $PTH_MODEL"
echo "输入图片目录:   $IMAGE_DIR"
echo "输出目录:       $OUT_DIR"
echo "Pano定义文件:   $PANO_JSON"
echo "最大图片数:     $MAX_IMAGES"
echo "GPU数量:        $NGPU"
echo "每GPU batch:    $BATCH_SIZE"
echo "DataLoader workers: $NUM_WORKERS"
echo "=========================================="
echo ""

cd "$ROOT_DIR"

if [ "$NGPU" -gt 1 ]; then
    echo "使用 torchrun 进行多GPU并行推理 (${NGPU} GPUs)..."
    torchrun \
        --nproc_per_node="$NGPU" \
        --master_port="${MASTER_PORT:-29500}" \
        xsam/xsam/tools/predict_genseg_pano_batch.py \
            "$CONFIG" \
            --pth_model "$PTH_MODEL" \
            --image-dir "$IMAGE_DIR" \
            --output-dir "$OUT_DIR" \
            --pano-categories-json "$PANO_JSON" \
            --max-images "$MAX_IMAGES" \
            --batch-size "$BATCH_SIZE" \
            --num-workers "$NUM_WORKERS" \
            --launcher pytorch
else
    echo "使用单GPU推理..."
    python xsam/xsam/tools/predict_genseg_pano.py \
        "$CONFIG" \
        --pth_model "$PTH_MODEL" \
        --image-dir "$IMAGE_DIR" \
        --output-dir "$OUT_DIR" \
        --pano-categories-json "$PANO_JSON" \
        --max-images "$MAX_IMAGES" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS"
fi
