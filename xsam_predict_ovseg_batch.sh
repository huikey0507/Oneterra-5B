#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODE_DIR="$ROOT_DIR/xsam"
export PYTHONPATH="$CODE_DIR:${PYTHONPATH:-}"

CONFIG="./xsam/xsam/configs/xsam/phi3_mini_4k_instruct_siglip2_so400m_p14_384/s3_mixed_fineture_base/xsam_base_mixed_finetune_all.py"
#DEFAULT_PTH_MODEL="./checkpoints/s3_mixed_fineture_base/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune_all_v1/pytorch_model.bin"
DEFAULT_PTH_MODEL="./checkpoints/s3_mixed_fineture_base/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_ovseg_subset_ft_flat_v1/iter_30000.pth"

PTH_MODEL="${1:-$DEFAULT_PTH_MODEL}"
IMAGE_DIR="${2:-$ROOT_DIR/test_images}"
OUT_DIR="${3:-$ROOT_DIR/predict_results_ovseg_021}"
MAX_IMAGES="${4:--1}"

# ========== 开集类别（请在此修改）==========
# 方式1：分别定义 thing / stuff（逗号分隔，推荐）
OVSEG_THING_CLASSES="airplane, car, building, bridage, dam, port, ship"
OVSEG_STUFF_CLASSES="road, water, farmland, grassland, forest"

# 方式2：直接写完整 prompt（非空时优先于上面的 thing/stuff）
# 格式: "thing: a, b; stuff: c, d"  或仅逗号列表（全部视为 stuff）
OVSEG_PROMPT=""
# OVSEG_PROMPT="thing: person, car; stuff: road, water, sky, vegetation"
# =============================================

NGPU="${NGPU:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 1)}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
THRESHOLD="${THRESHOLD:-0.0}"

if [ -n "${OVSEG_PROMPT:-}" ]; then
    PROMPT="$OVSEG_PROMPT"
elif [ -n "${OVSEG_THING_CLASSES:-}" ] || [ -n "${OVSEG_STUFF_CLASSES:-}" ]; then
    PROMPT="thing: ${OVSEG_THING_CLASSES}; stuff: ${OVSEG_STUFF_CLASSES}"
else
    echo "错误: 请在脚本中设置 OVSEG_PROMPT，或设置 OVSEG_THING_CLASSES / OVSEG_STUFF_CLASSES"
    exit 1
fi

if [ ! -f "$PTH_MODEL" ] && [ ! -d "$PTH_MODEL" ]; then
    echo "错误: checkpoint路径不存在: $PTH_MODEL"
    exit 1
fi

if [ ! -d "$IMAGE_DIR" ]; then
    echo "错误: 图片目录不存在: $IMAGE_DIR"
    exit 1
fi

echo "=========================================="
echo "X-SAM OVSeg (开集全景) 批量预测脚本"
echo "=========================================="
echo "配置文件:       $CONFIG"
echo "Checkpoint:     $PTH_MODEL"
echo "输入图片目录:   $IMAGE_DIR"
echo "输出目录:       $OUT_DIR"
echo "开集 Prompt:    $PROMPT"
echo "最大图片数:     $MAX_IMAGES"
echo "GPU数量:        $NGPU"
echo "每GPU batch:    $BATCH_SIZE"
echo "DataLoader workers: $NUM_WORKERS"
echo "后处理阈值:     $THRESHOLD"
echo "=========================================="
echo ""

cd "$ROOT_DIR"

if [ "$NGPU" -gt 1 ]; then
    echo "使用 torchrun 进行多GPU并行推理 (${NGPU} GPUs)..."
    torchrun \
        --nproc_per_node="$NGPU" \
        --master_port="${MASTER_PORT:-29501}" \
        xsam/xsam/tools/predict_ovseg_batch.py \
            "$CONFIG" \
            --pth_model "$PTH_MODEL" \
            --image-dir "$IMAGE_DIR" \
            --output-dir "$OUT_DIR" \
            --prompt "$PROMPT" \
            --max-images "$MAX_IMAGES" \
            --batch-size "$BATCH_SIZE" \
            --num-workers "$NUM_WORKERS" \
            --threshold "$THRESHOLD" \
            --launcher pytorch
else
    echo "使用单GPU推理..."
    python xsam/xsam/tools/predict_ovseg_batch.py \
        "$CONFIG" \
        --pth_model "$PTH_MODEL" \
        --image-dir "$IMAGE_DIR" \
        --output-dir "$OUT_DIR" \
        --prompt "$PROMPT" \
        --max-images "$MAX_IMAGES" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --threshold "$THRESHOLD"
fi
