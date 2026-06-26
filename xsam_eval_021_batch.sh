#!/bin/bash

# X-SAM 多卡/多 batch 评估：调用 xsam/xsam/tools/eval_ori_batch.py
# 单卡原版评测请用 xsam_eval_021.sh + eval_ori.py

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODE_DIR="$ROOT_DIR/xsam"
export PYTHONPATH="$CODE_DIR:${PYTHONPATH:-}"

CONFIG="./xsam/xsam/configs/xsam/phi3_mini_4k_instruct_siglip2_so400m_p14_384/s3_mixed_fineture_base/xsam_base_mixed_finetune_all.py"
#CONFIG="./xsam/xsam/configs/xsam/phi3_mini_4k_instruct_siglip2_so400m_p14_384/s3_mixed_fineture_021/xsam_021_siglip2_so400m_p14_384_sam_large_m2f_mixed_finetune_all_imgconv_optical_part1.py"
#CONFIG="/mnt_llm_A100_V1/shui/LAE/OneTerra-train/xsam/xsam/configs/xsam/phi3_mini_4k_instruct_siglip2_so400m_p14_384/s3_mixed_fineture_base/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_ovseg_subset_ft_flat.py"
#DEFAULT_PTH_MODEL="./checkpoints/s3_mixed_fineture_base/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune_all_v1/pytorch_model.bin"
#CONFIG="/mnt_llm_A100_V1/shui/LAE/RS-Xsam-main-old/xsam/xsam/configs/xsam/phi3_mini_4k_instruct_siglip2_so400m_p14_384/s3_mixed_fineture_base/xsam_v3_ultimate_4xA40_eval.py"



DEFAULT_PTH_MODEL="./checkpoints/s3_mixed_fineture_base/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_ovseg_subset_ft_flat_v1/iter_30000.pth"
#DEFAULT_PTH_MODEL="/mnt_llm_A100_V1/yangsen/workspace/OneTerra-train-stage3/wkdrs_nanhu/s3_mixed_fineture_base/xsam_021_siglip2_so400m_p14_384_sam_large_m2f_mixed_finetune_all_nanhu_v1/pytorch_model.bin"
#DEFAULT_PTH_MODEL="/mnt_llm_A100_V1/shui/LAE/OneTerra-train/wkdrs_optimization_20260611/iter_110000.pth"

PTH_MODEL="${1:-$DEFAULT_PTH_MODEL}"
WORK_DIR="${2:-./eval_results_base_20260616/s3_v2_onlyovseg}"

NGPU="${NGPU:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 1)}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_VIS="${SAVE_VIS:-1}"

if [ ! -f "$PTH_MODEL" ] && [ ! -d "$PTH_MODEL" ]; then
    echo "错误: checkpoint路径不存在: $PTH_MODEL"
    exit 1
fi

EVAL_EXTRA_ARGS=()
if [ "$SAVE_VIS" = "0" ]; then
    EVAL_EXTRA_ARGS+=(--no-vis)
fi

echo "=========================================="
echo "X-SAM 评估脚本 (多卡/多 batch)"
echo "=========================================="
echo "配置文件:       $CONFIG"
echo "Checkpoint:     $PTH_MODEL"
if [ "${1:-}" = "" ]; then
    echo "  (使用默认checkpoint)"
fi
echo "输出目录:       $WORK_DIR"
echo "GPU数量:        $NGPU"
echo "每GPU batch:    $BATCH_SIZE (genseg/ovseg 可用>1；refseg/reaseg/imgconv 自动降为1)"
echo "DataLoader workers: $NUM_WORKERS"
echo "保存可视化:     $SAVE_VIS (0=关闭)"
echo "=========================================="
echo ""

cd "$ROOT_DIR"

if [ "$NGPU" -gt 1 ]; then
    echo "使用 torchrun 多卡评测 (${NGPU} GPUs)..."
    torchrun \
        --nproc_per_node="$NGPU" \
        --master_port="${MASTER_PORT:-29502}" \
        xsam/xsam/tools/eval_ori_batch.py \
            "$CONFIG" \
            --pth_model "$PTH_MODEL" \
            --work-dir "$WORK_DIR" \
            --batch-size "$BATCH_SIZE" \
            --num-workers "$NUM_WORKERS" \
            --launcher pytorch \
            "${EVAL_EXTRA_ARGS[@]}"
else
    echo "使用单卡评测..."
    python xsam/xsam/tools/eval_ori_batch.py \
        "$CONFIG" \
        --pth_model "$PTH_MODEL" \
        --work-dir "$WORK_DIR" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        "${EVAL_EXTRA_ARGS[@]}"
fi

EVAL_EXIT_CODE=$?

if [ $EVAL_EXIT_CODE -ne 0 ]; then
    exit $EVAL_EXIT_CODE
fi

echo ""
echo "=========================================="
echo "任务评估完成！"
echo "结果保存在: $WORK_DIR"
echo "=========================================="
