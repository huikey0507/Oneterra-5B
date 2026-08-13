#!/bin/bash

# X-SAM SAR Align 多卡/多 batch 评估：调用 xsam/xsam/tools/eval_ori_batch.py
# 保留原 xsam_eval_021_batch.sh 不动，仅切换到 SAR 对齐专用 config。

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODE_DIR="$ROOT_DIR/xsam"
export PYTHONPATH="$CODE_DIR:${PYTHONPATH:-}"

CONFIG="./xsam/xsam/configs/xsam/phi3_mini_4k_instruct_siglip2_so400m_p14_384/s3_mixed_fineture_base/xsam_dev_sar_align_eval.py"
DEFAULT_PTH_MODEL="/mnt_llm_A100_V1/shui/LAE/OneTerra-train/wkdrs_sar_align_B/iter_11456.pth"
DEFAULT_PREV_SAR_A_CKPT="/mnt_llm_A100_V1/shui/LAE/OneTerra-train/wkdrs_sar_align_A/iter_91000.pth"

PTH_MODEL="${1:-$DEFAULT_PTH_MODEL}"
WORK_DIR="${2:-./eval_results_sar_align_stageB}"
PREV_SAR_A_CKPT="${PREV_SAR_A_CKPT:-$DEFAULT_PREV_SAR_A_CKPT}"

NGPU="${NGPU:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 1)}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_VIS="${SAVE_VIS:-1}"
FORCE_ROUTE_MODE="${FORCE_ROUTE_MODE:-auto}"

if [ ! -f "$PTH_MODEL" ] && [ ! -d "$PTH_MODEL" ]; then
    echo "错误: checkpoint路径不存在: $PTH_MODEL"
    exit 1
fi
if [ -n "$PREV_SAR_A_CKPT" ] && [ ! -f "$PREV_SAR_A_CKPT" ] && [ ! -d "$PREV_SAR_A_CKPT" ]; then
    echo "错误: Stage A checkpoint 不存在: $PREV_SAR_A_CKPT"
    exit 1
fi

EVAL_EXTRA_ARGS=()
if [ -n "$PREV_SAR_A_CKPT" ]; then
    EVAL_EXTRA_ARGS+=(--prev-pth-model "$PREV_SAR_A_CKPT")
fi
if [ "$SAVE_VIS" = "0" ]; then
    EVAL_EXTRA_ARGS+=(--no-vis)
fi
# 始终写入 force_route_mode，并把结果目录名拼上 _auto/_sar/_optical，避免互相覆盖
EVAL_EXTRA_ARGS+=(--cfg-options "model.force_route_mode=${FORCE_ROUTE_MODE}")
EVAL_EXTRA_ARGS+=(--pred-name-suffix "_${FORCE_ROUTE_MODE}")

echo "=========================================="
echo "X-SAM SAR Align 评估脚本 (多卡/多 batch)"
echo "=========================================="
echo "配置文件:       $CONFIG"
echo "Checkpoint:     $PTH_MODEL"
echo "Stage A (prev): $PREV_SAR_A_CKPT"
if [ "${1:-}" = "" ]; then
    echo "  (使用默认checkpoint)"
fi
echo "输出目录:       $WORK_DIR"
echo "结果子目录后缀: _${FORCE_ROUTE_MODE}  (pred_data/<data_name>_${FORCE_ROUTE_MODE})"
echo "GPU数量:        $NGPU"
echo "每GPU batch:    $BATCH_SIZE (genseg/ovseg 可用>1；refseg/reaseg/imgconv 自动降为1)"
echo "DataLoader workers: $NUM_WORKERS"
echo "保存可视化:     $SAVE_VIS (0=关闭)"
echo "路由模式:       $FORCE_ROUTE_MODE (auto/optical/sar)"
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
