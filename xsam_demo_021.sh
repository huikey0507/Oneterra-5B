#!/bin/bash

# 遥感基础模型 OneTerra-5B Gradio 推理台启动脚本（app_021.py）；与 xsam_eval_021.sh 使用相同默认 config / checkpoint。

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODE_DIR="$ROOT_DIR/xsam"
export PYTHONPATH="$CODE_DIR:${PYTHONPATH:-}"

# 与 xsam_eval_021.sh（→ eval_ori.py）默认 CONFIG / PTH_MODEL 保持一致
CONFIG="./xsam/xsam/configs/xsam/phi3_mini_4k_instruct_siglip2_so400m_p14_384/s3_mixed_fineture_base/xsam_base_mixed_finetune_all.py"
#CONFIG="/mnt_llm_A100_V1/shui/LAE/RS-Xsam-main-old/xsam/xsam/configs/xsam/phi3_mini_4k_instruct_siglip2_so400m_p14_384/s3_mixed_fineture_021/xsam_021_siglip2_so400m_p14_384_sam_large_m2f_mixed_finetune_all_imgconv_optical_part1.py"
#CONFIG="/mnt_llm_A100_V1/shui/LAE/RS-Xsam-main-old/xsam/xsam/configs/xsam/phi3_mini_4k_instruct_siglip2_so400m_p14_384/s3_mixed_fineture_base/xsam_v3_ultimate_4xA40_eval.py"
#CONFIG="/mnt_llm_A100_V1/shui/LAE/OneTerra-train/xsam/xsam/configs/xsam/phi3_mini_4k_instruct_siglip2_so400m_p14_384/s3_mixed_fineture_base/xsam_v3_finetune_v2_A100_64gpu.py"

DEFAULT_PTH_MODEL="./checkpoints/s3_mixed_fineture_base/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_ovseg_subset_ft_flat_v1/iter_30000.pth"
#DEFAULT_PTH_MODEL="/mnt_llm_A100_V1/shui/LAE/OneTerra-train/wkdrs_optimization_20260611/iter_10000.pth"
#DEFAULT_PTH_MODEL="/mnt_llm_A100_V1/yangsen/workspace/OneTerra-train-stage3/wkdrs_nanhu/s3_mixed_fineture_base/xsam_021_siglip2_so400m_p14_384_sam_large_m2f_mixed_finetune_all_nanhu_v1/pytorch_model.bin"
#DEFAULT_PTH_MODEL="/mnt_llm_A100_V1/shui/LAE/OneTerra-train/wkdrs_v3_ultimate_4xA40/iter_31000.pth"
#DEFAULT_PTH_MODEL="./checkpoints/s3_mixed_fineture_base/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune_all_v1/pytorch_model.bin"
# 参数与 eval 对齐：$1 checkpoint，$2 work-dir（与 eval 的第二个参数含义相同，用于 vis/ 与 latest）
PTH_MODEL="${1:-$DEFAULT_PTH_MODEL}"
WORK_DIR="${2:-./demo_work_021}"
PORT="${3:-7862}"
HOST="${4:-0.0.0.0}"
LOG_DIR="${5:-./demo_logs_021}"

if [ ! -f "$PTH_MODEL" ] && [ ! -d "$PTH_MODEL" ]; then
    echo "错误: checkpoint路径不存在: $PTH_MODEL"
    exit 1
fi

echo "=========================================="
echo "遥感基础模型OneTerra-5B 推理台 (app_021)"
echo "=========================================="
echo "配置文件: $CONFIG"
echo "Checkpoint: $PTH_MODEL"
echo "work-dir (vis/): $WORK_DIR"
echo "端口: $PORT"
echo "地址: $HOST"
echo "日志目录: $LOG_DIR"
echo "=========================================="
echo ""

cd "$ROOT_DIR"
# 界面在 app_021.py：输入为单图上传（无图层/画笔）；显示区 CSS 减少预览黑边。
python xsam/xsam/demo/app_021.py \
    "$CONFIG" \
    --pth_model "$PTH_MODEL" \
    --work-dir "$WORK_DIR" \
    --seed 0 \
    --port "$PORT" \
    --host "$HOST" \
    --log-dir "$LOG_DIR"
