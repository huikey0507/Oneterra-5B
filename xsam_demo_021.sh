#!/bin/bash

# X-SAM Demo 启动脚本（对齐 xsam_eval_021.sh 的配置与权重）

set -euo pipefail

# 保证从脚本所在目录启动，并设置与训练/评测一致的 PYTHONPATH
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODE_DIR="$ROOT_DIR/xsam"
export PYTHONPATH="$CODE_DIR:${PYTHONPATH:-}"

# 与 xsam_eval_021.sh 保持一致的默认配置
CONFIG="/mnt_llm_A100_V1/shui/LAE/RS-Xsam-main-old/xsam/xsam/configs/xsam/phi3_mini_4k_instruct_siglip2_so400m_p14_384/s3_mixed_fineture_base/xsam_base_mixed_finetune_all.py"
DEFAULT_PTH_MODEL="/mnt_llm_A100_V1/shui/LAE/OneTerra-train/wkdrs_01/s3_mixed_fineture_base/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune_all_v1/pytorch_model.bin"

# 可选参数
PTH_MODEL="${1:-$DEFAULT_PTH_MODEL}"
PORT="${2:-7862}"
HOST="${3:-0.0.0.0}"
LOG_DIR="${4:-./demo_logs_021}"

if [ ! -f "$PTH_MODEL" ] && [ ! -d "$PTH_MODEL" ]; then
    echo "错误: checkpoint路径不存在: $PTH_MODEL"
    exit 1
fi

echo "=========================================="
echo "X-SAM Demo 启动脚本"
echo "=========================================="
echo "配置文件: $CONFIG"
echo "Checkpoint: $PTH_MODEL"
echo "端口: $PORT"
echo "地址: $HOST"
echo "日志目录: $LOG_DIR"
echo "=========================================="
echo ""

cd "$ROOT_DIR"
python xsam/xsam/demo/app.py \
    "$CONFIG" \
    --pth_model "$PTH_MODEL" \
    --seed 0 \
    --port "$PORT" \
    --host "$HOST" \
    --log-dir "$LOG_DIR"
