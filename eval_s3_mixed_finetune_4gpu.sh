#!/bin/bash

# X-SAM 评估脚本 - 使用4张GPU进行分布式评估
# 评估所有测试集样本，使用最新的checkpoint

set -e

# 设置路径
CODE_DIR="/mnt_llm_A100_V1/shui/LAE/X-SAM/X-SAM"
CONFIG_FILE="xsam/xsam/configs/xsam/phi3_mini_4k_instruct_siglip2_so400m_p14_384/s3_mixed_finetune/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune3.py"
WORK_DIR="./eval_results/s3_mixed_finetune_test"

# 使用最新的checkpoint
MODEL_DIR="./wkdrs/s3_mixed_finetune/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune3"
# 查找最新的iter_*.pth目录
LATEST_CHECKPOINT=$(find "$MODEL_DIR" -name "iter_*.pth" -type d | sort -V | tail -1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "错误: 未找到checkpoint目录"
    echo "请检查模型目录: $MODEL_DIR"
    exit 1
fi

echo "=========================================="
echo "X-SAM 评估 - 4 GPU分布式评估"
echo "=========================================="
echo "配置文件: $CONFIG_FILE"
echo "工作目录: $WORK_DIR"
echo "使用checkpoint: $LATEST_CHECKPOINT"
echo "GPU数量: 4"
echo "评估模式: 所有测试集样本（无限制）"
echo "=========================================="
echo ""

# 进入代码目录
cd "$CODE_DIR" || exit 1

# 创建结果目录
mkdir -p "$WORK_DIR"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 检查checkpoint目录是否存在
if [ ! -d "$LATEST_CHECKPOINT" ]; then
    echo "错误: checkpoint目录不存在: $LATEST_CHECKPOINT"
    exit 1
fi

# 设置PYTHONPATH
export PYTHONPATH="$CODE_DIR:$PYTHONPATH"

# 设置分布式训练参数
MASTER_PORT=${MASTER_PORT:-29500}
GPUS=4

echo "使用torchrun启动分布式评估..."
echo "GPU数量: $GPUS"
echo "Master Port: $MASTER_PORT"
echo ""

# 使用torchrun启动分布式评估（4张GPU）
torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=$GPUS \
    --master_port=$MASTER_PORT \
    xsam/xsam/tools/eval.py \
    "$CONFIG_FILE" \
    --work-dir "$WORK_DIR" \
    --pth_model "$LATEST_CHECKPOINT" \
    --launcher pytorch

echo ""
echo "=========================================="
echo "评估完成！"
echo "结果保存在: $WORK_DIR"
echo "  - 预测结果: $WORK_DIR/pred_data/"
echo "  - 可视化图片: $WORK_DIR/visualizations/"
echo "  - LLM输出: $WORK_DIR/llm_outputs/"
echo "=========================================="

