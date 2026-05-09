#!/bin/bash

# X-SAM 评估脚本 - 使用2张GPU进行分布式评估（更稳定，避免同步超时）
# 评估所有测试集样本，使用训练完成后的最新checkpoint

set -e

# 设置路径
CODE_DIR="/mnt_llm_A100_V1/shui/LAE/X-SAM/X-SAM"
CONFIG_FILE="xsam/xsam/configs/xsam/phi3_mini_4k_instruct_siglip2_so400m_p14_384/s3_mixed_finetune/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune3.py"
WORK_DIR="./eval_results/s3_mixed_finetune_test_all"

# 使用最新的checkpoint（训练完成后的最终结果）
MODEL_DIR="./wkdrs/s3_mixed_finetune/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune3"
# 查找最新的iter_*.pth目录
LATEST_CHECKPOINT=$(find "$MODEL_DIR" -name "iter_*.pth" -type d | sort -V | tail -1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "错误: 未找到checkpoint目录"
    echo "请检查模型目录: $MODEL_DIR"
    exit 1
fi

echo "=========================================="
echo "X-SAM 评估 - 2 GPU分布式评估（更稳定）"
echo "=========================================="
echo "配置文件: $CONFIG_FILE"
echo "工作目录: $WORK_DIR"
echo "使用checkpoint: $LATEST_CHECKPOINT"
echo "GPU数量: 2（减少同步问题）"
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

# 设置环境变量（避免NCCL超时和其他分布式问题）
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export XTUNER_DATASET_TIMEOUT=120
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_NET_GDR_LEVEL=2
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4
export NCCL_TIMEOUT=36000  # 10小时超时，避免评估过程中的NCCL超时
export DIST_TIMEOUT=36000  # PyTorch分布式超时时间（秒）
export NCCL_DEBUG=WARN  # 减少NCCL日志输出

# 设置分布式训练参数
MASTER_PORT=${MASTER_PORT:-29500}
GPUS=2  # 使用2张GPU，减少同步问题

echo "使用torchrun启动分布式评估..."
echo "GPU数量: $GPUS（使用更少的GPU以减少同步超时问题）"
echo "Master Port: $MASTER_PORT"
echo ""

# 使用torchrun启动分布式评估（2张GPU）
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

