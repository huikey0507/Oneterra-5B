#!/bin/bash

# X-SAM 评估脚本
# 用于评估训练完成的模型，包括所有任务（genseg, ovseg, refseg, imgconv）

cd /mnt_llm_A100_V1/shui/LAE/X-SAM/X-SAM/

# 配置参数
CONFIG="xsam/xsam/configs/xsam/phi3_mini_4k_instruct_siglip2_so400m_p14_384/s3_mixed_finetune_geochat/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune_geochat.py"

# 默认checkpoint路径
DEFAULT_PTH_MODEL="./wkdrs/s3_mixed_finetune_geochat/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune_geochat/pytorch_model.bin"

# 获取参数（checkpoint路径和输出目录都是可选的）
PTH_MODEL="${1:-$DEFAULT_PTH_MODEL}"
WORK_DIR="${2:-./eval_results/s3_mixed_finetune_geochat}"

# 检查checkpoint是否存在
if [ ! -f "$PTH_MODEL" ] && [ ! -d "$PTH_MODEL" ]; then
    echo "错误: checkpoint路径不存在: $PTH_MODEL"
    exit 1
fi

echo "=========================================="
echo "X-SAM 评估脚本"
echo "=========================================="
echo "配置文件: $CONFIG"
echo "Checkpoint: $PTH_MODEL"
if [ "$1" = "" ]; then
    echo "  (使用默认checkpoint)"
fi
echo "输出目录: $WORK_DIR"
echo "=========================================="
echo ""

# 运行评估
python xsam/xsam/tools/eval.py \
    "$CONFIG" \
    --pth_model "$PTH_MODEL" \
    --work-dir "$WORK_DIR"

echo ""
echo "=========================================="
echo "评估完成！"
echo "结果保存在: $WORK_DIR"
echo "  - 预测结果: $WORK_DIR/pred_data/"
echo "  - 可视化: $WORK_DIR/visualizations/"
echo "  - LLM输出: $WORK_DIR/llm_outputs/"
echo "=========================================="
