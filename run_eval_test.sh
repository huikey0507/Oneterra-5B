#!/bin/bash

# X-SAM 验证脚本
# 用于在当前服务器上测试正在训练的模型性能

# 设置路径（根据实际情况修改）
CODE_DIR="/mnt_llm_A100_V1/shui/LAE/X-SAM/X-SAM"
CONFIG_FILE="xsam/xsam/configs/xsam/phi3_mini_4k_instruct_siglip2_so400m_p14_384/s3_mixed_finetune/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune3.py"
WORK_DIR="./eval_results/s3_mixed_finetune_test"

# 检查点路径：可以使用iter_*.pth目录或pytorch_model.bin文件
# 如果使用iter_*.pth目录，脚本会自动查找mp_rank_00_model_states.pt
MODEL_PATH="./wkdrs/s3_mixed_finetune/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune3/iter_28000.pth"

# 或者直接指定模型状态文件（如果上面的目录方式不工作）
# MODEL_PATH="./wkdrs/s3_mixed_finetune/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune3/iter_28000.pth/mp_rank_00_model_states.pt"

# 进入代码目录
cd "$CODE_DIR" || exit 1

# 创建结果目录
mkdir -p "$WORK_DIR"

# 检查模型文件或目录是否存在
if [ ! -e "$MODEL_PATH" ]; then
    echo "错误: 模型文件或目录不存在: $MODEL_PATH"
    echo "请检查模型路径是否正确"
    exit 1
fi

# 如果是目录，检查是否有模型状态文件
if [ -d "$MODEL_PATH" ]; then
    if [ -f "$MODEL_PATH/mp_rank_00_model_states.pt" ]; then
        echo "检测到DeepSpeed检查点目录，将使用: $MODEL_PATH/mp_rank_00_model_states.pt"
    elif [ -f "$MODEL_PATH/pytorch_model.bin" ]; then
        echo "检测到检查点目录，将使用: $MODEL_PATH/pytorch_model.bin"
    else
        echo "警告: 检查点目录中未找到模型文件"
        echo "  查找: mp_rank_00_model_states.pt 或 pytorch_model.bin"
    fi
fi

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    echo "请检查配置文件路径是否正确"
    exit 1
fi

echo "=========================================="
echo "X-SAM 验证测试"
echo "=========================================="
echo "配置文件: $CONFIG_FILE"
echo "工作目录: $WORK_DIR"
echo "模型路径: $MODEL_PATH"
echo "=========================================="
echo ""

# 设置PYTHONPATH（确保可以导入xsam模块）
export PYTHONPATH="$CODE_DIR:$PYTHONPATH"

# 运行验证
python xsam/xsam/tools/eval.py \
    "$CONFIG_FILE" \
    --work-dir "$WORK_DIR" \
    --pth_model "$MODEL_PATH"

echo ""
echo "=========================================="
echo "验证完成！"
echo "结果保存在: $WORK_DIR"
echo "  - 预测结果: $WORK_DIR/pred_data/"
echo "  - 可视化图片: $WORK_DIR/visualizations/"
echo "  - LLM输出: $WORK_DIR/llm_outputs/"
echo "=========================================="

