#!/bin/bash

# X-SAM 评估入口：调用 xsam/xsam/tools/eval_ori.py（评测任务与数据由 CONFIG 中 val_dataloader 等决定）

# 配置参数
#CONFIG="/mnt_llm_A100_V1/shui/LAE/XSAM-public/RS-Xsam-main/xsam/xsam/configs/xsam/phi3_mini_4k_instruct_siglip2_so400m_p14_384/s3_mixed_fineture_base/xsam_021_siglip2_so400m_p14_384_sam_large_m2f_mixed_finetune_all.py"
CONFIG="/mnt_llm_A100_V1/shui/LAE/RS-Xsam-main-old/xsam/xsam/configs/xsam/phi3_mini_4k_instruct_siglip2_so400m_p14_384/s3_mixed_fineture_base/xsam_base_mixed_finetune_all.py"
#CONFIG="/mnt_llm_A100_V1/shui/LAE/OneTerra-train/xsam/xsam/configs/xsam/phi3_mini_4k_instruct_siglip2_so400m_p14_384/s3_mixed_fineture_base/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_ovseg_subset_ft_flat.py"
# 默认checkpoint路径
#DEFAULT_PTH_MODEL="/mnt_llm_A100_V1/shui/LAE/XSAM-public/RS-Xsam-main/checkpoints_021/s3_mixed_fineture_base/xsam_021_siglip2_so400m_p14_384_sam_large_m2f_mixed_finetune_all_nanhu_debug_v1/xsam_iter56_fp32/pytorch_model.bin"
DEFAULT_PTH_MODEL="/mnt_llm_A100_V1/shui/LAE/OneTerra-train/wkdrs_01/s3_mixed_fineture_base/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune_all_v1/pytorch_model.bin"
#DEFAULT_PTH_MODEL="/mnt_llm_A100_V1/shui/LAE/OneTerra-train/wkdrs/s3_mixed_fineture_base/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_ovseg_subset_ft_flat_v1/iter_30000.pth"
# 获取参数（checkpoint路径和输出目录都是可选的）
PTH_MODEL="${1:-$DEFAULT_PTH_MODEL}"
WORK_DIR="${2:-./eval_results_base_20260511/s3_mixed_finetune}"

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


# 运行评估（使用原始配置文件）
python xsam/xsam/tools/eval_ori.py \
    "$CONFIG" \
    --pth_model "$PTH_MODEL" \
    --work-dir "$WORK_DIR"

EVAL_EXIT_CODE=$?

if [ $EVAL_EXIT_CODE -ne 0 ]; then
    exit $EVAL_EXIT_CODE
fi

echo ""
echo "=========================================="
echo "任务评估完成！"
echo "结果保存在: $WORK_DIR"
