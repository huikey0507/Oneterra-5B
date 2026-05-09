#!/bin/bash

# X-SAM 评估脚本 - 仅评测 ovseg

CONFIG="/mnt_llm_A100_V1/shui/LAE/RS-Xsam-main-old/xsam/xsam/configs/xsam/phi3_mini_4k_instruct_siglip2_so400m_p14_384/s3_mixed_fineture_base/xsam_base_mixed_finetune_ovseg_only.py"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEFAULT_PTH_MODEL="/mnt_llm_A100_V1/shui/LAE/OneTerra-train/wkdrs_01/s3_mixed_fineture_base/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune_all_v1/pytorch_model.bin"
PTH_MODEL="${1:-$DEFAULT_PTH_MODEL}"
WORK_DIR="${2:-./eval_results_base_20260319/s3_mixed_finetune_ovseg_only}"

if [ ! -f "$PTH_MODEL" ] && [ ! -d "$PTH_MODEL" ]; then
    echo "错误: checkpoint路径不存在: $PTH_MODEL"
    exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "错误: 未找到 Python 可执行文件: $PYTHON_BIN"
    echo "请先激活正确环境，或通过 PYTHON_BIN 指定路径。"
    exit 1
fi

if ! "$PYTHON_BIN" -c "import mmcv, terminaltables" >/dev/null 2>&1; then
    echo "错误: 当前 Python 环境缺少依赖(mmcv 或 terminaltables)"
    echo "请在当前环境执行:"
    echo "  $PYTHON_BIN -m pip install -U openmim"
    echo "  $PYTHON_BIN -m mim install mmcv"
    echo "  $PYTHON_BIN -m pip install terminaltables"
    exit 1
fi

echo "=========================================="
echo "X-SAM ovseg 评估脚本"
echo "=========================================="
echo "配置文件: $CONFIG"
echo "Checkpoint: $PTH_MODEL"
echo "输出目录: $WORK_DIR"
echo "=========================================="
echo ""

# 读取配置中的 ovseg 标注路径并做类别映射自检
OVSEG_ANN_PATH=$("$PYTHON_BIN" - <<'PY'
from mmengine.config import Config
cfg = Config.fromfile("/mnt_llm_A100_V1/shui/LAE/RS-Xsam-main-old/xsam/xsam/configs/xsam/phi3_mini_4k_instruct_siglip2_so400m_p14_384/s3_mixed_fineture_base/xsam_base_mixed_finetune_ovseg_only.py")
target = None
for ds in cfg.val_datasets:
    if "ovseg" in ds.get("data_name", ""):
        target = ds.get("data_path")
        break
print(target or "")
PY
)

if [ -z "$OVSEG_ANN_PATH" ] || [ ! -f "$OVSEG_ANN_PATH" ]; then
    echo "错误: 无法找到 ovseg 标注文件: $OVSEG_ANN_PATH"
    exit 1
fi

echo "[自检] ovseg 标注文件: $OVSEG_ANN_PATH"
"$PYTHON_BIN" - "$OVSEG_ANN_PATH" <<'PY'
import json
import sys

ann_path = sys.argv[1]
with open(ann_path, "r", encoding="utf-8") as f:
    data = json.load(f)

cats = data.get("categories", [])
cat_ids = [c.get("id") for c in cats]
uniq_ids = set(cat_ids)
has_isthing = all("isthing" in c for c in cats)

print(f"[自检] categories: {len(cats)}")
print(f"[自检] unique ids: {len(uniq_ids)}")
print(f"[自检] id unique: {len(cat_ids) == len(uniq_ids)}")
print(f"[自检] has isthing: {has_isthing}")
if cats:
    preview = ", ".join(f'{c.get("id")}:{c.get("name")}' for c in cats[:8])
    print(f"[自检] first categories: {preview}")
PY

echo ""
echo "开始评估 ovseg..."
"$PYTHON_BIN" xsam/xsam/tools/eval_ori.py \
    "$CONFIG" \
    --pth_model "$PTH_MODEL" \
    --work-dir "$WORK_DIR"

EVAL_EXIT_CODE=$?
if [ $EVAL_EXIT_CODE -ne 0 ]; then
    exit $EVAL_EXIT_CODE
fi

echo ""
echo "=========================================="
echo "ovseg 任务评估完成！"
echo "结果保存在: $WORK_DIR"
