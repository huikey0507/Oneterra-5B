#!/bin/bash

# 独立 HTTP API（oneterra_api/server.py），不改动 xsam/xsam/demo 下的 Gradio / 评测。
# 默认端口 7871。不要和 Gradio demo 同时在同一张卡上各加载一份模型。

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODE_DIR="$ROOT_DIR/xsam"
export PYTHONPATH="$CODE_DIR:${PYTHONPATH:-}"

CONFIG="./xsam/xsam/configs/xsam/phi3_mini_4k_instruct_siglip2_so400m_p14_384/s3_mixed_fineture_base/xsam_base_mixed_finetune_all.py"
DEFAULT_PTH_MODEL="/mnt_llm_A100_V1/shui/iter_44000.pth"

PTH_MODEL="${1:-$DEFAULT_PTH_MODEL}"
WORK_DIR="${2:-./api_work_021}"
PORT="${3:-7871}"
HOST="${4:-0.0.0.0}"
LOG_DIR="${5:-./api_logs_021}"

if [ ! -f "$PTH_MODEL" ] && [ ! -d "$PTH_MODEL" ]; then
    echo "错误: checkpoint路径不存在: $PTH_MODEL"
    exit 1
fi

if ! python - <<'PY'
import importlib.util
missing = [m for m in ("fastapi", "uvicorn") if importlib.util.find_spec(m) is None]
raise SystemExit(1 if missing else 0)
PY
then
    echo "未检测到 fastapi/uvicorn，正在安装..."
    pip install "fastapi>=0.110" "uvicorn[standard]>=0.27" "python-multipart"
fi

echo "=========================================="
echo "OneTerra-5B 独立 HTTP API (oneterra_api)"
echo "=========================================="
echo "配置文件: $CONFIG"
echo "Checkpoint: $PTH_MODEL"
echo "work-dir: $WORK_DIR"
echo "端口: $PORT"
echo "地址: $HOST"
echo "日志目录: $LOG_DIR"
echo "文档: http://${HOST}:${PORT}/docs"
echo "=========================================="
echo ""
echo "示例:"
echo "  curl -s http://127.0.0.1:${PORT}/health"
echo "  curl -s -F task=imgconv -F prompt='Can you describe this image briefly?' -F image=@/path/to.jpg http://127.0.0.1:${PORT}/v1/infer"
echo "  # ovseg 勿用 -F prompt='thing:...; stuff:...'（curl 会截断分号），请拆字段："
echo "  curl -s -F task=ovseg -F thing='car, building' -F stuff='road, grassland' -F image=@/path/to.jpg http://127.0.0.1:${PORT}/v1/infer"
echo "  curl -s -F task=refseg -F prompt='the road on the bottom right' -F image=@/path/to.jpg http://127.0.0.1:${PORT}/v1/infer"
echo ""

cd "$ROOT_DIR"
python oneterra_api/server.py \
    "$CONFIG" \
    --pth_model "$PTH_MODEL" \
    --work-dir "$WORK_DIR" \
    --seed 0 \
    --port "$PORT" \
    --host "$HOST" \
    --log-dir "$LOG_DIR"
