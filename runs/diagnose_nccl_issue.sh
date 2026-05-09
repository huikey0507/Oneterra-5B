#!/usr/bin/env bash
# NCCL通信问题诊断脚本

echo "=========================================="
echo "NCCL 通信问题诊断"
echo "=========================================="

echo ""
echo "1. 检查GPU状态..."
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader

echo ""
echo "2. 检查GPU拓扑..."
nvidia-smi topo -m

echo ""
echo "3. 检查共享内存..."
df -h /dev/shm
echo "建议：如果小于16GB，可能需要增加共享内存"
echo "增加方法：sudo mount -o remount,size=32G /dev/shm"

echo ""
echo "4. 检查系统内存..."
free -h

echo ""
echo "5. 检查NCCL环境变量..."
echo "NCCL_TIMEOUT: ${NCCL_TIMEOUT:-未设置}"
echo "DIST_TIMEOUT: ${DIST_TIMEOUT:-未设置}"
echo "TORCH_DISTRIBUTED_DEFAULT_TIMEOUT: ${TORCH_DISTRIBUTED_DEFAULT_TIMEOUT:-未设置}"
echo "NCCL_DEBUG: ${NCCL_DEBUG:-未设置}"
echo "NCCL_P2P_DISABLE: ${NCCL_P2P_DISABLE:-未设置}"
echo "NCCL_SHM_DISABLE: ${NCCL_SHM_DISABLE:-未设置}"
echo "NCCL_SHM_SIZE: ${NCCL_SHM_SIZE:-未设置}"

echo ""
echo "6. 检查是否有残留进程..."
ps aux | grep -E "train.py|torchrun|python.*xsam" | grep -v grep || echo "没有发现残留进程"

echo ""
echo "7. 检查NCCL版本..."
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'NCCL版本: {torch.cuda.nccl.version() if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null || echo "无法检查NCCL版本"

echo ""
echo "8. 检查网络接口..."
ip addr show | grep -E "^[0-9]+:|inet " | head -10

echo ""
echo "9. 测试GPU间通信（需要CUDA）..."
if command -v nvidia-smi >/dev/null 2>&1; then
    python3 << 'EOF'
import torch
if torch.cuda.is_available():
    print(f"CUDA可用，GPU数量: {torch.cuda.device_count()}")
    # 测试简单的GPU操作
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        x = torch.randn(1000, 1000).cuda()
        y = x @ x
        print(f"GPU {i}: 测试通过")
else:
    print("CUDA不可用")
EOF
fi

echo ""
echo "=========================================="
echo "诊断完成"
echo "=========================================="
echo ""
echo "建议的修复步骤："
echo "1. 如果共享内存不足，增加共享内存："
echo "   sudo mount -o remount,size=32G /dev/shm"
echo ""
echo "2. 如果某个GPU卡住，尝试减少GPU数量测试："
echo "   CUDA_VISIBLE_DEVICES=0,1 bash runs/run_official.sh ..."
echo ""
echo "3. 启用NCCL详细调试："
echo "   export NCCL_DEBUG=INFO"
echo ""
echo "4. 如果问题持续，尝试禁用P2P："
echo "   export NCCL_P2P_DISABLE=1"

