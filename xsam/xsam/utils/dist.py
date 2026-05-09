import argparse
import os
from datetime import timedelta
from typing import Tuple

import torch
import torch.distributed as dist

# 在导入 mmengine 之前就进行 patch，确保超时设置生效
# 读取环境变量设置超时
_timeout_seconds = int(os.environ.get("NCCL_TIMEOUT", os.environ.get("DIST_TIMEOUT", 36000)))
_timeout = timedelta(seconds=_timeout_seconds)

# 确保环境变量已设置
os.environ["NCCL_TIMEOUT"] = str(_timeout_seconds)
os.environ["DIST_TIMEOUT"] = str(_timeout_seconds)

# 在导入 mmengine 之前就 patch init_process_group
if not hasattr(dist, '_xsam_original_init_process_group'):
    _original_init_process_group = dist.init_process_group
    
    def _patched_init_process_group(*args_patch, **kwargs_patch):
        # 确保 timeout 被设置
        if 'timeout' not in kwargs_patch:
            kwargs_patch['timeout'] = _timeout
        # 添加调试信息
        import sys
        if os.environ.get("LOCAL_RANK", "0") == "0":
            print(f"[X-SAM] init_process_group called with timeout={kwargs_patch.get('timeout', 'NOT SET')}", file=sys.stderr)
        return _original_init_process_group(*args_patch, **kwargs_patch)
    
    dist._xsam_original_init_process_group = _original_init_process_group
    dist.init_process_group = _patched_init_process_group

from mmengine.dist import get_dist_info, init_dist
from mmengine.utils.dl_utils import set_multi_processing

from xsam.utils.logging import print_log


def setup_distributed(args: argparse.Namespace) -> Tuple[int, int, int]:
    """Setup distributed training environment."""
    if args.launcher != "none":
        set_multi_processing(distributed=True)
        
        # Set timeout for distributed operations
        # Read from environment variable or use default (10 hours)
        timeout_seconds = int(os.environ.get("NCCL_TIMEOUT", os.environ.get("DIST_TIMEOUT", 36000)))
        timeout = timedelta(seconds=timeout_seconds)
        
        # Ensure environment variables are set
        os.environ["NCCL_TIMEOUT"] = str(timeout_seconds)
        os.environ["DIST_TIMEOUT"] = str(timeout_seconds)
        
        # 验证 patch 是否生效
        if hasattr(dist, '_xsam_original_init_process_group'):
            print_log(f"init_process_group has been patched for timeout={timeout_seconds}s", logger="current")
        else:
            print_log("WARNING: init_process_group patch not found!", logger="current", level="WARNING")
        
        # Call mmengine's init_dist (it will use torch.distributed internally)
        # Our monkey-patch (done at module import) ensures timeout is passed to init_process_group
        init_dist(args.launcher)
        
        rank, world_size = get_dist_info()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        print_log(f"Rank: {rank} / Local rank: {local_rank} / World size: {world_size}", logger="current")
        print_log(f"NCCL timeout set to: {timeout_seconds} seconds ({timeout_seconds/3600:.1f} hours)", logger="current")
        
        # 验证实际超时设置（仅用于调试）
        if os.environ.get("NCCL_DEBUG", "").upper() in ["INFO", "DEBUG"]:
            try:
                # 尝试获取当前的 process group 超时（如果可能）
                if dist.is_initialized():
                    pg = dist.group.WORLD
                    if pg is not None:
                        print_log(f"Process group initialized successfully", logger="current")
            except Exception as e:
                print_log(f"Could not verify process group: {e}", logger="current", level="WARNING")
    else:
        rank = local_rank = 0
        world_size = 1
        print_log(f"Rank: {rank} / Local rank: {local_rank} / World size: {world_size} (single GPU, no distributed)", logger="current")

    return rank, local_rank, world_size
