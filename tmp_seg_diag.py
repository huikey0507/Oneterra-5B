"""临时诊断脚本：检查 segmentation 原始输出内容。"""
import os, sys, warnings, json
os.environ.update({
    "CUDA_VISIBLE_DEVICES": "0",
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_OFFLINE": "1",
    "TRANSFORMERS_VERBOSITY": "error",
})
warnings.filterwarnings("ignore")
sys.path.insert(0, "/mnt_llm_A100_V1/shui/LAE/XSAM-public/RS-Xsam-main/xsam")
sys.path.insert(0, "/mnt_llm_A100_V1/shui/LAE/XSAM-public/RS-Xsam-main")
os.chdir("/mnt_llm_A100_V1/shui/LAE/XSAM-public/RS-Xsam-main")

import numpy as np
import torch
from PIL import Image
from mmengine.config import Config
from mmengine.runner.utils import set_random_seed

set_random_seed(42)
cfg = Config.fromfile(
    "xsam/xsam/configs/xsam/phi3_mini_4k_instruct_siglip2_so400m_p14_384"
    "/s3_mixed_fineture_base/xsam_base_mixed_finetune_all.py"
)
from xsam.demo.demo import XSamDemo
demo = XSamDemo(
    cfg=cfg,
    pth_model=(
        "checkpoints/OneTerraV2-5B/s3_mixed_fineture_base/"
        "xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune_all_v1/"
        "iter_40000.pth/pytorch_model.bin"
    ),
    output_ids_with_output=False,
    max_length=4096,
    cond_type="phrase",
    pad_image_to_square=True,
)
print("MODEL_LOADED", flush=True)

img = Image.open("test_images/bridge_144_GSD_2.png").convert("RGB")

# patch model forward 仅捕获 seg_outputs
_cap = {}
_real = demo.model.__class__.__call__

def _hook(self, data_dict, data_samples=None, mode=None, **kw):
    lo, so = _real(self, data_dict, data_samples, mode=mode, **kw)
    _cap["so"] = so
    return lo, so

demo.model.__class__.__call__ = _hook

for task, prompt in [
    ("ovseg",  "stuff: road, water, building, vegetation, bridge"),
    ("refseg", "the bridge"),
]:
    print(f"\n=== {task} ===", flush=True)
    _cap.clear()
    _, text, seg_vis = demo.run_on_image(img, prompt, task)
    so = _cap.get("so")
    if so is None:
        print("  seg_out=None", flush=True)
    elif isinstance(so, (list, tuple)) and len(so) > 0:
        r0 = so[0]
        if isinstance(r0, dict):
            seg = r0.get("segmentation")
            si  = r0.get("segments_info")
            if seg is not None:
                arr = np.array(seg)
                u, c = np.unique(arr, return_counts=True)
                print(f"  seg shape={arr.shape} dtype={arr.dtype}", flush=True)
                print(f"  unique_ids={u.tolist()}", flush=True)
                print(f"  counts={c.tolist()}", flush=True)
                print(f"  fg_coverage={100*(arr.size-c[0])/arr.size:.2f}%", flush=True)
            if si is not None:
                if isinstance(si, list):
                    print(f"  segments_info: {len(si)} items", flush=True)
                    print(f"  first 3: {si[:3]}", flush=True)
                elif isinstance(si, dict):
                    print(f"  segments_info (dict): keys={list(si.keys())}, score={si.get('score')}", flush=True)
        else:
            print(f"  r0 type={type(r0)}", flush=True)
    else:
        print(f"  seg_out type={type(so)}", flush=True)

demo.model.__class__.__call__ = _real
print("\nDONE", flush=True)
