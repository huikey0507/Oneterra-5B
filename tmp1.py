import torch

sd = torch.load("checkpoints_021/s3_mixed_fineture_base/xsam_021_siglip2_so400m_p14_384_sam_large_m2f_mixed_finetune_all_nanhu_debug_v1/pytorch_model.bin", map_location="cpu")
# print(type(sd))
# print(list(sd.keys())[:50])

for k in sd.keys():
    if  "llm" in k.lower():
        print(k)