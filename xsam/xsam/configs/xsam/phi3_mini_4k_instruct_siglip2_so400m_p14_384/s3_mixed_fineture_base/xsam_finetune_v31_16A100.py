from copy import deepcopy
from os import getenv

import torch
from mmengine.hooks import CheckpointHook, DistSamplerSeedHook, IterTimerHook, LoggerHook, ParamSchedulerHook
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, SiglipProcessor, SiglipVisionModel
from xtuner.utils import PROMPT_TEMPLATE

from xsam.dataset import (
    ConcatDataset,
    GenSegDataset,
    ImgConvDataset,
    OVSegDataset,
    RefSegDataset,
    ReasonSegDataset,
)
from xsam.dataset.pano_seg_dataset import PanoSegDataset
from xsam.dataset.collate_fns import xsam_collate_fn
from xsam.dataset.map_fns import (
    dataset_map_fn_factory,
    template_map_fn_factory,
)
from xsam.dataset.map_fns.dataset_map_fns import image_conv_map_fn, generic_seg_map_fn, ov_seg_map_fn, refer_seg_map_fn, reason_seg_map_fn
from xsam.dataset.process_fns.postprocess_fns import (
    generic_seg_postprocess_fn,
    ov_seg_postprocess_fn,
    refer_seg_postprocess_fn,
    reason_seg_postprocess_fn,
)
from xsam.dataset.process_fns import process_map_fn_factory
from xsam.dataset.processors import SamImageProcessor
from xsam.dataset.samplers import SourceGroupedSampler
from xsam.engine.hooks import DatasetInfoHook, DistLossReduceHook, ModelInfoHook, PTCheckpointHook
from xsam.engine.runners.loops import TrainLoop
from peft import LoraConfig
from xsam.model import XSamModel
from xsam.model.segmentors import XSegmentor
from xsam.model.segmentors.mask2former import Mask2FormerConfig, Mask2FormerModel
from xsam.model.segmentors.sam import SamModel
import xsam.engine.runners.loops

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
base_root = "/mnt/si001883vtjl/"
#base_root = "/mnt_llm_A100_V1/"
code_dir = getenv("CODE_DIR", "./xsam/")
data_dir = getenv("DATA_DIR", "./datas/")
init_dir = getenv("INIT_DIR", "./inits/")
work_dir = getenv("WORK_DIR", "./checkpoints/")
# 预训练权重目录（s1/s2），与 work_dir 分离：work_dir 仅用于本次训练输出
checkpoint_dir = base_root + "shui/LAE/OneTerra-train/checkpoints/"

llm_name_or_path = init_dir + "Phi-3-mini-4k-instruct"
visual_encoder_name_or_path = init_dir + "siglip-so400m-patch14-384"
seg_encoder_name_or_path = init_dir + "sam-vit-large"
seg_decoder_name_or_path = init_dir + "mask2former-swin-large-coco-panoptic"

s1_pretrained_pth = checkpoint_dir + "s1_seg_finetune/pytorch_model.bin"
s2_pretrained_pth = checkpoint_dir + "xsam_s2_align_pretrain_skyscript_sar/iter_35874.pth"

# 恢复正确的 LoRA 参数
llm_lora_config = dict(
    type=LoraConfig,
    r=16,
    lora_alpha=32,
    target_modules=[
        "self_attn.qkv_proj",
        "self_attn.o_proj",
        "mlp.gate_up_proj",
        "mlp.down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

prompt_template = PROMPT_TEMPLATE.phi3_chat
max_length = int(4096 - (384 / 14) ** 2 - 1024)

# 🚀 32卡 A100 超参配置 (严格匹配平方根缩放法则)
batch_size = 2             
accumulative_counts = 4    
dataloader_num_workers = 2 
max_epochs = 2
optim_type = AdamW
lr = 4e-5                  # 🎯 回归基准学习率 (全局BS=256)
betas = (0.9, 0.999)
weight_decay = 0.05
max_norm = 1               
warmup_ratio = 0.03        

save_steps = 500
save_total_limit = 8  
logging_interval = 100

#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
special_tokens = ["<SEG>", "<p>", "</p>"]
cond_type = "phrase"
ignore_label = 255
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=llm_name_or_path,
    trust_remote_code=True,
    padding_side="right",
)

image_processor = dict(
    type=SiglipProcessor.from_pretrained,
    pretrained_model_name_or_path=visual_encoder_name_or_path,
    trust_remote_code=True,
)

extra_image_processor = dict(
    type=SamImageProcessor.from_pretrained,
    pretrained_model_name_or_path=seg_encoder_name_or_path,
    trust_remote_code=True,
    ignore_index=0,
)

model = dict(
    type=XSamModel,
    freeze_llm=False,
    freeze_visual_encoder=False,      # 🚀 解冻视觉主干
    freeze_segmentor_encoder=False,   # 🚀 解冻分割主干
    use_dual_encoder=True,
    use_vision_sampler=True,
    use_activation_checkpointing=True,
    connector_type="conv",
    cond_type=cond_type,
    seg_select_layers=[6, 12, 18, 24],
    connector_hidden_dim=512,
    connector_scale_factor=[4, 2, 1, 0.5],
    sampler_input_feat="extra_pixel_values",
    special_tokens=special_tokens,
    s1_pretrained_pth=s1_pretrained_pth,
    s2_pretrained_pth=s2_pretrained_pth,
    tokenizer=tokenizer,
    postprocess_fn=generic_seg_postprocess_fn,
    llm_lora=llm_lora_config,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=llm_name_or_path,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ),
    visual_encoder=dict(
        type=SiglipVisionModel.from_pretrained,
        pretrained_model_name_or_path=visual_encoder_name_or_path,
        torch_dtype=torch.bfloat16,
    ),
    segmentor=dict(
        type=XSegmentor,
        encoder=dict(
            type=SamModel.from_pretrained,
            pretrained_model_name_or_path=seg_encoder_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        ),
        decoder=dict(
            type=Mask2FormerModel._from_config,
            config=dict(
                type=Mask2FormerConfig.from_pretrained,
                pretrained_model_name_or_path=seg_decoder_name_or_path,
                use_backbone=False,
                feature_channels=[512, 1024, 2048],
                num_feature_levels=3,
                trust_remote_code=True,
            ),
            torch_dtype=torch.bfloat16,
        ),
        torch_dtype=torch.bfloat16,
        reinit_decoder=True, # 🚀 清洗记忆，重新学习开集
        open_cls=True,
    ),
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
pano_data_root = data_dir + "pano/"
imgconv_data_root = data_dir + "img_conv_data/"
oneterra_data_root = base_root + "/shui/oneterra_data/"
yangsen_data_root = base_root + "/yangsen/datasets/"
fitrs_data_root = oneterra_data_root + "imgconv/FIT-RS/raw_data/"
fitrs_imgconv_data_path = fitrs_data_root + "train_data_of_each_individual_task/"
fitrs_imgconv_image_folder = fitrs_data_root + "imgv2_split_512_100_vaild"
optical_caption_data_root = oneterra_data_root + "imgconv/image_caption/"
refseg_data_root = data_dir + "ref_seg_data/"

output_ids_with_output = True

train_extra_image_processor = deepcopy(extra_image_processor)
train_extra_image_processor.update(
    {
        "size": {"min_scale": 0.1, "max_scale": 2.0, "target_size": 1024},
        "do_crop": True,
        "crop_size": {"height": 1024, "width": 1024},
    }
)

# ================= 🔴 1. 背景字典与大盘问答（SAR/complexcompre 降压，geochat/VQA 保留） =================
geochat_imgconv_dataset = dict(
    type=ImgConvDataset, data_path=imgconv_data_root + "geochat/geochat_llava.json", tokenizer=tokenizer, cond_type=cond_type, special_tokens=special_tokens, image_folder=imgconv_data_root + "geochat/images", image_processor=image_processor, extra_image_processor=train_extra_image_processor, task_name="imgconv", data_name="geochat_imgconv", dataset_map_fn=dict(type=dataset_map_fn_factory, fn=image_conv_map_fn), template_map_fn=dict(type=template_map_fn_factory, template=prompt_template), max_length=max_length, pixel_values_ndim=2, is_multimodal=True, exclude_pure_text=True, pad_image_to_square=False, preprocess_text_data=True, 
    repeats_scale=1.0,
)  # 原~99,000 × 1.0 → ~99,000/epoch
sar_total_imgconv_dataset = dict(
    type=ImgConvDataset, data_path=yangsen_data_root + "sar_total/sft/train.json", tokenizer=tokenizer, cond_type=cond_type, special_tokens=special_tokens, image_folder=yangsen_data_root, image_processor=image_processor, extra_image_processor=train_extra_image_processor, task_name="imgconv", data_name="sar_total_imgconv", dataset_map_fn=dict(type=dataset_map_fn_factory, fn=image_conv_map_fn), template_map_fn=dict(type=template_map_fn_factory, template=prompt_template), max_length=max_length, pixel_values_ndim=2, is_multimodal=True, exclude_pure_text=True, pad_image_to_square=False, preprocess_text_data=True, 
    repeats_scale=0.3,
)  # 原~1,000,000 × 0.3 → ~300,000/epoch
fitrs_complexcompre_imgconv_dataset = dict(
    type=ImgConvDataset, data_path=fitrs_imgconv_data_path + "train_instruction_complexcompre_708k_cleaned.json", tokenizer=tokenizer, cond_type=cond_type, special_tokens=special_tokens, image_folder=fitrs_imgconv_image_folder, image_processor=image_processor, extra_image_processor=train_extra_image_processor, task_name="imgconv", data_name="fitrs_complexcompre_imgconv", dataset_map_fn=dict(type=dataset_map_fn_factory, fn=image_conv_map_fn), template_map_fn=dict(type=template_map_fn_factory, template=prompt_template), max_length=max_length, pixel_values_ndim=2, is_multimodal=True, exclude_pure_text=True, pad_image_to_square=False, preprocess_text_data=True, 
    repeats_scale=0.2,
)  # 原~708,000 × 0.2 → ~141,600/epoch
fitrs_vqa_imgconv_dataset = dict(
    type=ImgConvDataset, data_path=fitrs_imgconv_data_path + "train_instruction_vqa_400k_cleaned.json", tokenizer=tokenizer, cond_type=cond_type, special_tokens=special_tokens, image_folder=fitrs_imgconv_image_folder, image_processor=image_processor, extra_image_processor=train_extra_image_processor, task_name="imgconv", data_name="fitrs_vqa_imgconv", dataset_map_fn=dict(type=dataset_map_fn_factory, fn=image_conv_map_fn), template_map_fn=dict(type=template_map_fn_factory, template=prompt_template), max_length=max_length, pixel_values_ndim=2, is_multimodal=True, exclude_pure_text=True, pad_image_to_square=False, preprocess_text_data=True, 
    repeats_scale=0.5,
)  # 原~400,000 × 0.5 → ~200,000/epoch
fitrs_imageclassification_imgconv_dataset = dict(
    type=ImgConvDataset, data_path=fitrs_imgconv_data_path + "train_instruction_imageclassification_130k_cleaned.json", tokenizer=tokenizer, cond_type=cond_type, special_tokens=special_tokens, image_folder=fitrs_imgconv_image_folder, image_processor=image_processor, extra_image_processor=train_extra_image_processor, task_name="imgconv", data_name="fitrs_imageclassification_imgconv", dataset_map_fn=dict(type=dataset_map_fn_factory, fn=image_conv_map_fn), template_map_fn=dict(type=template_map_fn_factory, template=prompt_template), max_length=max_length, pixel_values_ndim=2, is_multimodal=True, exclude_pure_text=True, pad_image_to_square=False, preprocess_text_data=True, 
    repeats_scale=0.2,
)  # 原~130,000 × 0.2 → ~26,000/epoch
fitrs_multiturn_imgconv_dataset = dict(
    type=ImgConvDataset, data_path=fitrs_imgconv_data_path + "train_instruction_multiturn_50k_cleaned.json", tokenizer=tokenizer, cond_type=cond_type, special_tokens=special_tokens, image_folder=fitrs_imgconv_image_folder, image_processor=image_processor, extra_image_processor=train_extra_image_processor, task_name="imgconv", data_name="fitrs_multiturn_imgconv", dataset_map_fn=dict(type=dataset_map_fn_factory, fn=image_conv_map_fn), template_map_fn=dict(type=template_map_fn_factory, template=prompt_template), max_length=max_length, pixel_values_ndim=2, is_multimodal=True, exclude_pure_text=True, pad_image_to_square=False, preprocess_text_data=True, 
    repeats_scale=0.5,
)  # 原~50,000 × 0.5 → ~25,000/epoch

# ================= 🟢 2. Caption / VQA 打榜（小集提权，caption 合计 ~37.5万/epoch） =================
fitrs_imagecaption_imgconv_dataset = dict(
    type=ImgConvDataset, data_path=fitrs_imgconv_data_path + "train_instruction_imagecaption_65k_cleaned.json", tokenizer=tokenizer, cond_type=cond_type, special_tokens=special_tokens, image_folder=fitrs_imgconv_image_folder, image_processor=image_processor, extra_image_processor=train_extra_image_processor, task_name="imgconv", data_name="fitrs_imagecaption_imgconv", dataset_map_fn=dict(type=dataset_map_fn_factory, fn=image_conv_map_fn), template_map_fn=dict(type=template_map_fn_factory, template=prompt_template), max_length=max_length, pixel_values_ndim=2, is_multimodal=True, exclude_pure_text=True, pad_image_to_square=False, preprocess_text_data=True, 
    repeats_scale=2.5,
)  # 原~65,000 × 2.5 → ~162,500/epoch
fitrs_regioncaption_imgconv_dataset = dict(
    type=ImgConvDataset, data_path=fitrs_imgconv_data_path + "train_instruction_regioncaption_72k_cleaned.json", tokenizer=tokenizer, cond_type=cond_type, special_tokens=special_tokens, image_folder=fitrs_imgconv_image_folder, image_processor=image_processor, extra_image_processor=train_extra_image_processor, task_name="imgconv", data_name="fitrs_regioncaption_imgconv", dataset_map_fn=dict(type=dataset_map_fn_factory, fn=image_conv_map_fn), template_map_fn=dict(type=template_map_fn_factory, template=prompt_template), max_length=max_length, pixel_values_ndim=2, is_multimodal=True, exclude_pure_text=True, pad_image_to_square=False, preprocess_text_data=True, 
    repeats_scale=1.0,
)  # 原~72,000 × 1.0 → ~72,000/epoch
ucm_captions_imgconv_dataset = dict(
    type=ImgConvDataset, data_path=optical_caption_data_root + "UCM-Captions/dataset_qwenvl.json", tokenizer=tokenizer, cond_type=cond_type, special_tokens=special_tokens, image_folder=optical_caption_data_root + "UCM-Captions/imgs", image_processor=image_processor, extra_image_processor=train_extra_image_processor, task_name="imgconv", data_name="imgconv_UCM-Captions", dataset_map_fn=dict(type=dataset_map_fn_factory, fn=image_conv_map_fn), template_map_fn=dict(type=template_map_fn_factory, template=prompt_template), max_length=max_length, pixel_values_ndim=2, is_multimodal=True, exclude_pure_text=True, pad_image_to_square=False, preprocess_text_data=True, 
    repeats_scale=20.0,
)  # 原~2,000 × 20.0 → ~40,000/epoch
nwpu_captions_imgconv_dataset = dict(
    type=ImgConvDataset, data_path=optical_caption_data_root + "NWPU-Captions/dataset_nwpu_qwenvl_cleaned.json", tokenizer=tokenizer, cond_type=cond_type, special_tokens=special_tokens, image_folder=optical_caption_data_root + "NWPU-Captions/NWPU_images", image_processor=image_processor, extra_image_processor=train_extra_image_processor, task_name="imgconv", data_name="imgconv_NWPU-Captions", dataset_map_fn=dict(type=dataset_map_fn_factory, fn=image_conv_map_fn), template_map_fn=dict(type=template_map_fn_factory, template=prompt_template), max_length=max_length, pixel_values_ndim=2, is_multimodal=True, exclude_pure_text=True, pad_image_to_square=False, preprocess_text_data=True, 
    repeats_scale=4.0,
)  # 原~25,000 × 4.0 → ~100,000/epoch
rsvqa_lr_imgconv_dataset = dict(
    type=ImgConvDataset, data_path=oneterra_data_root + "imgconv/VQA/RSVQA-LR/train_cleaned.json", tokenizer=tokenizer, cond_type=cond_type, special_tokens=special_tokens, image_folder=oneterra_data_root + "imgconv/VQA/RSVQA-LR/Images_LR", image_processor=image_processor, extra_image_processor=train_extra_image_processor, task_name="imgconv", data_name="imgconv_RSVQA_LR", dataset_map_fn=dict(type=dataset_map_fn_factory, fn=image_conv_map_fn), template_map_fn=dict(type=template_map_fn_factory, template=prompt_template), max_length=max_length, pixel_values_ndim=2, is_multimodal=True, exclude_pure_text=True, pad_image_to_square=False, preprocess_text_data=True,
    repeats_scale=1.0,
)  # 原~57,000 × 1.0 → ~57,000/epoch

# ================= 🟣 3. 全景分割（genseg:ovseg = 2:3，OVSeg 开集采样已修复） =================
pano_genseg_dataset = dict(
    type=PanoSegDataset, data_path=pano_data_root + "annotations_train.json", image_folder=pano_data_root + "train/images", panseg_map_folder=pano_data_root + "train/panoptic_labels", tokenizer=tokenizer, task_name="genseg", data_name="pano_panoptic_genseg_train", cond_type=cond_type, special_tokens=special_tokens, extra_image_processor=train_extra_image_processor, image_processor=image_processor, dataset_map_fn=dict(type=dataset_map_fn_factory, fn=generic_seg_map_fn, cond_type=cond_type), template_map_fn=dict(type=template_map_fn_factory, template=prompt_template), max_length=max_length, use_variant_cat=True, pad_image_to_square=False, 
    repeats_scale=2.0,
)  # 原~110,000 × 2.0 → ~220,000/epoch
pano_ovseg_dataset = dict(
    type=OVSegDataset, data_path=pano_data_root + "annotations_train.json", image_folder=pano_data_root + "train/images", panseg_map_folder=pano_data_root + "train/panoptic_labels", tokenizer=tokenizer, task_name="ovseg", data_name="pano_ovseg_train", cond_type=cond_type, special_tokens=special_tokens, image_processor=image_processor, extra_image_processor=train_extra_image_processor, dataset_map_fn=dict(type=dataset_map_fn_factory, fn=ov_seg_map_fn, cond_type=cond_type), template_map_fn=dict(type=template_map_fn_factory, template=prompt_template), max_length=max_length, pad_image_to_square=False, 
    use_variant_cat=True, use_full_cat=True, sample_num=32,
    repeats_scale=3.0,
)  # 原~110,000 × 3.0 → ~330,000/epoch（use_full_cat + sample_num=32 子集/全类混合）

# 指代分割 refseg（合计 ~454,000/epoch）
remotesam_refseg_dataset = dict(
    type=RefSegDataset, data_root=refseg_data_root, image_folder=refseg_data_root + "images/remotesam_images", dataset="remotesam", data_split="train", tokenizer=tokenizer, task_name="refseg", data_name="remotesam_train_refseg", cond_type=cond_type, special_tokens=special_tokens, extra_image_processor=train_extra_image_processor, image_processor=image_processor, postprocess_fn=refer_seg_postprocess_fn, dataset_map_fn=dict(type=dataset_map_fn_factory, fn=refer_seg_map_fn, cond_type=cond_type), template_map_fn=dict(type=template_map_fn_factory, template=prompt_template), use_variant_cat=True, use_random_cat=True, max_length=max_length, pad_image_to_square=False, ignore_label=ignore_label, 
    repeats_scale=1.0,
)  # 原~210,000 × 1.0 → ~210,000/epoch
fast_refseg_dataset = dict(
    type=RefSegDataset, data_root=oneterra_data_root + "refseg/FAST/fast", image_folder=oneterra_data_root + "refseg/FAST/images", dataset="fast", data_split="train", tokenizer=tokenizer, task_name="refseg", data_name="refseg_fast_train", cond_type=cond_type, special_tokens=special_tokens, extra_image_processor=train_extra_image_processor, image_processor=image_processor, postprocess_fn=refer_seg_postprocess_fn, dataset_map_fn=dict(type=dataset_map_fn_factory, fn=refer_seg_map_fn, cond_type=cond_type), template_map_fn=dict(type=template_map_fn_factory, template=prompt_template), use_variant_cat=True, use_random_cat=True, max_length=max_length, pad_image_to_square=False, ignore_label=ignore_label, 
    repeats_scale=1.0,
)  # 原~120,000 × 1.0 → ~120,000/epoch
risbench_refseg_dataset = dict(
    type=RefSegDataset, data_root=oneterra_data_root + "refseg/RISBench", image_folder=oneterra_data_root + "refseg/RISBench/RISBench_dataset/img_rgb", dataset="risbench", data_split="train", tokenizer=tokenizer, task_name="refseg", data_name="refseg_risbench_train", cond_type=cond_type, special_tokens=special_tokens, extra_image_processor=train_extra_image_processor, image_processor=image_processor, postprocess_fn=refer_seg_postprocess_fn, dataset_map_fn=dict(type=dataset_map_fn_factory, fn=refer_seg_map_fn, cond_type=cond_type), template_map_fn=dict(type=template_map_fn_factory, template=prompt_template), use_variant_cat=True, use_random_cat=True, max_length=max_length, pad_image_to_square=False, ignore_label=ignore_label, 
    repeats_scale=2.0, sample_num=8,
)  # 原~52,000 × 2.0 → ~104,000/epoch
rrsisd_refseg_dataset = dict(
    type=RefSegDataset, data_root=oneterra_data_root + "refseg/RRSIS-D", image_folder=oneterra_data_root + "refseg/RRSIS-D/images/rrsisd/JPEGImages", dataset="rrsisd", data_split="train", tokenizer=tokenizer, task_name="refseg", data_name="refseg_rrsisd_train", cond_type=cond_type, special_tokens=special_tokens, extra_image_processor=train_extra_image_processor, image_processor=image_processor, postprocess_fn=refer_seg_postprocess_fn, dataset_map_fn=dict(type=dataset_map_fn_factory, fn=refer_seg_map_fn, cond_type=cond_type), template_map_fn=dict(type=template_map_fn_factory, template=prompt_template), use_variant_cat=True, use_random_cat=True, max_length=max_length, pad_image_to_square=False, ignore_label=ignore_label, 
    repeats_scale=2.0, sample_num=8,
)  # 原~10,000 × 2.0 → ~20,000/epoch

# 推理分割 reaseg（earthreason 小集提权，diy1 全量；合计 ~130,000/epoch）
earthreason_reaseg_dataset = dict(
    type=ReasonSegDataset, data_root=oneterra_data_root + "reasonseg/EarthReason_convert", image_folder=oneterra_data_root + "reasonseg/EarthReason_convert/train", explain_path=oneterra_data_root + "reasonseg/EarthReason_convert/explanatory/train.json", data_split="train", tokenizer=tokenizer, task_name="reaseg", data_name="reaseg_earthreason_train", cond_type=cond_type, special_tokens=special_tokens, extra_image_processor=train_extra_image_processor, image_processor=image_processor, postprocess_fn=reason_seg_postprocess_fn, dataset_map_fn=dict(type=dataset_map_fn_factory, fn=reason_seg_map_fn, cond_type=cond_type), template_map_fn=dict(type=template_map_fn_factory, template=prompt_template), use_variant_cat=True, use_random_cat=True, max_length=max_length, pad_image_to_square=False, ignore_label=ignore_label, use_threads=True, 
    repeats_scale=10.0,
)  # 原~2,000 × 10.0 → ~20,000/epoch
diy1_reaseg_dataset = dict(
    type=ReasonSegDataset, data_root=oneterra_data_root + "reasonseg/diy1", image_folder=oneterra_data_root + "reasonseg/diy1/train", data_split="train", tokenizer=tokenizer, task_name="reaseg", data_name="reaseg_diy1_train", cond_type=cond_type, special_tokens=special_tokens, extra_image_processor=train_extra_image_processor, image_processor=image_processor, postprocess_fn=reason_seg_postprocess_fn, dataset_map_fn=dict(type=dataset_map_fn_factory, fn=reason_seg_map_fn, cond_type=cond_type), template_map_fn=dict(type=template_map_fn_factory, template=prompt_template), use_variant_cat=True, use_random_cat=True, max_length=max_length, pad_image_to_square=False, ignore_label=ignore_label, use_threads=True, 
    repeats_scale=1.0,
)  # 原~110,000 × 1.0 → ~110,000/epoch

# 全任务配比估算：imgconv ~122万 (52%) | 分割 ~113万 (48%) | 合计 ~235万/epoch
combined_train_dataset = dict(
    type=ConcatDataset,
    oversample_ratio=0.0,
    datasets=[
        geochat_imgconv_dataset,
        sar_total_imgconv_dataset,
        fitrs_complexcompre_imgconv_dataset,
        fitrs_vqa_imgconv_dataset,
        fitrs_imagecaption_imgconv_dataset,
        fitrs_imageclassification_imgconv_dataset,
        fitrs_multiturn_imgconv_dataset,
        fitrs_regioncaption_imgconv_dataset,
        rsvqa_lr_imgconv_dataset,
        ucm_captions_imgconv_dataset,
        nwpu_captions_imgconv_dataset,
        pano_genseg_dataset,
        pano_ovseg_dataset,
        fast_refseg_dataset,
        remotesam_refseg_dataset,
        risbench_refseg_dataset,
        rrsisd_refseg_dataset,
        earthreason_reaseg_dataset,
        diy1_reaseg_dataset,
    ],
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    pin_memory=True,
    persistent_workers=False,
    dataset=combined_train_dataset,
    sampler=dict(
        type=SourceGroupedSampler,
        length_property="source_length",
        mega_batch_mult=1,
        per_device_batch_size=batch_size * accumulative_counts,
    ),
    collate_fn=dict(type=xsam_collate_fn),
)

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale="dynamic",
    dtype="float16",
    paramwise_cfg=dict(
        bypass_duplicate=True,
        custom_keys={
            # 🛡️ 绝对核心：主干保护伞，防止SOTA底座遗忘 (实际 lr = 4e-6)
            "segmentor.encoder": dict(lr_mult=0.1, decay_mult=1.0),
            "visual_encoder": dict(lr_mult=0.1, decay_mult=1.0),
        },
    ),
)

param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True,
    ),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True,
    ),
]

train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
custom_hooks = [
    dict(type=ModelInfoHook, module_names=["llm", "visual_encoder", "projector", "connector", "segmentor"], display_params=True),
    dict(type=DatasetInfoHook, tokenizer=tokenizer, special_tokens=special_tokens),
    dict(type=PTCheckpointHook, clean_pth=False),
]

default_hooks = dict(
    dist_loss_reduce=dict(type=DistLossReduceHook),
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=logging_interval),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(type=CheckpointHook, by_epoch=False, interval=save_steps, max_keep_ckpts=save_total_limit),
    sampler_seed=dict(type=DistSamplerSeedHook),
)

env_cfg = dict(cudnn_benchmark=False, mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0), dist_cfg=dict(backend="nccl"))

log_level = "INFO"

# 🚀 坚决切断旧权重，纯净加载打地基
load_from = None
resume = False

randomness = dict(seed=None, deterministic=False)
log_processor = dict(by_epoch=False, window_size=1, mean_pattern=r".*(loss|time|data_time|grad_norm|tflops).*")