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
    # generic_seg_map_fn,
    # imgconv_map_fn,
    # ovseg_map_fn,
    # refer_seg_map_fn,
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
from xsam.engine.hooks import DatasetInfoHook, DistLossReduceHook, EvaluateChatHook, ModelInfoHook, PTCheckpointHook
from xsam.engine.runners.loops import TrainLoop
from xsam.evaluation.evaluators import (
    GenSegEvaluator,
    ImgConvEvaluator,
    OVSegEvaluator,
    RefSegEvaluator,
    ReasonSegEvaluator,
)
from peft import LoraConfig
from xsam.model import XSamModel
from xsam.model.segmentors import XSegmentor
from xsam.model.segmentors.mask2former import Mask2FormerConfig, Mask2FormerModel
from xsam.model.segmentors.sam import SamModel
from xsam.utils.visualize import Visualizer
import xsam.engine.runners.loops

# 独立配置（无 read_base / 无 _base_）：由 mixed_finetune_all + OVSeg 子集微调覆盖合并而成。
# 训练命令请将 --config 指向本文件（勿用 xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_ovseg_openvocab_subset_ft.py）。

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Directories
base_root = "/mnt_llm_A100_V1/"
code_dir = getenv("CODE_DIR", "./xsam/")
data_dir = getenv("DATA_DIR", "./datas/")
init_dir = getenv("INIT_DIR", "./inits/")
work_dir = getenv(
    "WORK_DIR",
    "/mnt_llm_A100_V1/shui/LAE/OneTerra-train/checkpoints/xsam_s3_ovseg_openvocab_subset_ft",
)
# 预训练权重目录（s1/s2 检查点），与 work_dir 分离：work_dir 仅用于本次训练输出
checkpoint_dir = base_root + "/shui/LAE/OneTerra-train/checkpoints/"

# Model
llm_name_or_path = init_dir + "Phi-3-mini-4k-instruct"
visual_encoder_name_or_path = init_dir + "siglip-so400m-patch14-384"
seg_encoder_name_or_path = init_dir + "sam-vit-large"
seg_decoder_name_or_path = init_dir + "mask2former-swin-large-coco-panoptic"

# Specify the pretrained pth（从 checkpoint_dir 读，不随 WORK_DIR 变）
s1_pretrained_pth = checkpoint_dir + "s1_seg_finetune/pytorch_model.bin"
s2_pretrained_pth = (
    checkpoint_dir
    + "xsam_s2_align_pretrain_skyscript_sar/iter_35874.pth"
)  # noqa: E501

# LoRA配置 - 大幅减少LLM显存占用（从45.6GB降至0.31GB）
llm_lora_config = dict(
    type=LoraConfig,
    r=16,  # LoRA rank，可以调整为8/16/32/64，越大效果越好但显存占用更多
    lora_alpha=32,  # LoRA alpha，通常是r的2倍
    target_modules=[
        "self_attn.qkv_proj",  # Phi-3的attention层
        "self_attn.o_proj",
        "mlp.gate_up_proj",    # Phi-3的MLP层
        "mlp.down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Prompt
prompt_template = PROMPT_TEMPLATE.phi3_chat
max_length = int(4096 - (384 / 14) ** 2 - 1024)

# Scheduler & Optimizer
batch_size = 1  # per_device（减小以降低多节点 OOM 导致 NCCL 断连）
accumulative_counts = 8  # 梯度累积；每卡有效=8，64卡全局有效 batch=512（与原先一致）
dataloader_num_workers = 4 # 减少worker数量以避免共享内存不足（shm不足会导致bus error）
max_epochs = 2
optim_type = AdamW
# 64卡×2×4=512 有效batch：按 sqrt(batch) 从 4e-5 缩放，推荐 8e-5；若收敛慢可试 1e-4
lr = 8e-5  # 原 4e-5 为小 batch 设计；512 batch 下 8e-5 更合适
betas = (0.9, 0.999)
weight_decay = 0.05
max_norm = 1  # grad clip
warmup_ratio = 0.03

# Save
save_steps = 2000
save_total_limit = 4  # Maximum checkpoints to keep (-1 means unlimited)

# Logging
logging_interval = 10

# Evaluate the generation performance during the training
evaluation_freq = 2000
SYSTEM = ""
evaluation_images = [
    code_dir + "xsam/configs/xsam/images/imgconv.png",
    code_dir + "xsam/configs/xsam/images/imgconv.png",
    code_dir + "xsam/configs/xsam/images/imgconv.png",
    code_dir + "xsam/configs/xsam/images/imgconv.png",
]
evaluation_inputs = [
    "Can you describe this image in detail? Please elaborate in your response.",
    "Can you generate segmentation masks for this image based on the specified categories: <p>water</p>, <p>tree</p>, <p>car</p>, <p>forest</p>, <p>airplane</p>, <p>grass</p>, <p>harbor</p>, <p>ship</p>, <p>building</p>? Please output the segmentation mask.",
    "Can you segment <p>the harbor on the left side of the image</p> in this image? Please output the corresponding segmentation mask.",
    "<p>Where does a ship go to sleep?</p> Please explain why and output the corresponding segmentation mask.",
]
vprompt_masks = [
    (None,),  # imgconv
    (None,),  # genseg
    (None,),  # refseg
    (None,),  # reaseg
]

#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
# TODO: add special tokens via import from xsam.utils
special_tokens = ["<SEG>", "<p>", "</p>"]
cond_type = "phrase"  # "phrase" "cls" "all"
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
    freeze_llm=False,  # 不冻结LLM，但使用LoRA微调（大幅节省显存）
    freeze_visual_encoder=False,
    freeze_segmentor_encoder=False,
    use_dual_encoder=True,
    use_vision_sampler=True,
    use_activation_checkpointing=True,  # 启用梯度检查点以节省内存
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
    llm_lora=llm_lora_config,  # 添加LoRA配置，节省约45GB显存
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
        reinit_decoder=True,
        open_cls=True,
    ),
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
# 数据路径配置
genseg_data_root = data_dir + "gen_seg_data/"
ovseg_data_root = data_dir + "ov_seg_data/"
refseg_data_root = data_dir + "ref_seg_data/"
reasonseg_data_root = data_dir + "reasonseg/"
pano_data_root = data_dir + "pano/"
imgconv_data_root = data_dir + "img_conv_data/"
# 绝对路径基准，避免相对路径报错

oneterra_data_root = base_root + "/shui/oneterra_data/"
yangsen_data_root = base_root + "/yangsen/datasets/"

fitrs_data_root = oneterra_data_root + "imgconv/FIT-RS/raw_data/"
fitrs_imgconv_data_path = fitrs_data_root + "train_data_of_each_individual_task/"
fitrs_imgconv_image_folder = fitrs_data_root + "imgv2_split_512_100_vaild"
# 光学图像描述数据集路径
optical_caption_data_root = oneterra_data_root + "imgconv/image_caption/"

# False for predict mode, True for tensor mode
output_ids_with_output = True

# Training image processor with data augmentation
train_extra_image_processor = deepcopy(extra_image_processor)
train_extra_image_processor.update(
    {
        "size": {"min_scale": 0.1, "max_scale": 2.0, "target_size": 1024},  # 减小到512以节省内存
        "do_crop": True,
        "crop_size": {"height": 1024, "width": 1024},  # 减小到512以节省内存
    }
)

# Training datasets configuration
# 数据量平衡策略（已优化）：
# 目标：每个任务类型约100K-150K样本/epoch，避免数据不平衡
# 
# Image Conversation策略：
#   - 大数据集（FIT-RS complexcompre 708K, VQA 400K）降低权重至0.1-0.15
#   - 中等数据集（GeoChat 99K, classification 130K）保持适中权重0.8-1.2
#   - 小数据集（caption 65K, multiturn 50K等）提高权重至1.5-2.0
#   - 目标：Image Conversation总样本数约150K-200K/epoch
# 
# Segmentation策略：
#   - Pano数据集（111K样本）保持repeats_scale=1.0
#   - 目标：约222K样本/epoch（genseg + ovseg）
# 
# Referring Segmentation策略：
#   - RemoteSAM（149K样本）降低权重至0.8
#   - 目标：约120K样本/epoch
# 
# Reasoning Segmentation策略：
#   - 数据量未知，保持repeats_scale=1.0
#   - 根据实际数据量后续调整

# 1. GeoChat 遥感对话数据 (imgconv) - LLaVA格式
# 优化：
# - 启用 preprocess_text_data=True 以在初始化时预处理文本，避免运行时 tokenize 导致的同步问题
# - 使用 repeats_scale=1.25 平衡数据量，使每个epoch约124K样本
geochat_imgconv_dataset = dict(
    type=ImgConvDataset,
    data_path=imgconv_data_root + "geochat/geochat_llava.json",
    tokenizer=tokenizer,
    cond_type=cond_type,
    special_tokens=special_tokens,
    image_folder=imgconv_data_root + "geochat/images",
    image_processor=image_processor,
    extra_image_processor=train_extra_image_processor,
    task_name="imgconv",
    data_name="geochat_imgconv",
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=image_conv_map_fn,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pixel_values_ndim=2,
    is_multimodal=True,
    exclude_pure_text=True,
    pad_image_to_square=False,
    preprocess_text_data=True,  # 启用预处理，避免运行时 tokenize 导致的同步问题
    repeats_scale=1,  # 99,740 * 1.2 ≈ 119,688 样本/epoch
)

# FIT-RS imgconv训练数据集（使用cleaned.json文件）
# 1. Complex Comprehension (复杂理解)
fitrs_complexcompre_imgconv_dataset = dict(
    type=ImgConvDataset,
    data_path=fitrs_imgconv_data_path + "train_instruction_complexcompre_708k_cleaned.json",
    tokenizer=tokenizer,
    cond_type=cond_type,
    special_tokens=special_tokens,
    image_folder=fitrs_imgconv_image_folder,
    image_processor=image_processor,
    extra_image_processor=train_extra_image_processor,
    task_name="imgconv",
    data_name="fitrs_complexcompre_imgconv",
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=image_conv_map_fn,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pixel_values_ndim=2,
    is_multimodal=True,
    exclude_pure_text=True,
    pad_image_to_square=False,
    preprocess_text_data=True,
    repeats_scale=1,  # 708K样本，降低权重避免数据过多
)

# 2. Image Caption (图像描述)
fitrs_imagecaption_imgconv_dataset = dict(
    type=ImgConvDataset,
    data_path=fitrs_imgconv_data_path + "train_instruction_imagecaption_65k_cleaned.json",
    tokenizer=tokenizer,
    cond_type=cond_type,
    special_tokens=special_tokens,
    image_folder=fitrs_imgconv_image_folder,
    image_processor=image_processor,
    extra_image_processor=train_extra_image_processor,
    task_name="imgconv",
    data_name="fitrs_imagecaption_imgconv",
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=image_conv_map_fn,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pixel_values_ndim=2,
    is_multimodal=True,
    exclude_pure_text=True,
    pad_image_to_square=False,
    preprocess_text_data=True,
    repeats_scale=1,  # 65K样本，提高权重
)

# 3. Image Classification (图像分类)
fitrs_imageclassification_imgconv_dataset = dict(
    type=ImgConvDataset,
    data_path=fitrs_imgconv_data_path + "train_instruction_imageclassification_130k_cleaned.json",
    tokenizer=tokenizer,
    cond_type=cond_type,
    special_tokens=special_tokens,
    image_folder=fitrs_imgconv_image_folder,
    image_processor=image_processor,
    extra_image_processor=train_extra_image_processor,
    task_name="imgconv",
    data_name="fitrs_imageclassification_imgconv",
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=image_conv_map_fn,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pixel_values_ndim=2,
    is_multimodal=True,
    exclude_pure_text=True,
    pad_image_to_square=False,
    preprocess_text_data=True,
    repeats_scale=1,  # 130K样本，适中权重
)

# 4. Multi-turn Conversation (多轮对话)
fitrs_multiturn_imgconv_dataset = dict(
    type=ImgConvDataset,
    data_path=fitrs_imgconv_data_path + "train_instruction_multiturn_50k_cleaned.json",
    tokenizer=tokenizer,
    cond_type=cond_type,
    special_tokens=special_tokens,
    image_folder=fitrs_imgconv_image_folder,
    image_processor=image_processor,
    extra_image_processor=train_extra_image_processor,
    task_name="imgconv",
    data_name="fitrs_multiturn_imgconv",
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=image_conv_map_fn,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pixel_values_ndim=2,
    is_multimodal=True,
    exclude_pure_text=True,
    pad_image_to_square=False,
    preprocess_text_data=True,
    repeats_scale=1,  # 50K样本，提高权重
)

# 5. Region Caption (区域描述)
fitrs_regioncaption_imgconv_dataset = dict(
    type=ImgConvDataset,
    data_path=fitrs_imgconv_data_path + "train_instruction_regioncaption_72k_cleaned.json",
    tokenizer=tokenizer,
    cond_type=cond_type,
    special_tokens=special_tokens,
    image_folder=fitrs_imgconv_image_folder,
    image_processor=image_processor,
    extra_image_processor=train_extra_image_processor,
    task_name="imgconv",
    data_name="fitrs_regioncaption_imgconv",
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=image_conv_map_fn,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pixel_values_ndim=2,
    is_multimodal=True,
    exclude_pure_text=True,
    pad_image_to_square=False,
    preprocess_text_data=True,
    repeats_scale=1,  # 72K样本，提高权重
)

# 6. VQA (视觉问答)
fitrs_vqa_imgconv_dataset = dict(
    type=ImgConvDataset,
    data_path=fitrs_imgconv_data_path + "train_instruction_vqa_400k_cleaned.json",
    tokenizer=tokenizer,
    cond_type=cond_type,
    special_tokens=special_tokens,
    image_folder=fitrs_imgconv_image_folder,
    image_processor=image_processor,
    extra_image_processor=train_extra_image_processor,
    task_name="imgconv",
    data_name="fitrs_vqa_imgconv",
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=image_conv_map_fn,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pixel_values_ndim=2,
    is_multimodal=True,
    exclude_pure_text=True,
    pad_image_to_square=False,
    preprocess_text_data=True,
    repeats_scale=1,  # 400K样本，降低权重避免数据过多
)



#SAR imgconv训练数据集，整合所有的SAR训练数据
sar_total_imgconv_dataset = dict(
    type=ImgConvDataset,
    data_path=yangsen_data_root + "sar_total/sft/train.json",
    tokenizer=tokenizer,
    cond_type=cond_type,
    special_tokens=special_tokens,
    image_folder=yangsen_data_root,
    image_processor=image_processor,
    extra_image_processor=train_extra_image_processor,
    task_name="imgconv",
    data_name="sar_total_imgconv",
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=image_conv_map_fn,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pixel_values_ndim=2,
    is_multimodal=True,
    exclude_pure_text=True,
    pad_image_to_square=False,
    preprocess_text_data=True,
    repeats_scale=1,
)


# 2. Generic Segmentation (genseg) - 使用pano数据集的train模式
# 注意：PanoSegDataset 与 GenSeg 共用 annotations 中 categories 顺序作为 contiguous（与 val/eval 一致）
pano_genseg_dataset = dict(
    type=PanoSegDataset,
    data_path=pano_data_root + "annotations_train.json",
    image_folder=pano_data_root + "train/images",
    panseg_map_folder=pano_data_root + "train/panoptic_labels",
    tokenizer=tokenizer,
    task_name="genseg",
    data_name="pano_panoptic_genseg_train",
    cond_type=cond_type,
    special_tokens=special_tokens,
    extra_image_processor=train_extra_image_processor,
    image_processor=image_processor,
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=generic_seg_map_fn,
        cond_type=cond_type,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    use_variant_cat=True,
    pad_image_to_square=False,
    repeats_scale=3,
)

# 3. Open-Vocabulary Segmentation (ovseg) - OVSeg open-vocab 子集/全类混合采样（sample_num=32）
pano_ovseg_dataset = dict(
    type=OVSegDataset,
    data_path=pano_data_root + "annotations_train.json",
    image_folder=pano_data_root + "train/images",
    panseg_map_folder=pano_data_root + "train/panoptic_labels",
    tokenizer=tokenizer,
    task_name="ovseg",
    data_name="pano_ovseg_train",
    cond_type=cond_type,
    special_tokens=special_tokens,
    image_processor=image_processor,
    extra_image_processor=train_extra_image_processor,
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=ov_seg_map_fn,
        cond_type=cond_type,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=False,
    use_variant_cat=True,
    use_full_cat=True,
    sample_num=32,
    repeats_scale=1,
)

# 4. Referring Segmentation (refseg) - RemoteSAM data
remotesam_refseg_dataset = dict(
    type=RefSegDataset,
    data_root=refseg_data_root,  # 应该是包含remotesam子目录的父目录
    image_folder=refseg_data_root + "images/remotesam_images",
    dataset="remotesam",  # 数据集名称，REFER类会在data_root下查找remotesam子目录
    data_split="train",  # 根据数据中的split字段调整
    tokenizer=tokenizer,
    task_name="refseg",
    data_name="remotesam_train_refseg",
    cond_type=cond_type,
    special_tokens=special_tokens,
    extra_image_processor=train_extra_image_processor,
    image_processor=image_processor,
    postprocess_fn=refer_seg_postprocess_fn,
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=refer_seg_map_fn,
        cond_type=cond_type,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    use_variant_cat=True,
    use_random_cat=True,
    max_length=max_length,
    pad_image_to_square=False,
    ignore_label=ignore_label,
    repeats_scale=3,  # 149,519 * 0.8 ≈ 119,615 样本/epoch
)

# 5. Referring Segmentation (refseg) - FAST train (根据验证配置，将val改为train)
fast_refseg_dataset = dict(
    type=RefSegDataset,
    data_root=oneterra_data_root + "refseg/FAST/fast",
    image_folder=oneterra_data_root + "refseg/FAST/images",
    dataset="fast",
    data_split="train",  # 从验证配置的val改为train
    tokenizer=tokenizer,
    task_name="refseg",
    data_name="refseg_fast_train",
    cond_type=cond_type,
    special_tokens=special_tokens,
    extra_image_processor=train_extra_image_processor,
    image_processor=image_processor,
    postprocess_fn=refer_seg_postprocess_fn,
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=refer_seg_map_fn,
        cond_type=cond_type,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    use_variant_cat=True,
    use_random_cat=True,
    max_length=max_length,
    pad_image_to_square=False,
    ignore_label=ignore_label,
    repeats_scale=3,
)

# 6. Reasoning Segmentation (reaseg) - EarthReason train (根据验证配置，将test改为train)
earthreason_reaseg_dataset = dict(
    type=ReasonSegDataset,
    data_root=oneterra_data_root + "reasonseg/EarthReason_convert",
    image_folder=oneterra_data_root + "reasonseg/EarthReason_convert/train",  # 从test改为train
    explain_path=oneterra_data_root + "reasonseg/EarthReason_convert/explanatory/train.json",
    data_split="train",  # 从test改为train
    tokenizer=tokenizer,
    task_name="reaseg",
    data_name="reaseg_earthreason_train",
    cond_type=cond_type,
    special_tokens=special_tokens,
    extra_image_processor=train_extra_image_processor,
    image_processor=image_processor,
    postprocess_fn=reason_seg_postprocess_fn,
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=reason_seg_map_fn,
        cond_type=cond_type,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    use_variant_cat=True,
    use_random_cat=True,
    max_length=max_length,
    pad_image_to_square=False,
    ignore_label=ignore_label,
    use_threads=True,
    repeats_scale=3,
)

# 7. Reasoning Segmentation (reaseg) - diy1 train (根据验证配置，将test改为train)
diy1_reaseg_dataset = dict(
    type=ReasonSegDataset,
    data_root=oneterra_data_root + "reasonseg/diy1",
    image_folder=oneterra_data_root + "reasonseg/diy1/train",  # 从test改为train
    data_split="train",  # 从test改为train
    tokenizer=tokenizer,
    task_name="reaseg",
    data_name="reaseg_diy1_train",
    cond_type=cond_type,
    special_tokens=special_tokens,
    extra_image_processor=train_extra_image_processor,
    image_processor=image_processor,
    postprocess_fn=reason_seg_postprocess_fn,
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=reason_seg_map_fn,
        cond_type=cond_type,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    use_variant_cat=True,
    use_random_cat=True,
    max_length=max_length,
    pad_image_to_square=False,
    ignore_label=ignore_label,
    use_threads=True,
    repeats_scale=3,
)

# 仅保留 OVSeg 任务继续微调（与旧 read_base 子配置行为一致）
combined_train_dataset = dict(
    type=ConcatDataset,
    oversample_ratio=0.0,
    datasets=[pano_ovseg_dataset],
)

# Training dataloader configuration
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    pin_memory=True,
    persistent_workers=False,  # 禁用persistent_workers以避免共享内存不足问题
    dataset=combined_train_dataset,
    sampler=dict(
        type=SourceGroupedSampler,
        length_property="source_length",
        mega_batch_mult=1,
        per_device_batch_size=batch_size * accumulative_counts,
    ),
    collate_fn=dict(type=xsam_collate_fn),
)

val_datasets = [
    # 1. Generic Segmentation (genseg) - 与 RS-Xsam 离线测评一致：Pano 官方 val（非 gen_seg_data/sota）
    dict(
        type=GenSegDataset,
        data_path=pano_data_root + "annotations_val.json",
        image_folder=pano_data_root + "val/images",
        panseg_map_folder=pano_data_root + "val/panoptic_labels",
        data_mode="eval",
        tokenizer=tokenizer,
        task_name="genseg",
        data_name="panoptic_genseg_pano_val",
        cond_type=cond_type,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        image_processor=image_processor,
        output_ids_with_output=output_ids_with_output,
        postprocess_fn=dict(
            type=process_map_fn_factory,
            fn=generic_seg_postprocess_fn,
            task_name="panoptic_genseg",
            threshold=0.0,
        ),
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=generic_seg_map_fn,
            cond_type=cond_type,
        ),
        template_map_fn=dict(
            type=template_map_fn_factory,
            template=prompt_template,
            output_suffix=output_ids_with_output,
        ),
        max_length=max_length,
        pad_image_to_square=True,
    ),
    # 2. Open-Vocabulary Segmentation (ovseg) - 与 genseg 相同 Pano val，便于训练过程与离线测评对齐
    dict(
        type=OVSegDataset,
        data_path=pano_data_root + "annotations_val.json",
        image_folder=pano_data_root + "val/images",
        panseg_map_folder=pano_data_root + "val/panoptic_labels",
        data_mode="eval",
        tokenizer=tokenizer,
        task_name="ovseg",
        data_name="panoptic_ovseg_pano_val",
        output_ids_with_output=output_ids_with_output,
        cond_type=cond_type,
        special_tokens=special_tokens,
        image_processor=image_processor,
        extra_image_processor=extra_image_processor,
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=ov_seg_map_fn,
            cond_type=cond_type,
        ),
        postprocess_fn=dict(
            type=process_map_fn_factory,
            fn=ov_seg_postprocess_fn,
            task_name="panoptic_ovseg",
            threshold=0.0,
        ),
        template_map_fn=dict(
            type=template_map_fn_factory,
            template=prompt_template,
            output_suffix=output_ids_with_output,
        ),
        max_length=max_length,
        pad_image_to_square=True,
    ),
    # 3. Referring Segmentation (refseg) - RemoteSAM validation data
    dict(
        type=RefSegDataset,
        data_root=refseg_data_root,
        image_folder=refseg_data_root + "images/remotesam_images",
        dataset="remotesam",
        data_split="val",
        data_mode="eval",
        tokenizer=tokenizer,
        task_name="refseg",
        data_name="remotesam_val_refseg",
        cond_type=cond_type,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        output_ids_with_output=output_ids_with_output,
        image_processor=image_processor,
        postprocess_fn=refer_seg_postprocess_fn,
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=refer_seg_map_fn,
            cond_type=cond_type,
        ),
        template_map_fn=dict(
            type=template_map_fn_factory,
            template=prompt_template,
            output_suffix=output_ids_with_output,
        ),
        max_length=max_length,
        pad_image_to_square=True,
        ignore_label=ignore_label,
    ),
    # 4. Referring Segmentation (refseg) - RemoteSAM test data
    dict(
        type=RefSegDataset,
        data_root=refseg_data_root,
        image_folder=refseg_data_root + "images/remotesam_images",
        dataset="remotesam",
        data_split="test",
        data_mode="eval",
        tokenizer=tokenizer,
        task_name="refseg",
        data_name="remotesam_test_refseg",
        cond_type=cond_type,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        output_ids_with_output=output_ids_with_output,
        image_processor=image_processor,
        postprocess_fn=refer_seg_postprocess_fn,
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=refer_seg_map_fn,
            cond_type=cond_type,
        ),
        template_map_fn=dict(
            type=template_map_fn_factory,
            template=prompt_template,
            output_suffix=output_ids_with_output,
        ),
        max_length=max_length,
        pad_image_to_square=True,
        ignore_label=ignore_label,
    ),
    # 4. Image Conversation (imgconv) - GeoChat validation data
    dict(
        type=ImgConvDataset,
        data_path=imgconv_data_root + "geochat/geochat_llava_val.json",
        tokenizer=tokenizer,
        cond_type=cond_type,
        special_tokens=special_tokens,
        image_folder=imgconv_data_root + "geochat/images",
        image_processor=image_processor,
        extra_image_processor=extra_image_processor,
        task_name="imgconv",
        data_name="geochat_imgconv_val",
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=image_conv_map_fn,
        ),
        template_map_fn=dict(
            type=template_map_fn_factory,
            template=prompt_template,
            output_suffix=True,  # 评估时需要output来计算指标
        ),
        max_length=max_length,
        pixel_values_ndim=2,
        is_multimodal=True,
        exclude_pure_text=True,
        pad_image_to_square=False,
        output_ids_with_output=True,  # 评估时需要output来计算指标
    ),
    # 5. Reasoning Segmentation (reaseg) - LISA validation data
    dict(
        type=ReasonSegDataset,
        data_root=reasonseg_data_root + "lisa",
        image_folder=reasonseg_data_root + "lisa/val",
        explain_path=reasonseg_data_root + "lisa/explanatory/val.json",
        data_mode="eval",
        tokenizer=tokenizer,
        task_name="reaseg",
        data_name="lisa_reaseg_val",
        cond_type=cond_type,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        output_ids_with_output=output_ids_with_output,
        image_processor=image_processor,
        postprocess_fn=reason_seg_postprocess_fn,
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=reason_seg_map_fn,
            cond_type=cond_type,
        ),
        template_map_fn=dict(
            type=template_map_fn_factory,
            template=prompt_template,
            output_suffix=output_ids_with_output,
        ),
        max_length=max_length,
        pad_image_to_square=True,
        ignore_label=ignore_label,
        use_threads=True,  # 使用线程池以提高I/O性能
    ),
]

val_evaluators = [
    # 1. Generic Segmentation (genseg) - Pano val（data_name 须与 val_datasets 一致以挂 MetadataCatalog）
    dict(
        type=GenSegEvaluator,
        distributed=True,
        data_name="panoptic_genseg_pano_val",
    ),
    # 2. Open-Vocabulary Segmentation (ovseg)
    dict(
        type=OVSegEvaluator,
        data_name="panoptic_ovseg_pano_val",
        distributed=True,
    ),
    # 3. Referring Segmentation (refseg) - RemoteSAM validation
    dict(
        type=RefSegEvaluator,
        distributed=True,
        data_name="remotesam_val_refseg",
    ),
    # 4. Image Conversation (imgconv) - GeoChat validation
    dict(
        type=ImgConvEvaluator,
        distributed=True,
        data_name="geochat_imgconv_val",
    ),
    # 5. Reasoning Segmentation (reaseg) - LISA validation
    dict(
        type=ReasonSegEvaluator,
        distributed=True,
        data_name="lisa_reaseg_val",
    ),
]

vis_datasets = deepcopy(val_datasets)
for dataset in vis_datasets:
    if dataset["task_name"] in ["genseg", "ovseg"]:
        dataset["postprocess_fn"]["threshold"] = 0.5  # type: ignore

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale="dynamic",
    dtype="float16",
    paramwise_cfg=dict(
        # Avoid adding tied/shared parameters (e.g., embedding <-> lm_head) multiple times
        # when traversing complex HF modules
        bypass_duplicate=True,
        custom_keys={
            "segmentor.encoder": dict(lr_mult=0.1, decay_mult=1.0),
            "visual_encoder": dict(lr_mult=0.1, decay_mult=1.0),
        },
    ),
)

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
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

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# set visualizer
visualizer = dict(
    type=Visualizer,
    scale=1.0,
    font_size_scale=1.0,
)

# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(
        type=ModelInfoHook,
        module_names=["llm", "visual_encoder", "projector", "connector", "segmentor"],
        display_params=True,
    ),
    dict(type=DatasetInfoHook, tokenizer=tokenizer, special_tokens=special_tokens),
    dict(
        type=EvaluateChatHook,
        tokenizer=tokenizer,
        special_tokens=special_tokens,
        image_processor=image_processor,
        postprocess_fns=[
            None,  # imgconv
            generic_seg_postprocess_fn,  # genseg
            refer_seg_postprocess_fn,  # refseg
            reason_seg_postprocess_fn,  # reaseg
        ],
        extra_image_processor=extra_image_processor,
        visualizer=visualizer,
        every_n_iters=evaluation_freq,
        evaluation_inputs=evaluation_inputs,
        evaluation_images=evaluation_images,
        vprompt_masks=vprompt_masks,
        system=SYSTEM,
        prompt_template=prompt_template,
    ),
    dict(type=PTCheckpointHook, clean_pth=False),
]

# configure default hooks
default_hooks = dict(
    # 先对 loss 做跨卡 mean，再写 log，这样打印的是 64 卡平均 loss
    dist_loss_reduce=dict(type=DistLossReduceHook),
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=logging_interval),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit,
    ),
    # set sampler seed in distributed environment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend="nccl"),
)

# set log level
log_level = "INFO"

# 从已训完的 S3 全任务权重继续；环境变量 PREV_S3_CKPT 可覆盖默认路径
prev_s3_ckpt = getenv(
    "PREV_S3_CKPT",
    "/mnt_llm_A100_V1/shui/LAE/OneTerra-train/wkdrs_01/s3_mixed_fineture_base/"
    "xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune_all_v1/iter_61176.pth",
)
load_from = prev_s3_ckpt
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(
    by_epoch=False,
    window_size=1,
    mean_pattern=r".*(loss|time|data_time|grad_norm|tflops).*",
)

