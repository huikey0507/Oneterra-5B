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
)
from xsam.dataset.collate_fns import xsam_collate_fn
from xsam.dataset.map_fns import (
    dataset_map_fn_factory,
    # generic_seg_map_fn,
    # imgconv_map_fn,
    # ovseg_map_fn,
    # refer_seg_map_fn,
    template_map_fn_factory,
)
from xsam.dataset.map_fns.dataset_map_fns import image_conv_map_fn, generic_seg_map_fn, ov_seg_map_fn, refer_seg_map_fn
from xsam.dataset.process_fns.postprocess_fns import (
    generic_seg_postprocess_fn,
    ov_seg_postprocess_fn,
    refer_seg_postprocess_fn,
)
from xsam.dataset.process_fns import process_map_fn_factory
from xsam.dataset.processors import SamImageProcessor
from xsam.dataset.samplers import SourceGroupedSampler
from xsam.engine.hooks import DatasetInfoHook, EvaluateChatHook, ModelInfoHook, PTCheckpointHook
from xsam.engine.runners.loops import TrainLoop
from xsam.evaluation.evaluators import (
    GenSegEvaluator,
    ImgConvEvaluator,
    OVSegEvaluator,
    RefSegEvaluator,
)
from peft import LoraConfig
from xsam.model import XSamModel
from xsam.model.segmentors import XSegmentor
from xsam.model.segmentors.mask2former import Mask2FormerConfig, Mask2FormerModel
from xsam.model.segmentors.sam import SamModel
from xsam.utils.visualize import Visualizer
import xsam.engine.runners.loops



#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Directories
code_dir = getenv("CODE_DIR", "./xsam/")
data_dir = "/mnt_llm_A100_V1/shui/LAE/X-SAM/X-SAM/datas/"
init_dir = getenv("INIT_DIR", "./inits/")
work_dir = getenv("WORK_DIR", "./checkpoints/")
checkpoints_dir = work_dir = getenv("WORK_DIR", "/mnt_llm_A100_V1/shui/LAE/XSAM-public/RS-Xsam-main/checkpoints/")
print("checkpoints_dir:", checkpoints_dir)
# Model
llm_name_or_path = init_dir + "Phi-3-mini-4k-instruct"
visual_encoder_name_or_path = init_dir + "siglip-so400m-patch14-384"
seg_encoder_name_or_path = init_dir + "sam-vit-large"
seg_decoder_name_or_path = init_dir + "mask2former-swin-large-coco-panoptic"

# Specify the pretrained pth
# Stage 1: 使用wkdrs/s1_seg_finetune（已链接到wkdrs3/s1_seg_finetune）
s1_pretrained_pth = work_dir + "s1_seg_finetune/xsam_sota_s1_finetune/iter_66000.pth/pytorch_model.bin"
# Stage 2: 使用iter_47466.pth目录（最新检查点）
s2_pretrained_pth = (
    work_dir
    + "s2_align_pretrain/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_e1_gpu16_align_pretrain_skyscript/iter_47466.pth"
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
batch_size = 2  # per_device (进一步减小以节省内存)
accumulative_counts = 4  # 梯度累积，保持有效batch_size=4
dataloader_num_workers = 4 # 减少worker数量以避免共享内存不足（shm不足会导致bus error）
max_epochs = 2
optim_type = AdamW
lr = 4e-5
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
]
evaluation_inputs = [
    "Can you describe this image in detail? Please elaborate in your response.",
    "Can you generate segmentation masks for this image based on the specified categories: <p>person</p>, <p>bicycle</p>, <p>car</p>, <p>motorcycle</p>, <p>airplane</p>, <p>bus</p>, <p>train</p>, <p>truck</p>, <p>boat</p>, <p>traffic light</p>, <p>fire hydrant</p>, <p>stop sign</p>, <p>parking meter</p>, <p>bench</p>, <p>bird</p>, <p>cat</p>, <p>dog</p>, <p>horse</p>, <p>sheep</p>, <p>cow</p>, <p>elephant</p>, <p>bear</p>, <p>zebra</p>, <p>giraffe</p>, <p>backpack</p>, <p>umbrella</p>, <p>handbag</p>, <p>tie</p>, <p>suitcase</p>, <p>frisbee</p>, <p>skis</p>, <p>snowboard</p>, <p>sports ball</p>, <p>kite</p>, <p>baseball bat</p>, <p>baseball glove</p>, <p>skateboard</p>, <p>surfboard</p>, <p>tennis racket</p>, <p>bottle</p>, <p>wine glass</p>, <p>cup</p>, <p>fork</p>, <p>knife</p>, <p>spoon</p>, <p>bowl</p>, <p>banana</p>, <p>apple</p>, <p>sandwich</p>, <p>orange</p>, <p>broccoli</p>, <p>carrot</p>, <p>hot dog</p>, <p>pizza</p>, <p>donut</p>, <p>cake</p>, <p>chair</p>, <p>couch</p>, <p>potted plant</p>, <p>bed</p>, <p>dining table</p>, <p>toilet</p>, <p>tv</p>, <p>laptop</p>, <p>mouse</p>, <p>remote</p>, <p>keyboard</p>, <p>cell phone</p>, <p>microwave</p>, <p>oven</p>, <p>toaster</p>, <p>sink</p>, <p>refrigerator</p>, <p>book</p>, <p>clock</p>, <p>vase</p>, <p>scissors</p>, <p>teddy bear</p>, <p>hair drier</p>, <p>toothbrush</p>, <p>banner</p>, <p>blanket</p>, <p>bridge</p>, <p>cardboard</p>, <p>counter</p>, <p>curtain</p>, <p>door</p>, <p>floor wood</p>, <p>flower</p>, <p>fruit</p>, <p>gravel</p>, <p>house</p>, <p>light</p>, <p>mirror</p>, <p>net</p>, <p>pillow</p>, <p>platform</p>, <p>playingfield</p>, <p>railroad</p>, <p>river</p>, <p>road</p>, <p>roof</p>, <p>sand</p>, <p>sea</p>, <p>shelf</p>, <p>snow</p>, <p>stairs</p>, <p>tent</p>, <p>towel</p>, <p>wall brick</p>, <p>wall stone</p>, <p>wall tile</p>, <p>wall wood</p>, <p>water</p>, <p>window blind</p>, <p>window</p>, <p>tree</p>, <p>fence</p>, <p>ceiling</p>, <p>sky</p>, <p>cabinet</p>, <p>table</p>, <p>floor</p>, <p>pavement</p>, <p>mountain</p>, <p>grass</p>, <p>dirt</p>, <p>paper</p>, <p>food</p>, <p>building</p>, <p>rock</p>, <p>wall</p>, <p>rug</p>? Please output the segmentation mask.",
    "Can you segment <p>the women with red coat</p> in this image? Please output the corresponding segmentation mask.",
]
vprompt_masks = [
    (None,),  # imgconv
    (None,),  # genseg
    (None,),  # refseg
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
imgconv_data_root = data_dir + "img_conv_data/"

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
# 数据量平衡策略（已启用）：
# - GeoChat: 99,740样本 × repeats_scale=1.25 ≈ 124,675样本/epoch
# - 通用分割: 15,732样本 × repeats_scale=7.92 ≈ 124,597样本/epoch
# - 开集分割: 15,732样本 × repeats_scale=7.92 ≈ 124,597样本/epoch
# - 指代分割: 149,519样本 × repeats_scale=1.0 = 149,519样本/epoch (数据量已足够大，不重复)
# 这样前三个任务在每个epoch中出现的次数相近（约124K），指代分割数据量最大保持原样
# 总数据量：约523K样本/epoch，确保所有任务都能得到充分训练

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
    repeats_scale=1,  # 99,740 * 1.25 ≈ 124,675 样本/epoch
)

# 2. Generic Segmentation (genseg) - SOTA data
sota_genseg_dataset = dict(
    type=GenSegDataset,
    data_path=genseg_data_root + "sota/train_annotations.json",
    image_folder=genseg_data_root + "sota/images",
    panseg_map_folder=genseg_data_root + "sota/panoptic_labels",
    tokenizer=tokenizer,
    task_name="genseg",
    data_name="sota_panoptic_genseg_train",
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
    repeats_scale=5,  # 15,732 * 7.92 ≈ 124,597 样本/epoch
)

# 3. Open-Vocabulary Segmentation (ovseg) - SOTA data
sota_ovseg_dataset = dict(
    type=OVSegDataset,
    data_path=ovseg_data_root + "sota/train/train_annotations.json",
    image_folder=ovseg_data_root + "sota/train/images",
    panseg_map_folder=ovseg_data_root + "sota/train/panoptic_labels",
    tokenizer=tokenizer,
    task_name="ovseg",
    data_name="sota_panoptic_ovseg_train",
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
    repeats_scale=5,  # 15,732 * 7.92 ≈ 124,597 样本/epoch
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
    repeats_scale=1.0,  # 149,519 * 1.0 = 149,519 样本/epoch (数据量已足够大，不重复)
)

# Combine training datasets: geochat, genseg, ovseg, refseg
# 注意：使用repeats_scale后，oversample_ratio会被覆盖，设为0.0禁用自动平衡
combined_train_dataset = dict(
    type=ConcatDataset,
    oversample_ratio=0.0,  # 设为0.0，使用手动设置的repeats_scale
    datasets=[
        geochat_imgconv_dataset,
        sota_genseg_dataset,
        sota_ovseg_dataset,
        remotesam_refseg_dataset,
    ],
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
    # 1. Generic Segmentation (genseg) - SOTA validation data
    dict(
        type=GenSegDataset,
        data_path=genseg_data_root + "sota/val_annotations.json",
        image_folder=genseg_data_root + "sota/val/images",
        panseg_map_folder=genseg_data_root + "sota/val/panoptic_labels",
        data_mode="eval",
        tokenizer=tokenizer,
        task_name="genseg",
        data_name="sota_panoptic_genseg_val",
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
        #max_eval_samples=50,
    ),
    # 2. Open-Vocabulary Segmentation (ovseg) - SOTA validation data
    dict(
        type=OVSegDataset,
        data_path=ovseg_data_root + "sota/val_annotations.json",
        image_folder=ovseg_data_root + "sota/val/images",
        panseg_map_folder=ovseg_data_root + "sota/val/panoptic_labels",
        data_mode="eval",
        tokenizer=tokenizer,
        task_name="ovseg",
        data_name="sota_panoptic_ovseg_val",
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
        #max_eval_samples=50,
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
        max_eval_samples=50,
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
        max_eval_samples=50,
    ),
    # 4. Image Conversation (imgconv) - GeoChat validation data
    dict(
        type=ImgConvDataset,
        data_path=imgconv_data_root + "geochat/geochat_llava_val_tmp.json",
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
]

val_evaluators = [
    # 1. Generic Segmentation (genseg) - SOTA validation
    dict(
        type=GenSegEvaluator,
        distributed=True,
        data_name="sota_panoptic_genseg_val",
    ),
    # 2. Open-Vocabulary Segmentation (ovseg) - SOTA validation
    dict(
        type=OVSegEvaluator,
        data_name="sota_panoptic_ovseg_val",
        distributed=True,
    ),
    # 3. Referring Segmentation (refseg) - RemoteSAM validation
    dict(
        type=RefSegEvaluator,
        distributed=True,
        data_name="remotesam_val_refseg",
    ),
    # 4. Referring Segmentation (refseg) - RemoteSAM test
    dict(
        type=RefSegEvaluator,
        distributed=True,
        data_name="remotesam_test_refseg",
    ),
    # 5. Image Conversation (imgconv) - GeoChat validation
    dict(
        type=ImgConvEvaluator,
        distributed=True,
        data_name="geochat_imgconv_val",
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

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(
    by_epoch=False,
    window_size=1,
    mean_pattern=r".*(loss|time|data_time|grad_norm|tflops).*",
)



