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
from xsam.engine.hooks import DatasetInfoHook, EvaluateChatHook, ModelInfoHook, PTCheckpointHook
from xsam.engine.runners.loops import TrainLoop
from xsam.evaluation.evaluators import (
    GenSegEvaluator,
    ImgConvEvaluator,
    ImgConvCCEvaluator,
    OVSegEvaluator,
    RefSegEvaluator,
    ImgConvMLSCEvaluator,
    ReasonSegEvaluator,
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
dataloader_num_workers = 1 # 减少worker数量以避免共享内存不足（shm不足会导致bus error）
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
reasonseg_data_root = "/mnt_llm_A100_V1/shui/oneterra_data/reasonseg/"
# reasonseg_data_root = "/mnt_llm_A100_V1/yangsen/datasets/xsam/"
imgconv_data_root = data_dir + "img_conv_data/"
imgconv_cc_data_root = "/mnt_llm_A100_V1/shui/oneterra_data/imgconv/FIT-RS/"
rsvqa_hr_data_root = "/mnt_llm_A100_V1/shui/oneterra_data/imgconv/VQA/RSVQA_HR/"
rsvqa_lr_data_root = "/mnt_llm_A100_V1/shui/oneterra_data/imgconv/VQA/RSVQA-LR/"
optical_caption_data_root = "/mnt_llm_A100_V1/shui/oneterra_data/imgconv/image_caption/"
SAR_data_root = "/mnt_llm_A100_V1/yangsen/datasets/sar_total/"
FuSAR_data_root = "/mnt_llm_A100_V1/yangsen/datasets/fusar_clip/"
SARLANG_data_root = "/mnt_llm_A100_V1/shui/oneterra_data/imgconv/SAR-LANG/"
WHU_RS19_data_root = "/mnt_llm_A100_V1/yangsen/datasets/WHU-RS19/"
AID_data_root = "/mnt_llm_A100_V1/yangsen/datasets/AID/"
NWPU_RESISC45_data_root = "/mnt_llm_A100_V1/yangsen/datasets/NWPU-RESISC45/"
SIRI_WHU_data_root = "/mnt_llm_A100_V1/yangsen/datasets/SIRI-WHU/"
UC_Merced_data_root = "/mnt_llm_A100_V1/yangsen/datasets/UC_Merced/"
AID_multilabel_data_root = "/mnt_llm_A100_V1/yangsen/datasets/AID_multilabel/"

# False for predict mode, True for tensor mode
output_ids_with_output = True


val_datasets = [
    # # 1. Generic Segmentation (genseg) - SOTA validation data
    # dict(
    #     type=GenSegDataset,
    #     data_path=genseg_data_root + "sota/val_annotations.json",
    #     image_folder=genseg_data_root + "sota/val/images",
    #     panseg_map_folder=genseg_data_root + "sota/val/panoptic_labels",
    #     data_mode="eval",
    #     tokenizer=tokenizer,
    #     task_name="genseg",
    #     data_name="panoptic_genseg_sota_val",
    #     cond_type=cond_type,
    #     special_tokens=special_tokens,
    #     extra_image_processor=extra_image_processor,
    #     image_processor=image_processor,
    #     output_ids_with_output=output_ids_with_output,
    #     postprocess_fn=dict(
    #         type=process_map_fn_factory,
    #         fn=generic_seg_postprocess_fn,
    #         task_name="panoptic_genseg",
    #         threshold=0.0,
    #     ),
    #     dataset_map_fn=dict(
    #         type=dataset_map_fn_factory,
    #         fn=generic_seg_map_fn,
    #         cond_type=cond_type,
    #     ),
    #     template_map_fn=dict(
    #         type=template_map_fn_factory,
    #         template=prompt_template,
    #         output_suffix=output_ids_with_output,
    #     ),
    #     max_length=max_length,
    #     pad_image_to_square=True,
    # ),
    # # 2. Open-Vocabulary Segmentation (ovseg) - SOTA validation data
    # dict(
    #     type=OVSegDataset,
    #     data_path=ovseg_data_root + "sota/val_annotations.json",
    #     image_folder=ovseg_data_root + "sota/val/images",
    #     panseg_map_folder=ovseg_data_root + "sota/val/panoptic_labels",
    #     data_mode="eval",
    #     tokenizer=tokenizer,
    #     task_name="ovseg",
    #     data_name="panoptic_ovseg_sota_val",
    #     output_ids_with_output=output_ids_with_output,
    #     cond_type=cond_type,
    #     special_tokens=special_tokens,
    #     image_processor=image_processor,
    #     extra_image_processor=extra_image_processor,
    #     dataset_map_fn=dict(
    #         type=dataset_map_fn_factory,
    #         fn=ov_seg_map_fn,
    #         cond_type=cond_type,
    #     ),
    #     postprocess_fn=dict(
    #         type=process_map_fn_factory,
    #         fn=ov_seg_postprocess_fn,
    #         task_name="panoptic_ovseg",
    #         threshold=0.0,
    #     ),
    #     template_map_fn=dict(
    #         type=template_map_fn_factory,
    #         template=prompt_template,
    #         output_suffix=output_ids_with_output,
    #     ),
    #     max_length=max_length,
    #     pad_image_to_square=True,
    #     #max_eval_samples=50,
    # ),
    # # 3. Referring Segmentation (refseg) - RemoteSAM validation
    # dict(
    #     type=RefSegDataset,
    #     data_root=refseg_data_root,
    #     image_folder=refseg_data_root + "images/remotesam_images",
    #     dataset="remotesam",
    #     data_split="val",
    #     data_mode="eval",
    #     tokenizer=tokenizer,
    #     task_name="refseg",
    #     data_name="refseg_remotesam_val",
    #     cond_type=cond_type,
    #     special_tokens=special_tokens,
    #     extra_image_processor=extra_image_processor,
    #     output_ids_with_output=output_ids_with_output,
    #     image_processor=image_processor,
    #     postprocess_fn=refer_seg_postprocess_fn,
    #     dataset_map_fn=dict(
    #         type=dataset_map_fn_factory,
    #         fn=refer_seg_map_fn,
    #         cond_type=cond_type,
    #     ),
    #     template_map_fn=dict(
    #         type=template_map_fn_factory,
    #         template=prompt_template,
    #         output_suffix=output_ids_with_output,
    #     ),
    #     max_length=max_length,
    #     pad_image_to_square=True,
    #     ignore_label=ignore_label,
    # ),
    # # 4. Referring Segmentation (refseg) - RemoteSAM test
    # dict(
    #     type=RefSegDataset,
    #     data_root=refseg_data_root,
    #     image_folder=refseg_data_root + "images/remotesam_images",
    #     dataset="remotesam",
    #     data_split="test",
    #     data_mode="eval",
    #     tokenizer=tokenizer,
    #     task_name="refseg",
    #     data_name="refseg_remotesam_test",
    #     cond_type=cond_type,
    #     special_tokens=special_tokens,
    #     extra_image_processor=extra_image_processor,
    #     output_ids_with_output=output_ids_with_output,
    #     image_processor=image_processor,
    #     postprocess_fn=refer_seg_postprocess_fn,
    #     dataset_map_fn=dict(
    #         type=dataset_map_fn_factory,
    #         fn=refer_seg_map_fn,
    #         cond_type=cond_type,
    #     ),
    #     template_map_fn=dict(
    #         type=template_map_fn_factory,
    #         template=prompt_template,
    #         output_suffix=output_ids_with_output,
    #     ),
    #     max_length=max_length,
    #     pad_image_to_square=True,
    #     ignore_label=ignore_label,
    # #     max_eval_samples=50,
    # ),
    # # 5. Referring Segmentation (refseg) - RefSegRS test
    # dict(
    #     type=RefSegDataset,
    #     data_root="/mnt_llm_A100_V1/shui/oneterra_data/refseg/RefSegRS",
    #     image_folder="/mnt_llm_A100_V1/shui/oneterra_data/refseg/RefSegRS/images",
    #     dataset="refsegrs",
    #     data_split="test",
    #     data_mode="eval",
    #     tokenizer=tokenizer,
    #     task_name="refseg",
    #     data_name="refseg_refsegrs_test",
    #     cond_type=cond_type,
    #     special_tokens=special_tokens,
    #     extra_image_processor=extra_image_processor,
    #     output_ids_with_output=output_ids_with_output,
    #     image_processor=image_processor,
    #     postprocess_fn=refer_seg_postprocess_fn,
    #     dataset_map_fn=dict(
    #         type=dataset_map_fn_factory,
    #         fn=refer_seg_map_fn,
    #         cond_type=cond_type,
    #     ),
    #     template_map_fn=dict(
    #         type=template_map_fn_factory,
    #         template=prompt_template,
    #         output_suffix=output_ids_with_output,
    #     ),
    #     max_length=max_length,
    #     pad_image_to_square=True,
    #     ignore_label=ignore_label,
    #     # max_eval_samples=50,
    # ),
    # # 6. Referring Segmentation (refseg) - RRSIS-D test
    # dict(
    #     type=RefSegDataset,
    #     data_root="/mnt_llm_A100_V1/shui/oneterra_data/refseg/RRSIS-D",
    #     image_folder="/mnt_llm_A100_V1/shui/oneterra_data/refseg/RRSIS-D/images/rrsisd/JPEGImages",
    #     dataset="rrsisd",
    #     data_split="test",
    #     data_mode="eval",
    #     tokenizer=tokenizer,
    #     task_name="refseg",
    #     data_name="refseg_rrsisd_test",
    #     cond_type=cond_type,
    #     special_tokens=special_tokens,
    #     extra_image_processor=extra_image_processor,
    #     output_ids_with_output=output_ids_with_output,
    #     image_processor=image_processor,
    #     postprocess_fn=refer_seg_postprocess_fn,
    #     dataset_map_fn=dict(
    #         type=dataset_map_fn_factory,
    #         fn=refer_seg_map_fn,
    #         cond_type=cond_type,
    #     ),
    #     template_map_fn=dict(
    #         type=template_map_fn_factory,
    #         template=prompt_template,
    #         output_suffix=output_ids_with_output,
    #     ),
    #     max_length=max_length,
    #     pad_image_to_square=True,
    #     ignore_label=ignore_label,
    # ),
    # # 7. Referring Segmentation (refseg) - RISBench test
    # dict(
    #     type=RefSegDataset,
    #     data_root="/mnt_llm_A100_V1/shui/oneterra_data/refseg/RISBench",
    #     image_folder="/mnt_llm_A100_V1/shui/oneterra_data/refseg/RISBench/RISBench_dataset/img_rgb",
    #     dataset="risbench",
    #     data_split="test",
    #     data_mode="eval",
    #     tokenizer=tokenizer,
    #     task_name="refseg",
    #     data_name="refseg_risbench_test",
    #     cond_type=cond_type,
    #     special_tokens=special_tokens,
    #     extra_image_processor=extra_image_processor,
    #     output_ids_with_output=output_ids_with_output,
    #     image_processor=image_processor,
    #     postprocess_fn=refer_seg_postprocess_fn,
    #     dataset_map_fn=dict(
    #         type=dataset_map_fn_factory,
    #         fn=refer_seg_map_fn,
    #         cond_type=cond_type,
    #     ),
    #     template_map_fn=dict(
    #         type=template_map_fn_factory,
    #         template=prompt_template,
    #         output_suffix=output_ids_with_output,
    #     ),
    #     max_length=max_length,
    #     pad_image_to_square=True,
    #     ignore_label=ignore_label,
    # ),
    # # 7. Referring Segmentation (refseg) - FAST val
    # dict(
    #     type=RefSegDataset,
    #     data_root="/mnt_llm_A100_V1/shui/oneterra_data/refseg/FAST",
    #     image_folder="/mnt_llm_A100_V1/shui/oneterra_data/refseg/FAST/images",
    #     dataset="fast",
    #     data_split="val",
    #     data_mode="eval",
    #     tokenizer=tokenizer,
    #     task_name="refseg",
    #     data_name="refseg_fast_val",
    #     cond_type=cond_type,
    #     special_tokens=special_tokens,
    #     extra_image_processor=extra_image_processor,
    #     output_ids_with_output=output_ids_with_output,
    #     image_processor=image_processor,
    #     postprocess_fn=refer_seg_postprocess_fn,
    #     dataset_map_fn=dict(
    #         type=dataset_map_fn_factory,
    #         fn=refer_seg_map_fn,
    #         cond_type=cond_type,
    #     ),
    #     template_map_fn=dict(
    #         type=template_map_fn_factory,
    #         template=prompt_template,
    #         output_suffix=output_ids_with_output,
    #     ),
    #     max_length=max_length,
    #     pad_image_to_square=True,
    #     ignore_label=ignore_label,
    # ),
    # 8. Reasoning Segmentation (reasonseg) - EarthReason test
    dict(
        type=ReasonSegDataset,
        data_root=reasonseg_data_root + "EarthReason_convert",
        image_folder=reasonseg_data_root + "EarthReason_convert/test",
        data_split="test",
        data_mode="eval",
        tokenizer=tokenizer,
        task_name="reaseg",
        data_name="reaseg_earthreason_test",
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
            type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
        ),
        max_length=max_length,
        pad_image_to_square=True,
        ignore_label=ignore_label,
    ),
    # # 8. Reasoning Segmentation (reasonseg) - reaseg_diy1 test
    # dict(
    #     type=ReasonSegDataset,
    #     data_root=reasonseg_data_root + "diy1",
    #     image_folder=reasonseg_data_root + "diy1/test",
    #     data_split="test",
    #     data_mode="eval",
    #     tokenizer=tokenizer,
    #     task_name="reaseg",
    #     data_name="reaseg_diy1_test",
    #     cond_type=cond_type,
    #     special_tokens=special_tokens,
    #     extra_image_processor=extra_image_processor,
    #     output_ids_with_output=output_ids_with_output,
    #     image_processor=image_processor,
    #     postprocess_fn=reason_seg_postprocess_fn,
    #     dataset_map_fn=dict(
    #         type=dataset_map_fn_factory,
    #         fn=reason_seg_map_fn,
    #         cond_type=cond_type,
    #     ),
    #     template_map_fn=dict(
    #         type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
    #     ),
    #     max_length=max_length,
    #     pad_image_to_square=True,
    #     ignore_label=ignore_label,
    # ),
    # # 9. Image Conversation (Scene Classification) - WHU-RS19
    # dict(
    #     type=ImgConvDataset,
    #     data_path=WHU_RS19_data_root + "data.json",
    #     tokenizer=tokenizer,
    #     cond_type=cond_type,
    #     special_tokens=special_tokens,
    #     image_folder=WHU_RS19_data_root,
    #     image_processor=image_processor,
    #     extra_image_processor=extra_image_processor,
    #     task_name="imgconv",
    #     data_name="imgconv_WHU-RS19",
    #     dataset_map_fn=dict(
    #         type=dataset_map_fn_factory,
    #         fn=image_conv_map_fn,
    #     ),
    #     template_map_fn=dict(
    #         type=template_map_fn_factory,
    #         template=prompt_template,
    #         output_suffix=True,  # 评估时需要output来计算指标
    #     ),
    #     max_length=max_length,
    #     pixel_values_ndim=2,
    #     is_multimodal=True,
    #     exclude_pure_text=True,
    #     pad_image_to_square=False,
    #     output_ids_with_output=False,  # 评估时需要output来计算指标
    # ),
    # # 10. Image Conversation (Scene Classification) - AID
    # dict(
    #     type=ImgConvDataset,
    #     data_path=AID_data_root + "data.json",
    #     tokenizer=tokenizer,
    #     cond_type=cond_type,
    #     special_tokens=special_tokens,
    #     image_folder=AID_data_root,
    #     image_processor=image_processor,
    #     extra_image_processor=extra_image_processor,
    #     task_name="imgconv",
    #     data_name="imgconv_AID",
    #     dataset_map_fn=dict(
    #         type=dataset_map_fn_factory,
    #         fn=image_conv_map_fn,
    #     ),
    #     template_map_fn=dict(
    #         type=template_map_fn_factory,
    #         template=prompt_template,
    #         output_suffix=True,  # 评估时需要output来计算指标
    #     ),
    #     max_length=max_length,
    #     pixel_values_ndim=2,
    #     is_multimodal=True,
    #     exclude_pure_text=True,
    #     pad_image_to_square=False,
    #     output_ids_with_output=False,  # 评估时需要output来计算指标
    # ),
    # # 11. Image Conversation (Scene Classification) - NWPU-RESISC45
    # dict(
    #     type=ImgConvDataset,
    #     data_path=NWPU_RESISC45_data_root + "data.json",
    #     tokenizer=tokenizer,
    #     cond_type=cond_type,
    #     special_tokens=special_tokens,
    #     image_folder=NWPU_RESISC45_data_root,
    #     image_processor=image_processor,
    #     extra_image_processor=extra_image_processor,
    #     task_name="imgconv",
    #     data_name="imgconv_NWPU-RESISC45",
    #     dataset_map_fn=dict(
    #         type=dataset_map_fn_factory,
    #         fn=image_conv_map_fn,
    #     ),
    #     template_map_fn=dict(
    #         type=template_map_fn_factory,
    #         template=prompt_template,
    #         output_suffix=True,  # 评估时需要output来计算指标
    #     ),
    #     max_length=max_length,
    #     pixel_values_ndim=2,
    #     is_multimodal=True,
    #     exclude_pure_text=True,
    #     pad_image_to_square=False,
    #     output_ids_with_output=False,  # 评估时需要output来计算指标
    # ),
    # # 12. Image Conversation (Scene Classification) - SIRI-WHU
    # dict(
    #     type=ImgConvDataset,
    #     data_path=SIRI_WHU_data_root + "data.json",
    #     tokenizer=tokenizer,
    #     cond_type=cond_type,
    #     special_tokens=special_tokens,
    #     image_folder=SIRI_WHU_data_root,
    #     image_processor=image_processor,
    #     extra_image_processor=extra_image_processor,
    #     task_name="imgconv",
    #     data_name="imgconv_SIRI-WHU",
    #     dataset_map_fn=dict(
    #         type=dataset_map_fn_factory,
    #         fn=image_conv_map_fn,
    #     ),
    #     template_map_fn=dict(
    #         type=template_map_fn_factory,
    #         template=prompt_template,
    #         output_suffix=True,  # 评估时需要output来计算指标
    #     ),
    #     max_length=max_length,
    #     pixel_values_ndim=2,
    #     is_multimodal=True,
    #     exclude_pure_text=True,
    #     pad_image_to_square=False,
    #     output_ids_with_output=False,  # 评估时需要output来计算指标
    # ),
    # # 13. Image Conversation (Scene Classification) - UC_Merced
    # dict(
    #     type=ImgConvDataset,
    #     data_path=UC_Merced_data_root + "data.json",
    #     tokenizer=tokenizer,
    #     cond_type=cond_type,
    #     special_tokens=special_tokens,
    #     image_folder=UC_Merced_data_root,
    #     image_processor=image_processor,
    #     extra_image_processor=extra_image_processor,
    #     task_name="imgconv",
    #     data_name="imgconv_UC_Merced",
    #     dataset_map_fn=dict(
    #         type=dataset_map_fn_factory,
    #         fn=image_conv_map_fn,
    #     ),
    #     template_map_fn=dict(
    #         type=template_map_fn_factory,
    #         template=prompt_template,
    #         output_suffix=True,  # 评估时需要output来计算指标
    #     ),
    #     max_length=max_length,
    #     pixel_values_ndim=2,
    #     is_multimodal=True,
    #     exclude_pure_text=True,
    #     pad_image_to_square=False,
    #     output_ids_with_output=False,  # 评估时需要output来计算指标
    # ),
    # # 14. Image Conversation (Scene Classification) - AID multilabel
    # dict(
    #     type=ImgConvDataset,
    #     data_path=AID_multilabel_data_root + "data.json",
    #     tokenizer=tokenizer,
    #     cond_type=cond_type,
    #     special_tokens=special_tokens,
    #     image_folder=AID_multilabel_data_root,
    #     image_processor=image_processor,
    #     extra_image_processor=extra_image_processor,
    #     task_name="imgconv",
    #     data_name="imgconv_AID_multilabel",
    #     dataset_map_fn=dict(
    #         type=dataset_map_fn_factory,
    #         fn=image_conv_map_fn,
    #     ),
    #     template_map_fn=dict(
    #         type=template_map_fn_factory,
    #         template=prompt_template,
    #         output_suffix=True,  # 评估时需要output来计算指标
    #     ),
    #     max_length=max_length,
    #     pixel_values_ndim=2,
    #     is_multimodal=True,
    #     exclude_pure_text=True,
    #     pad_image_to_square=False,
    #     output_ids_with_output=False,  # 评估时需要output来计算指标
    # ),
    # # 15. Image Conversation (Image Caption) - UCM-Captions
    # dict(
    #     type=ImgConvDataset,
    #     data_path=optical_caption_data_root + "UCM-Captions/dataset_qwenvl.json",
    #     tokenizer=tokenizer,
    #     cond_type=cond_type,
    #     special_tokens=special_tokens,
    #     image_folder=optical_caption_data_root + "UCM-Captions/imgs",
    #     image_processor=image_processor,
    #     extra_image_processor=extra_image_processor,
    #     task_name="imgconv",
    #     data_name="imgconv_UCM-Captions",
    #     dataset_map_fn=dict(
    #         type=dataset_map_fn_factory,
    #         fn=image_conv_map_fn,
    #     ),
    #     template_map_fn=dict(
    #         type=template_map_fn_factory,
    #         template=prompt_template,
    #         output_suffix=True,  # 评估时需要output来计算指标
    #     ),
    #     max_length=max_length,
    #     pixel_values_ndim=2,
    #     is_multimodal=True,
    #     exclude_pure_text=True,
    #     pad_image_to_square=False,
    #     output_ids_with_output=False,  # 评估时需要output来计算指标
    # ),
    # # 16. Image Conversation (Image Caption) - NWPU-Captions
    # dict(
    #     type=ImgConvDataset,
    #     data_path=optical_caption_data_root + "NWPU-Captions/dataset_nwpu_qwenvl.json",
    #     tokenizer=tokenizer,
    #     cond_type=cond_type,
    #     special_tokens=special_tokens,
    #     image_folder=optical_caption_data_root + "NWPU-Captions/NWPU_images",
    #     image_processor=image_processor,
    #     extra_image_processor=extra_image_processor,
    #     task_name="imgconv",
    #     data_name="imgconv_NWPU-Captions",
    #     dataset_map_fn=dict(
    #         type=dataset_map_fn_factory,
    #         fn=image_conv_map_fn,
    #     ),
    #     template_map_fn=dict(
    #         type=template_map_fn_factory,
    #         template=prompt_template,
    #         output_suffix=True,  # 评估时需要output来计算指标
    #     ),
    #     max_length=max_length,
    #     pixel_values_ndim=2,
    #     is_multimodal=True,
    #     exclude_pure_text=True,
    #     pad_image_to_square=False,
    #     output_ids_with_output=False,  # 评估时需要output来计算指标
    # ),
    # # 17. Image Conversation (VQA, complex conversation) - FIT-RSFG-Bench
    # dict(
    #     type=ImgConvDataset,
    #     data_path=imgconv_cc_data_root + "FIT-RSFG/FIT-RSFG-Bench/test_FITRS_complex_comprehension_eval_qwenvl_debug.json",
    #     tokenizer=tokenizer,
    #     cond_type=cond_type,
    #     special_tokens=special_tokens,
    #     image_folder=imgconv_cc_data_root + "raw_data/imgv2_split_512_100_vaild",
    #     image_processor=image_processor,
    #     extra_image_processor=extra_image_processor,
    #     task_name="imgconv",
    #     data_name="imgconv_FIT-RSFG_Benchmark_CC",
    #     dataset_map_fn=dict(
    #         type=dataset_map_fn_factory,
    #         fn=image_conv_map_fn,
    #     ),
    #     template_map_fn=dict(
    #         type=template_map_fn_factory,
    #         template=prompt_template,
    #         output_suffix=True,  # 评估时需要output来计算指标
    #     ),
    #     max_length=max_length,
    #     pixel_values_ndim=2,
    #     is_multimodal=True,
    #     exclude_pure_text=True,
    #     pad_image_to_square=False,
    #     output_ids_with_output=False,  # 评估时需要output来计算指标
    # ),
    # # 18. Image Conversation (Image Caption) - FIT-RSFG-Bench caption
    # dict(
    #     type=ImgConvDataset,
    #     data_path=imgconv_cc_data_root + "FIT-RSFG/FIT-RSFG-Bench/test_FITRS_image_caption_eval_qwenvl_debug.json",
    #     tokenizer=tokenizer,
    #     cond_type=cond_type,
    #     special_tokens=special_tokens,
    #     image_folder=imgconv_cc_data_root + "raw_data/imgv2_split_512_100_vaild",
    #     image_processor=image_processor,
    #     extra_image_processor=extra_image_processor,
    #     task_name="imgconv",
    #     data_name="imgconv_FIT-RSFG_Benchmark_caption",
    #     dataset_map_fn=dict(
    #         type=dataset_map_fn_factory,
    #         fn=image_conv_map_fn,
    #     ),
    #     template_map_fn=dict(
    #         type=template_map_fn_factory,
    #         template=prompt_template,
    #         output_suffix=True,  # 评估时需要output来计算指标
    #     ),
    #     max_length=max_length,
    #     pixel_values_ndim=2,
    #     is_multimodal=True,
    #     exclude_pure_text=True,
    #     pad_image_to_square=False,
    #     output_ids_with_output=False,  # 评估时需要output来计算指标
    # ),
    # # 19. Image Conversation (Region Caption) - FIT-RSFG-Bench region caption
    # dict(
    #     type=ImgConvDataset,
    #     data_path=imgconv_cc_data_root + "FIT-RSFG/FIT-RSFG-Bench/test_FITRS_region_caption_eval_qwen_debug.json",
    #     tokenizer=tokenizer,
    #     cond_type=cond_type,
    #     special_tokens=special_tokens,
    #     image_folder=imgconv_cc_data_root + "raw_data/imgv2_split_512_100_vaild",
    #     image_processor=image_processor,
    #     extra_image_processor=extra_image_processor,
    #     task_name="imgconv",
    #     data_name="imgconv_FIT-RSFG_Benchmark_region_caption",
    #     dataset_map_fn=dict(
    #         type=dataset_map_fn_factory,
    #         fn=image_conv_map_fn,
    #     ),
    #     template_map_fn=dict(
    #         type=template_map_fn_factory,
    #         template=prompt_template,
    #         output_suffix=True,  # 评估时需要output来计算指标
    #     ),
    #     max_length=max_length,
    #     pixel_values_ndim=2,
    #     is_multimodal=True,
    #     exclude_pure_text=True,
    #     pad_image_to_square=False,
    #     output_ids_with_output=False,  # 评估时需要output来计算指标
    # ),
    # # 20. Image Conversation (VQA) - FIT-RSFG-Bench VQA
    # dict(
    #     type=ImgConvDataset,
    #     data_path=imgconv_cc_data_root + "FIT-RSFG/FIT-RSFG-Bench/test_FITRS_vqa_eval_qwenvl_debug.json",
    #     tokenizer=tokenizer,
    #     cond_type=cond_type,
    #     special_tokens=special_tokens,
    #     image_folder=imgconv_cc_data_root + "raw_data/imgv2_split_512_100_vaild",
    #     image_processor=image_processor,
    #     extra_image_processor=extra_image_processor,
    #     task_name="imgconv",
    #     data_name="imgconv_FIT-RSFG_Benchmark_VQA",
    #     dataset_map_fn=dict(
    #         type=dataset_map_fn_factory,
    #         fn=image_conv_map_fn,
    #     ),
    #     template_map_fn=dict(
    #         type=template_map_fn_factory,
    #         template=prompt_template,
    #         output_suffix=True,  # 评估时需要output来计算指标
    #     ),
    #     max_length=max_length,
    #     pixel_values_ndim=2,
    #     is_multimodal=True,
    #     exclude_pure_text=True,
    #     pad_image_to_square=False,
    #     output_ids_with_output=False,  # 评估时需要output来计算指标
    # ),
    # # 21. Image Conversation (Scene Classification) - FIT-RSFG-Bench scene classification
    # dict(
    #     type=ImgConvDataset,
    #     data_path=imgconv_cc_data_root + "FIT-RSFG/FIT-RSFG-Bench/test_FITRS_imageclassify_eval_qwenvl_debug.json",
    #     tokenizer=tokenizer,
    #     cond_type=cond_type,
    #     special_tokens=special_tokens,
    #     image_folder=imgconv_cc_data_root + "raw_data/imgv2_split_512_100_vaild",
    #     image_processor=image_processor,
    #     extra_image_processor=extra_image_processor,
    #     task_name="imgconv",
    #     data_name="imgconv_FIT-RSFG_Benchmark_scene_classification",
    #     dataset_map_fn=dict(
    #         type=dataset_map_fn_factory,
    #         fn=image_conv_map_fn,
    #     ),
    #     template_map_fn=dict(
    #         type=template_map_fn_factory,
    #         template=prompt_template,
    #         output_suffix=True,  # 评估时需要output来计算指标
    #     ),
    #     max_length=max_length,
    #     pixel_values_ndim=2,
    #     is_multimodal=True,
    #     exclude_pure_text=True,
    #     pad_image_to_square=False,
    #     output_ids_with_output=False,  # 评估时需要output来计算指标
    # ),
    # # 22. Image Conversation (VQA) - FIT-RSRC
    # dict(
    #     type=ImgConvDataset,
    #     data_path=imgconv_cc_data_root + "FIT-RSRC/FIT-RSRC_Questions_2k_qwenvl_debug.json",
    #     tokenizer=tokenizer,
    #     cond_type=cond_type,
    #     special_tokens=special_tokens,
    #     image_folder=imgconv_cc_data_root + "raw_data/imgv2_split_512_100_vaild",
    #     image_processor=image_processor,
    #     extra_image_processor=extra_image_processor,
    #     task_name="imgconv",
    #     data_name="imgconv_FIT-RSRC",
    #     dataset_map_fn=dict(
    #         type=dataset_map_fn_factory,
    #         fn=image_conv_map_fn,
    #     ),
    #     template_map_fn=dict(
    #         type=template_map_fn_factory,
    #         template=prompt_template,
    #         output_suffix=True,  # 评估时需要output来计算指标
    #     ),
    #     max_length=max_length,
    #     pixel_values_ndim=2,
    #     is_multimodal=True,
    #     exclude_pure_text=True,
    #     pad_image_to_square=False,
    #     output_ids_with_output=False,  # 评估时需要output来计算指标
    # ),
    # # 23. Image Conversation (VQA) - RSVQA_HR
    # dict(
    #     type=ImgConvDataset,
    #     data_path=imgconv_cc_data_root + "FIT-RSFG/FIT-RSFG-Bench/hrben_qwenvl_debug.json",
    #     tokenizer=tokenizer,
    #     cond_type=cond_type,
    #     special_tokens=special_tokens,
    #     image_folder=rsvqa_hr_data_root + "Data",
    #     image_processor=image_processor,
    #     extra_image_processor=extra_image_processor,
    #     task_name="imgconv",
    #     data_name="imgconv_RSVQA_HR",
    #     dataset_map_fn=dict(
    #         type=dataset_map_fn_factory,
    #         fn=image_conv_map_fn,
    #     ),
    #     template_map_fn=dict(
    #         type=template_map_fn_factory,
    #         template=prompt_template,
    #         output_suffix=True,  # 评估时需要output来计算指标
    #     ),
    #     max_length=max_length,
    #     pixel_values_ndim=2,
    #     is_multimodal=True,
    #     exclude_pure_text=True,
    #     pad_image_to_square=False,
    #     output_ids_with_output=False,  # 评估时需要output来计算指标
    # ),
    # # 24. Image Conversation (VQA) - RSVQA_LR
    # dict(
    #     type=ImgConvDataset,
    #     data_path=imgconv_cc_data_root + "FIT-RSFG/FIT-RSFG-Bench/lrben_qwenvl_debug.json",
    #     tokenizer=tokenizer,
    #     cond_type=cond_type,
    #     special_tokens=special_tokens,
    #     image_folder=rsvqa_lr_data_root + "Images_LR",
    #     image_processor=image_processor,
    #     extra_image_processor=extra_image_processor,
    #     task_name="imgconv",
    #     data_name="imgconv_RSVQA_LR",
    #     dataset_map_fn=dict(
    #         type=dataset_map_fn_factory,
    #         fn=image_conv_map_fn,
    #     ),
    #     template_map_fn=dict(
    #         type=template_map_fn_factory,
    #         template=prompt_template,
    #         output_suffix=True,  # 评估时需要output来计算指标
    #     ),
    #     max_length=max_length,
    #     pixel_values_ndim=2,
    #     is_multimodal=True,
    #     exclude_pure_text=True,
    #     pad_image_to_square=False,
    #     output_ids_with_output=False,  # 评估时需要output来计算指标
    # ),
    # # 25. Image Conversation (Image Caption, SAR) - FuSAR_caption
    # dict(
    #     type=ImgConvDataset,
    #     data_path=SAR_data_root + "sft/test/fusar_clip/caption/test.json",
    #     tokenizer=tokenizer,
    #     cond_type=cond_type,
    #     special_tokens=special_tokens,
    #     image_folder=FuSAR_data_root + "sft",
    #     image_processor=image_processor,
    #     extra_image_processor=extra_image_processor,
    #     task_name="imgconv",
    #     data_name="imgconv_FuSAR_caption",
    #     dataset_map_fn=dict(
    #         type=dataset_map_fn_factory,
    #         fn=image_conv_map_fn,
    #     ),
    #     template_map_fn=dict(
    #         type=template_map_fn_factory,
    #         template=prompt_template,
    #         output_suffix=True,  # 评估时需要output来计算指标
    #     ),
    #     max_length=max_length,
    #     pixel_values_ndim=2,
    #     is_multimodal=True,
    #     exclude_pure_text=True,
    #     pad_image_to_square=False,
    #     output_ids_with_output=False,  # 评估时需要output来计算指标
    # ),
    # # 26. Image Conversation (Image Caption, SAR) - SARLANG_caption
    # dict(
    #     type=ImgConvDataset,
    #     data_path=SAR_data_root + "sft/test/SARLANG-1M/caption/test.json",
    #     tokenizer=tokenizer,
    #     cond_type=cond_type,
    #     special_tokens=special_tokens,
    #     image_folder="/mnt_llm_A100_V1/yangsen/datasets",
    #     image_processor=image_processor,
    #     extra_image_processor=extra_image_processor,
    #     task_name="imgconv",
    #     data_name="imgconv_SARLANG_caption",
    #     dataset_map_fn=dict(
    #         type=dataset_map_fn_factory,
    #         fn=image_conv_map_fn,
    #     ),
    #     template_map_fn=dict(
    #         type=template_map_fn_factory,
    #         template=prompt_template,
    #         output_suffix=True,  # 评估时需要output来计算指标
    #     ),
    #     max_length=max_length,
    #     pixel_values_ndim=2,
    #     is_multimodal=True,
    #     exclude_pure_text=True,
    #     pad_image_to_square=False,
    #     output_ids_with_output=False,  # 评估时需要output来计算指标
    # ),
    # # 27. Image Conversation (Image Caption, SAR) - SARTEXT_caption
    # dict(
    #     type=ImgConvDataset,
    #     data_path=SAR_data_root + "sft/test/sar_text/HRSID_test_caption_qwen.json",
    #     tokenizer=tokenizer,
    #     cond_type=cond_type,
    #     special_tokens=special_tokens,
    #     image_folder="/mnt_llm_A100_V1/yangsen/datasets",
    #     image_processor=image_processor,
    #     extra_image_processor=extra_image_processor,
    #     task_name="imgconv",
    #     data_name="imgconv_SARTEXT_caption",
    #     dataset_map_fn=dict(
    #         type=dataset_map_fn_factory,
    #         fn=image_conv_map_fn,
    #     ),
    #     template_map_fn=dict(
    #         type=template_map_fn_factory,
    #         template=prompt_template,
    #         output_suffix=True,  # 评估时需要output来计算指标
    #     ),
    #     max_length=max_length,
    #     pixel_values_ndim=2,
    #     is_multimodal=True,
    #     exclude_pure_text=True,
    #     pad_image_to_square=False,
    #     output_ids_with_output=False,  # 评估时需要output来计算指标
    # ),
    # # 28. Image Conversation (Image Caption, SAR) - FSAR_caption
    # dict(
    #     type=ImgConvDataset,
    #     data_path=SAR_data_root + "sft/test/FSAR-Cap/caption/test.json",
    #     tokenizer=tokenizer,
    #     cond_type=cond_type,
    #     special_tokens=special_tokens,
    #     image_folder="/mnt_llm_A100_V1/yangsen/datasets",
    #     image_processor=image_processor,
    #     extra_image_processor=extra_image_processor,
    #     task_name="imgconv",
    #     data_name="imgconv_FSAR_caption",
    #     dataset_map_fn=dict(
    #         type=dataset_map_fn_factory,
    #         fn=image_conv_map_fn,
    #     ),
    #     template_map_fn=dict(
    #         type=template_map_fn_factory,
    #         template=prompt_template,
    #         output_suffix=True,  # 评估时需要output来计算指标
    #     ),
    #     max_length=max_length,
    #     pixel_values_ndim=2,
    #     is_multimodal=True,
    #     exclude_pure_text=True,
    #     pad_image_to_square=False,
    #     output_ids_with_output=False,  # 评估时需要output来计算指标
    # ),
    # # 29. Image Conversation (VQA, SAR) - FuSAR_VQA
    # dict(
    #     type=ImgConvDataset,
    #     data_path=SAR_data_root + "sft/test/fusar_clip/VQA/test_all_categories.json",
    #     tokenizer=tokenizer,
    #     cond_type=cond_type,
    #     special_tokens=special_tokens,
    #     image_folder=FuSAR_data_root + "sft",
    #     image_processor=image_processor,
    #     extra_image_processor=extra_image_processor,
    #     task_name="imgconv",
    #     data_name="imgconv_FuSAR_VQA",
    #     dataset_map_fn=dict(
    #         type=dataset_map_fn_factory,
    #         fn=image_conv_map_fn,
    #     ),
    #     template_map_fn=dict(
    #         type=template_map_fn_factory,
    #         template=prompt_template,
    #         output_suffix=True,  # 评估时需要output来计算指标
    #     ),
    #     max_length=max_length,
    #     pixel_values_ndim=2,
    #     is_multimodal=True,
    #     exclude_pure_text=True,
    #     pad_image_to_square=False,
    #     output_ids_with_output=False,  # 评估时需要output来计算指标
    # ),
    # # 30. Image Conversation (VQA, SAR) - SAR-LANG_VQA
    # dict(
    #     type=ImgConvDataset,
    #     data_path=SAR_data_root + "sft/test/SARLANG-1M/VQA/test.json",
    #     tokenizer=tokenizer,
    #     cond_type=cond_type,
    #     special_tokens=special_tokens,
    #     image_folder="/mnt_llm_A100_V1/yangsen/datasets",
    #     image_processor=image_processor,
    #     extra_image_processor=extra_image_processor,
    #     task_name="imgconv",
    #     data_name="imgconv_SAR-LANG_VQA",
    #     dataset_map_fn=dict(
    #         type=dataset_map_fn_factory,
    #         fn=image_conv_map_fn,
    #     ),
    #     template_map_fn=dict(
    #         type=template_map_fn_factory,
    #         template=prompt_template,
    #         output_suffix=True,  # 评估时需要output来计算指标
    #     ),
    #     max_length=max_length,
    #     pixel_values_ndim=2,
    #     is_multimodal=True,
    #     exclude_pure_text=True,
    #     pad_image_to_square=False,
    #     output_ids_with_output=False,  # 评估时需要output来计算指标
    # ),
]

val_evaluators = [
    # # 1. Generic Segmentation (genseg) - SOTA validation
    # dict(
    #     type=GenSegEvaluator,
    #     distributed=True,
    #     data_name="panoptic_genseg_sota_val",
    # ),
    # # 2. Open-Vocabulary Segmentation (ovseg) - SOTA validation
    # dict(
    #     type=OVSegEvaluator,
    #     data_name="panoptic_ovseg_sota_val",
    #     distributed=True,
    # ),
    # # 3. Referring Segmentation (refseg) - RemoteSAM validation
    # dict(
    #     type=RefSegEvaluator,
    #     distributed=True,
    #     data_name="refseg_remotesam_val",
    # ),
    # # 4. Referring Segmentation (refseg) - RemoteSAM test
    # dict(
    #     type=RefSegEvaluator,
    #     distributed=True,
    #     data_name="refseg_remotesam_test",
    # ),
    # # 5. Referring Segmentation (refseg) - refsegrs test
    # dict(
    #     type=RefSegEvaluator,
    #     distributed=True,
    #     data_name="refseg_refsegrs_test",
    # ),
    # # 6. Referring Segmentation (refseg) - RRSISD test
    # dict(
    #     type=RefSegEvaluator,
    #     distributed=True,
    #     data_name="refseg_rrsisd_test",
    # ),
    # # 7. Referring Segmentation (refseg) - RISBench test
    # dict(
    #     type=RefSegEvaluator,
    #     distributed=True,
    #     data_name="refseg_risbench_test",
    # ),
    # # 7. Referring Segmentation (refseg) - fast val
    # dict(
    #     type=RefSegEvaluator,
    #     distributed=True,
    #     data_name="refseg_fast_val",
    # ),
    # 8. Reasoning Segmentation (reasonseg) - ReasonSeg validation
    dict(
        type=ReasonSegEvaluator,
        distributed=True,
       data_name="reaseg_earthreason_test",
    ),
    # # 8. Reasoning Segmentation (reasonseg) - ReasonSeg validation
    # dict(
    #     type=ReasonSegEvaluator,
    #     distributed=True,
    #    data_name="reaseg_diy1_test",
    # ),
    # # 9. Image Conversation (Scene Classification) - WHU-RS19
    # dict(
    #     type=ImgConvEvaluator,
    #     distributed=True,
    #     data_name="imgconv_WHU-RS19",
    #     metrics= ["accuracy"],
    # ),
    # # 10. Image Conversation (Scene Classification) - AID
    # dict(
    #     type=ImgConvEvaluator,
    #     distributed=True,
    #     data_name="imgconv_AID",
    #     metrics= ["accuracy"],
    # ),
    # # 11. Image Conversation (Scene Classification) - NWPU-RESISC45
    # dict(
    #     type=ImgConvEvaluator,
    #     distributed=True,
    #     data_name="imgconv_NWPU-RESISC45",
    #     metrics= ["accuracy"],
    # ),
    # # 12. Image Conversation (Scene Classification) - SIRI-WHU
    # dict(
    #     type=ImgConvEvaluator,
    #     distributed=True,
    #     data_name="imgconv_SIRI-WHU",
    #     metrics= ["accuracy"],
    # ),
    # # 13. Image Conversation (Scene Classification) - UC_Merced
    # dict(
    #     type=ImgConvEvaluator,
    #     distributed=True,
    #     data_name="imgconv_UC_Merced",
    #     metrics= ["accuracy"],
    # ),
    # # 14. Image Conversation (Scene Classification) - AID multilabel
    # dict(
    #     type=ImgConvMLSCEvaluator,
    #     distributed=True,
    #     data_name="imgconv_AID_multilabel"
    # ),
    # # 15. Image Conversation (Image Caption) - UCM-Captions
    # dict(
    #     type=ImgConvEvaluator,
    #     distributed=True,
    #     data_name="imgconv_UCM-Captions",
    #     metrics=["bleu4", "meteor", "rougeL", "cider"],
    # ),
    # # 16. Image Conversation (Image Caption) - NWPU-Captions
    # dict(
    #     type=ImgConvEvaluator,
    #     distributed=True,
    #     data_name="imgconv_NWPU-Captions",
    #     metrics= ["meteor", "rougeL", "cider"],
    # ),
    # # 17. Image Conversation (VQA, complex conversation) - FIT-RSFG-Bench
    # dict(
    #     type=ImgConvCCEvaluator,
    #     distributed=True,
    #     data_name="imgconv_FIT-RSFG_Benchmark_CC",
    # ),
    # # 18. Image Conversation (Image Caption) - FIT-RSFG-Bench caption
    # dict(
    #     type=ImgConvEvaluator,
    #     distributed=True,
    #     data_name="imgconv_FIT-RSFG_Benchmark_caption",
    #     metrics=["bleu1", "bleu2", "bleu3", "bleu4", "meteor", "rougeL"],
    # ),
    # # 19. Image Conversation (Region Caption) - FIT-RSFG-Bench region caption
    # dict(
    #     type=ImgConvEvaluator,
    #     distributed=True,
    #     data_name="imgconv_FIT-RSFG_Benchmark_region_caption",
    #     metrics=["bleu1", "bleu2", "bleu3", "bleu4", "meteor", "rougeL"],
    # ),
    # # 20. Image Conversation (VQA) - FIT-RSFG-Bench VQA
    # dict(
    #     type=ImgConvEvaluator,
    #     distributed=True,
    #     data_name="imgconv_FIT-RSFG_Benchmark_VQA",
    #     metrics=["accuracy"],
    # ),
    # # 21. Image Conversation (Scene Classification) - FIT-RSFG-Bench scene classification
    # dict(
    #     type=ImgConvMLSCEvaluator,
    #     distributed=True,
    #     data_name="imgconv_FIT-RSFG_Benchmark_scene_classification",
    # ),
    # # 22. Image Conversation (VQA) - FIT-RSRC
    # dict(
    #     type=ImgConvEvaluator,
    #     distributed=True,
    #     data_name="imgconv_FIT-RSRC",
    #     metrics=["accuracy"],
    # ),
    # # 23. Image Conversation (VQA) - RSVQA_HR
    # dict(
    #     type=ImgConvEvaluator,
    #     distributed=True,
    #     data_name="imgconv_RSVQA_HR",
    #     metrics=["accuracy"],
    # ),
    # # 24. Image Conversation (VQA) - RSVQA_LR
    # dict(
    #     type=ImgConvEvaluator,
    #     distributed=True,
    #     data_name="imgconv_RSVQA_LR",
    #     metrics=["accuracy"],
    # ),
    # # 25. Image Conversation (Image Caption, SAR) - FuSAR_caption
    # dict(
    #     type=ImgConvEvaluator,
    #     distributed=True,
    #     data_name="imgconv_FuSAR_caption",
    #     metrics=["bleu4", "meteor", "cider","spice"],
    # ),
    # # 26. Image Conversation (Image Caption, SAR) - SARLANG_caption
    # dict(
    #     type=ImgConvEvaluator,
    #     distributed=True,
    #     data_name="imgconv_SARLANG_caption",
    #     metrics=["bleu1", "bleu2", "bleu3", "bleu4", "rougeL", "cider"],
    # ),
    # # 27. Image Conversation (Image Caption, SAR) - SARTEXT_caption
    # dict(
    #     type=ImgConvEvaluator,
    #     distributed=True,
    #     data_name="imgconv_SARTEXT_caption",
    #     metrics=["bleu1", "bleu2", "bleu3", "bleu4", "rougeL", "cider", "spice"],
    # ),
    # # 28. Image Conversation (Image Caption, SAR) - FSAR_caption
    # dict(
    #     type=ImgConvEvaluator,
    #     distributed=True,
    #     data_name="imgconv_FSAR_caption",
    #     metrics=["bleu1", "bleu4", "rougeL", "cider", "spice"],
    # ),
    # # 29. Image Conversation (VQA, SAR) - FuSAR_VQA
    # dict(
    #     type=ImgConvEvaluator,
    #     distributed=True,
    #     data_name="imgconv_FuSAR_VQA",
    #     metrics=["accuracy"],
    # ),
    # # 30. Image Conversation (VQA, SAR) - SAR-LANG_VQA
    # dict(
    #     type=ImgConvEvaluator,
    #     distributed=True,
    #     data_name="imgconv_SAR-LANG_VQA",
    #     metrics=["accuracy"],
    # ),
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



