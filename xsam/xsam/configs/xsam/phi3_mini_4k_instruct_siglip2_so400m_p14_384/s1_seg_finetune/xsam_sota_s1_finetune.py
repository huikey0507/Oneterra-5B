from copy import deepcopy
from os import getenv

import torch
from mmengine.hooks import CheckpointHook, DistSamplerSeedHook, IterTimerHook, LoggerHook, ParamSchedulerHook
from mmengine.optim import AmpOptimWrapper, LinearLR, MultiStepLR
from torch.optim import AdamW
from xtuner.dataset.samplers import LengthGroupedSampler
from mmengine.dataset import DefaultSampler

from xsam.dataset import GenericSegDataset
from xsam.dataset.collate_fns import xsam_collate_fn
from xsam.dataset.process_fns import generic_seg_postprocess_fn, process_map_fn_factory
from xsam.dataset.processors import SamImageProcessor
from xsam.engine.hooks import DatasetInfoHook, ModelInfoHook, PTCheckpointHook
from xsam.engine.runners import TrainLoop, ValLoop
from xsam.evaluation.evaluators import GenericSegEvaluator
from xsam.model import XSamModel
from xsam.model.segmentors import XSegmentor
from xsam.model.segmentors.mask2former import Mask2FormerConfig, Mask2FormerModel
from xsam.model.segmentors.sam import SamModel

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Directories
code_dir = getenv("CODE_DIR", "./xsam/")
data_dir = getenv("DATA_DIR", "./datas/")  # 修改为datas
init_dir = getenv("INIT_DIR", "./inits/")
work_dir = getenv("WORK_DIR", "./wkdrs/")

# Model
seg_encoder_name_or_path = init_dir + "sam-vit-large"
seg_decoder_name_or_path = init_dir + "mask2former-swin-large-coco-panoptic"

# Data - 多数据集配置
# SOTA数据集路径
sota_data_root = data_dir + "sota/"
sota_train_data_path = sota_data_root + "train_annotations.json"
sota_train_image_folder = sota_data_root + "train/images"
sota_train_panseg_map_folder = sota_data_root + "train/panoptic_labels"

# LandDiscover数据集路径
landdiscover_data_root = "/mnt_llm_A100_V1/pxy/data/landdiscover_panoptic/train/SIOR/"
landdiscover_train_data_path = landdiscover_data_root + "gt_remap_panoptic_coco_dense/annotations/panoptic_annotations.json"
landdiscover_train_image_folder = landdiscover_data_root + "img/"
landdiscover_train_panseg_map_folder = landdiscover_data_root + "gt_remap_panoptic_coco_dense/panoptic_rgb_ids/"

# 验证集路径（训练时不使用，仅用于后续评估）
# val_data_path = data_root + "val_annotations.json"
# val_image_folder = data_root + "val/images" 
# val_panseg_map_folder = data_root + "val/panoptic_labels"

# Scheduler & Optimizer - 调整训练参数
batch_size = 2  # 减少批次大小，避免内存不足
accumulative_counts = 4  # 增加累积步数，保持有效batch_size=4
dataloader_num_workers = 2  # 进一步减少worker数量
max_epochs = 36  # 减少训练轮数，先快速验证
optim_type = AdamW
lr = 1e-4
betas = (0.9, 0.999)
weight_decay = 0.05
max_norm = 0.01  # grad clip
warmup_ratio = 0.1  # 增加warmup比例

# Save
save_steps = 1000  # 减少保存频率
save_total_limit = 5  # 增加保存的checkpoint数量

# Logging
logging_interval = 100

#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
# TODO: add special tokens via import from xsam.utils

extra_image_processor = dict(
    type=SamImageProcessor.from_pretrained,
    pretrained_model_name_or_path=seg_encoder_name_or_path,
    trust_remote_code=True,
    ignore_index=0,
)

model = dict(
    type=XSamModel,
    freeze_segmentor_encoder=True,
    use_activation_checkpointing=True,  # 启用梯度检查点，节省内存
    postprocess_fn=generic_seg_postprocess_fn,
    connector_type="conv",
    seg_select_layers=[6, 12, 18, 24],
    connector_hidden_dim=512,
    connector_scale_factor=[4, 2, 1, 0.5],
    # 加载第一阶段COCO数据集训练的预训练权重
    s1_pretrained_pth="/mnt_llm_A100_V1/shui/LAE/X-SAM/X-SAM/checkpoints/s1_seg_finetune/xsam_sam_large_m2f_e36_gpu16_seg_finetune/pytorch_model.bin",
    segmentor=dict(
        type=XSegmentor,
        encoder=dict(
            type=SamModel.from_pretrained,
            pretrained_model_name_or_path=seg_encoder_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ),
        decoder=dict(
            type=Mask2FormerModel._from_config,
            config=dict(
                type=Mask2FormerConfig.from_pretrained,
                pretrained_model_name_or_path=seg_decoder_name_or_path,
                use_backbone=False,
                feature_channels=[512, 1024, 2048],
                num_feature_levels=3,
                num_labels=39,  # 设置为您自己的数据集的39类
                trust_remote_code=True,
            ),
            torch_dtype=torch.bfloat16,
        ),
        torch_dtype=torch.bfloat16,
        reinit_decoder=True,
        close_cls=True,
    ),
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
train_extra_image_processor = deepcopy(extra_image_processor)
train_extra_image_processor.update(
    {
        "size": {"min_scale": 0.1, "max_scale": 2.0, "target_size": 1024},
        "do_crop": True,
        "crop_size": {"height": 1024, "width": 1024},
    }
)

# SOTA训练集配置
sota_panoptic_genseg_dataset = dict(
    type=GenericSegDataset,
    data_path=sota_train_data_path,  # SOTA训练集标注文件
    image_folder=sota_train_image_folder,
    panseg_map_folder=sota_train_panseg_map_folder,
    extra_image_processor=train_extra_image_processor,
    task_name="genseg",
    data_name="sota_panoptic_genseg_train",  # SOTA训练集名称
    pad_image_to_square=True,
)

# LandDiscover训练集配置
landdiscover_panoptic_genseg_dataset = dict(
    type=GenericSegDataset,
    data_path=landdiscover_train_data_path,  # LandDiscover训练集标注文件
    image_folder=landdiscover_train_image_folder,
    panseg_map_folder=landdiscover_train_panseg_map_folder,
    extra_image_processor=train_extra_image_processor,
    task_name="genseg",
    data_name="landdiscover_panoptic_genseg_train",  # LandDiscover训练集名称
    pad_image_to_square=True,
)

# 合并数据集配置
from xsam.dataset import ConcatDataset
combined_dataset = dict(
    type=ConcatDataset,
    datasets=[sota_panoptic_genseg_dataset, landdiscover_panoptic_genseg_dataset],
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    pin_memory=True,
    dataset=combined_dataset,  # 使用合并的数据集
    sampler=dict(
        type=DefaultSampler,
        shuffle=True
    ),
    collate_fn=dict(type=xsam_collate_fn),
)

# 验证集配置（训练时不使用）
# val_datasets = [...]

# 验证评估器配置（训练时不使用）
# val_evaluator = dict(...)

# 验证数据加载器配置（训练时不使用）
# val_dataloader = dict(...)

# 验证配置（训练时不使用）
# val_cfg = dict(type="ValLoop")

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, norm_type=2, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale="dynamic",
    dtype="float16",
    paramwise_cfg=dict(
        custom_keys={
            "segmentor.encoder": dict(lr_mult=0.1, decay_mult=1.0),
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
        type=MultiStepLR,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        by_epoch=True,
        milestones=[24, 30],  # 调整milestone以适应36个epoch
        gamma=0.1,
        convert_to_iter_based=True,
    ),
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

# 验证频率设置（训练时不使用）
# val_interval = 1000

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# set visualizer
visualizer = None

# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(
        type=ModelInfoHook,
        module_names=["llm", "connector", "segmentor.encoder", "segmentor.pixel_decoder", "segmentor.decoder"],
        display_params=True,
    ),
    dict(type=DatasetInfoHook),
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
    # 验证钩子（训练时不使用）
    # val=dict(
    #     type="ValLoop",
    #     interval=val_interval,
    # ),
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

find_unused_parameters = True 