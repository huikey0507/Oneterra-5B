from copy import deepcopy
from os import getenv
import os

import torch
from mmengine.dataset import DefaultSampler

from xsam.dataset import GenericSegDataset
from xsam.dataset.collate_fns import xsam_collate_fn
from xsam.dataset.process_fns import generic_seg_postprocess_fn
from xsam.dataset.processors import SamImageProcessor
from xsam.evaluation.evaluators import GenericSegEvaluator

#######################################################################
#                         验证配置设置                                 #
#######################################################################

# 获取项目根目录的绝对路径
# 配置文件位于: xsam/xsam/configs/xsam/.../validate_s1.py
# 需要向上4级到达项目根目录
config_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(config_file_dir))))

# 打印调试信息
print(f"Config file directory: {config_file_dir}")
print(f"Project root: {project_root}")

# 目录设置 - 使用绝对路径
code_dir = getenv("CODE_DIR", os.path.join(project_root, "xsam"))
data_dir = getenv("DATA_DIR", os.path.join(project_root, "xsam_data"))
init_dir = getenv("INIT_DIR", os.path.join(project_root, "inits"))

# 模型设置 - 使用绝对路径
seg_encoder_name_or_path = os.path.join(init_dir, "sam-vit-large")
seg_decoder_name_or_path = os.path.join(init_dir, "mask2former-swin-large-coco-panoptic")

# 打印模型路径
print(f"Encoder path: {seg_encoder_name_or_path}")
print(f"Decoder path: {seg_decoder_name_or_path}")

# 数据设置 - 验证集路径
data_root = os.path.join(data_dir, "sota")
val_data_path = os.path.join(data_root, "val_annotations.json")  # 验证集标注文件
val_image_folder = os.path.join(data_root, "val/images")  # 验证集图像文件夹
val_panseg_map_folder = os.path.join(data_root, "val/panoptic_labels")  # 验证集RGB标签文件夹

# 验证设置
batch_size = 1  # 验证时使用较小的批次大小
dataloader_num_workers = 2

#######################################################################
#           模型配置                                                   #
#######################################################################

extra_image_processor = dict(
    type="SamImageProcessor.from_pretrained",
    pretrained_model_name_or_path=seg_encoder_name_or_path,
    trust_remote_code=True,
    ignore_index=0,
)

model = dict(
    type="XSamModel",
    freeze_segmentor_encoder=False,
    use_activation_checkpointing=False,  # 验证时不需要梯度检查点
    postprocess_fn="generic_seg_postprocess_fn",
    connector_type="conv",
    seg_select_layers=[6, 12, 18, 24],
    connector_hidden_dim=512,
    connector_scale_factor=[4, 2, 1, 0.5],
    segmentor=dict(
        type="XSegmentor",
        encoder=dict(
            type="SamModel.from_pretrained",
            pretrained_model_name_or_path=seg_encoder_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ),
        decoder=dict(
            type="Mask2FormerModel._from_config",
            config=dict(
                type="Mask2FormerConfig.from_pretrained",
                pretrained_model_name_or_path=seg_decoder_name_or_path,
                use_backbone=False,
                feature_channels=[512, 1024, 2048],
                num_queries=100,
                num_transformer_enc_layers=6,
                num_transformer_dec_layers=6,
                num_feature_levels=3,
                enforce_input_proj=False,
                mask_predictor_hidden_dim=256,
                num_classes=133,  # COCO类别数
            ),
        ),
    ),
)

#######################################################################
#           数据集配置                                                 #
#######################################################################

# 验证数据集
val_dataset = dict(
    type="GenericSegDataset",
    data_root=data_root,
    ann_file=val_data_path,
    data_prefix=dict(
        img=val_image_folder,
        seg=val_panseg_map_folder,
    ),
    pipeline=[
        dict(type="LoadImageFromFile"),
        dict(type="LoadPanopticAnnotations"),
        dict(type="Resize", scale=(384, 384), keep_ratio=False),
        dict(type="PackInputs"),
    ],
)

# 验证数据加载器
val_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=val_dataset,
    sampler=dict(type="DefaultSampler", shuffle=False),
    collate_fn="xsam_collate_fn",
    persistent_workers=True,
)

#######################################################################
#           评估器配置                                                 #
#######################################################################

val_evaluator = dict(
    type="GenericSegEvaluator",
    data_name="panoptic_genseg",
    output_dir="./validation_results/evaluation",
    distributed=False,
    show_categories=True,
)

#######################################################################
#           其他设置                                                   #
#######################################################################

# 随机种子
randomness = dict(seed=42, deterministic=False)

# 日志设置
log_level = "INFO"
log_processor = dict(type="LogProcessor", window_size=50, by_epoch=False)

# 可视化设置
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    type="Visualizer",
    vis_backends=vis_backends,
    name="visualizer"
) 