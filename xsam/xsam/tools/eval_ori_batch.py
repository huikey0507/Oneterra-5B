#!/usr/bin/env python

import argparse
import copy
import os
import os.path as osp
import sys

import mmcv
import numpy as np

# 添加项目根目录到Python路径
# 获取当前文件的目录，然后向上找到项目根目录（包含xsam目录的目录）
current_dir = osp.dirname(osp.abspath(__file__))
# eval.py 在 xsam/xsam/tools/ 下，需要向上3级到项目根目录
project_root = osp.dirname(osp.dirname(osp.dirname(current_dir)))
# 添加项目根目录和xsam目录到Python路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# 同时添加xsam目录（因为xsam模块在xsam/xsam/下）
xsam_dir = osp.join(project_root, "xsam")
if xsam_dir not in sys.path:
    sys.path.insert(0, xsam_dir)

import re
import traceback
import warnings
import math
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from mmengine.config import Config, DictAction
from mmengine.runner.utils import set_random_seed
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import GenerationConfig, StoppingCriteriaList
from xtuner.configs import cfgs_name_path
from xtuner.registry import BUILDER
from xtuner.tools.utils import set_model_resource
from xsam.dataset.collate_fns import xsam_collate_fn
from xsam.utils.checkpoint import load_checkpoint
from xsam.utils.config import setup_model_config
from xsam.utils.constants import DEFAULT_SEG_TOKEN
from xsam.utils.dist import setup_distributed
from xsam.utils.logging import print_log, set_default_logging_format
from xsam.utils.misc import data_dict_to_device
from xsam.utils.utils import register_function
from xsam.dataset.utils.process import sem_seg_postprocess
from xsam.structures import BitMasks, Boxes, BoxMode, Instances, RotatedBoxes
from xsam.utils.visualize import Visualizer
from xsam.utils.visualize_eval import EvalVisualizer
from xsam.utils.xtuner_patch import patch_xtuner_llama_attn_for_single_gpu

# Xtuner llama_attn_forward calls dist.get_rank(); single-GPU eval (launcher=none) has no process group.
patch_xtuner_llama_attn_for_single_gpu()

# Global setup
set_default_logging_format()
warnings.filterwarnings("ignore")


def _build_visualizer_from_cfg(cfg):
    """评测使用 EvalVisualizer，与 demo 的 Visualizer 解耦。"""
    if not hasattr(cfg, "visualizer") or cfg.visualizer is None:
        return None
    vis_cfg = cfg.visualizer
    if isinstance(vis_cfg, dict):
        vis_type = vis_cfg.get("type")
        kwargs = {k: v for k, v in vis_cfg.items() if k != "type"}
        if vis_type is EvalVisualizer:
            return EvalVisualizer(**kwargs)
        if vis_type is Visualizer:
            return EvalVisualizer(**kwargs)
        if vis_type is not None and not isinstance(vis_type, str):
            return vis_type(**kwargs)
    return EvalVisualizer()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate model (multi-GPU / multi-batch)")
    parser.add_argument("config", help="config file name or path")
    parser.add_argument("--work-dir", help="directory to save logs and models")
    parser.add_argument(
        "--pth_model",
        type=str,
        default=None,
        help="path to model checkpoint for evaluation",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override config options, format: xxx=yyy",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="inference batch size per GPU")
    parser.add_argument("--num-workers", type=int, default=4, help="dataloader num_workers per GPU")
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="skip saving pred/gt visualization PNGs (much faster)",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher type",
    )
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    return parser.parse_args()


def get_gcg_phrases(input_ids, tokenizer, pstart_token_idx, pend_token_idx):
    pstart_idx = [i for i, x in enumerate(input_ids) if x == pstart_token_idx]
    pend_idx = [i + 1 for i, x in enumerate(input_ids) if x == pend_token_idx]
    phrases = []
    for ps, pe in zip(pstart_idx, pend_idx):
        phrase_ids = input_ids[ps + 1 : pe - 1]
        if (phrase_ids < 0).any():
            phrase = ""
        else:
            phrase = tokenizer.decode(phrase_ids).strip()
        phrases.append(phrase)
    return phrases


def get_gcg_caption(llm_generation_output):
    if DEFAULT_SEG_TOKEN not in llm_generation_output:
        return ""

    parts = llm_generation_output.split(".")
    sents = [part.strip() for part in parts if DEFAULT_SEG_TOKEN not in part]
    caption = ". ".join(sents)
    caption = re.sub(r"<.*?>", "", caption)
    caption = " ".join(caption.split()).strip("'").strip()
    return caption

def get_img_conv_prediction(llm_outputs, tokenizer, stop_criteria):
    """从LLM输出中提取图像对话任务的预测答案字符串"""
    predictions = []
    if llm_outputs is not None and hasattr(llm_outputs, "sequences"):
        llm_outputs = tokenizer.batch_decode(llm_outputs.sequences)
        for llm_output in llm_outputs:
            prediction = llm_output.strip()
            for stop_crit in stop_criteria:
                stop_word = stop_crit.stop_word
                if stop_word:
                    prediction = prediction.split(stop_word)[0]
            predictions.append(prediction)
    return predictions


def process_batch(
    model,
    data: Dict,
    data_name: str,
    metadata: Dict,
    generation_config: Optional[GenerationConfig] = None,
    stop_criteria: Optional[StoppingCriteriaList] = None,
    mode: str = "tensor",
) -> Tuple[bool, Optional[torch.Tensor]]:
    """Process a single batch of data.

    Args:
        model: The model to evaluate
        data: Input data dictionary
        data_name: Name of the dataset
        generation_config: Generation configuration for LLM
        stop_criteria: Stopping criteria for LLM
        mode: Mode of the model

    Returns:
        Tuple of (success status, segmentation outputs)
    """
    data_samples = data["data_samples"]
    image_files = data_samples.image_files

    data_dict = {
        "input_ids": data["data_dict"].get("input_ids", None),
        "pixel_values": data["data_dict"].get("pixel_values", None),
        "seg_pixel_values": data["data_dict"].get("seg_pixel_values", None),
        "cond_ids": data["data_dict"].get("cond_ids", None),
        "seg_ids": data["data_dict"].get("seg_ids", None),
        "vprompt_masks": data["data_dict"].get("vprompt_masks", None),
    }

    llm_question_input = ""
    if data_dict["input_ids"] is not None:
        _input_ids = data_dict["input_ids"]
        llm_question_input = model.tokenizer.decode(_input_ids[_input_ids > 0])

    data_dict = data_dict_to_device(data_dict, device=model.device, dtype=model.dtype)
    with torch.no_grad():
        llm_outputs, seg_outputs = model(
            data_dict,
            data_samples,
            mode=mode,
            generation_config=generation_config,
            stopping_criteria=stop_criteria,
            metadata=metadata,
            do_postprocess=True,
            do_loss=False,
        )

    # print(f"Processed batch for data_name: {data_name}, image_files: {image_files}")
    if "imgconv" in data_name:
        llm_outputs = get_img_conv_prediction(llm_outputs, model.tokenizer, stop_criteria)
        return True, None, llm_outputs
    else:
        if seg_outputs is None:
            llm_generation_output = ""
            if llm_outputs is not None and hasattr(llm_outputs, "sequences"):
                llm_generation_output = model.tokenizer.batch_decode(llm_outputs.sequences)

            print_log(
                rf"Failed to get segmentation outputs: {image_files}, "
                rf"llm question_input: {repr(llm_question_input)}, "
                rf"llm generation_output: {repr(llm_generation_output)}",
                logger="current",
            )
            return False, None, None
        else:
            if "gcg" in data_name and llm_outputs is not None and hasattr(llm_outputs, "sequences"):
                print_log(
                    f"Processing GCG outputs for data_name: {data_name}, image_files: {image_files}",
                    logger="current",
                )
                llm_generation_output = model.tokenizer.batch_decode(llm_outputs.sequences)
                gcg_phrases = [
                    get_gcg_phrases(output_ids, model.tokenizer, model.pstart_token_idx, model.pend_token_idx)
                    for output_ids in llm_outputs.sequences
                ]
                gcg_captions = [get_gcg_caption(output) for output in llm_generation_output]
                for i, segmentation_output in enumerate(seg_outputs):
                    segmentation_output.update({"gcg_phrases": gcg_phrases[i], "gcg_caption": gcg_captions[i]})

            return True, seg_outputs, None

def resize_mask_gt(mask_gt, target_hw, scaled_size):
    target_hw = np.array(target_hw)
    # print_log(f"scaled_size: {scaled_size}, target_hw: {target_hw}", logger="current")
    mask_gt = sem_seg_postprocess(mask_gt, scaled_size, target_hw[0], target_hw[1], mode="nearest")
    return mask_gt

def _panoptic_class_index_to_dataset_id(class_index: int, sampled_labels) -> int:
    """将 dataloader 中的 class 索引还原为 COCO/dataset category_id（用于可视化命名与 isthing）。"""
    if sampled_labels is None:
        return int(class_index)
    if 0 <= int(class_index) < len(sampled_labels):
        return int(sampled_labels[int(class_index)])
    return int(class_index)


def _panoptic_pred_segments_info_for_vis(segments_info, metadata):
    """将预测 segments_info 的 category_id 统一转换为 dataset category_id。

    约定可视化链路只使用 dataset_id 语义：
    - Pred: 后处理输出 contiguous_id，这里显式映射到 dataset_id
    - GT: prepare_gt_data_pan 已输出 dataset_id
    """
    if segments_info is None or not isinstance(segments_info, list):
        return segments_info
    ds_to_cont = getattr(metadata, "dataset_id_to_contiguous_id", None) or {}
    if not ds_to_cont:
        return segments_info
    cont_to_ds = {int(v): int(k) for k, v in ds_to_cont.items()}
    thing_ds_keys = set(getattr(metadata, "thing_dataset_id_to_contiguous_id", None) or {})
    out = []
    for s in segments_info:
        t = dict(s)
        cid = t.get("category_id")
        if cid is not None:
            cid = int(cid)
            # compute_segments 输出使用 contiguous_id，统一映射回 dataset_id。
            # 若出现未知 id，回退保留原值，避免可视化流程中断。
            ds_cat = cont_to_ds.get(cid, cid)
            t["category_id"] = ds_cat
            t["isthing"] = ds_cat in thing_ds_keys
        out.append(t)
    return out


def prepare_gt_data_pan(mask_gt, class_gt, segmentation_output, metadata, scaled_size, image_info=None):
    # mask_gt的shape是[N, H, W], 需要resize跟segmentation_output["segmentation"]的shape一致
    pred_seg = segmentation_output["segmentation"]
    target_hw = tuple(pred_seg.shape[-2:])  # 更稳：只取(H,W)
    mask_gt = resize_mask_gt(mask_gt, target_hw, scaled_size)
    id_num = mask_gt.shape[0]
    # 实例/段 id 必须从 1 开始：0 保留给背景/VOID。若用 arange(0,N) 则第一个 mask 乘 0 会整层消失，
    # 且 id=0 会与背景像素混淆，导致 draw_panoptic 文字贴到整图背景上。
    seg_ids = torch.arange(1, id_num + 1, device=mask_gt.device, dtype=mask_gt.dtype)
    mask_gt = mask_gt * seg_ids.view(-1, 1, 1)
    mask_gt = mask_gt.sum(dim=0)

    sampled_labels = image_info.get("sampled_labels") if image_info is not None else None
    thing_ds_ids = getattr(metadata, "thing_dataset_id_to_contiguous_id", None) or {}

    cls_flat = class_gt.reshape(-1)
    if cls_flat.numel() != id_num:
        print_log(
            f"prepare_gt_data_pan: mask count {id_num} != class_labels count {cls_flat.numel()}, "
            f"truncating to min length for vis.",
            logger="current",
        )
    n = min(int(id_num), int(cls_flat.numel()))

    segments_info_gt = []
    for i in range(n):
        class_index = int(cls_flat[i].item())
        dataset_cat_id = _panoptic_class_index_to_dataset_id(class_index, sampled_labels)
        isthing = dataset_cat_id in thing_ds_ids
        segments_info_gt.append(
            {
                "id": i + 1,
                "category_id": dataset_cat_id,
                "score": 1.0,
                "isthing": isthing,
            }
        )

    gt = {"segmentation": mask_gt, "segments_info": segments_info_gt}

    return gt

def prepare_gt_data_refseg(mask_gt,class_gt,segmentation_output, scaled_size):
    pred_seg = segmentation_output["segmentation"]
    target_hw = tuple(pred_seg.shape[-2:])  # 更稳：只取(H,W)
    mask_gt = resize_mask_gt(mask_gt, target_hw, scaled_size).squeeze(0)  # [H, W]
    mask_gt = torch.where(mask_gt > 0, class_gt, torch.tensor(255, dtype=mask_gt.dtype))  # 将背景像素设为255，前景像素设为对应的类别id
    segments_info_gt = {"id": 0, "category_id": class_gt.item(), "score": 1.0}
    gt = {"segmentation": mask_gt, "segments_info": segments_info_gt}
    return gt


def prepare_gt_data_semantic(segmentation_output, metadata, scaled_size, image_info):
    """Semantic OVSeg：GT 从 semseg_map_folder 读取（dataloader 在 eval 时不带 mask）。"""
    pred_seg = segmentation_output["segmentation"]
    target_hw = tuple(pred_seg.shape[-2:])
    file_name = image_info["file_name"]
    semseg_sufix = getattr(metadata, "semseg_sufix", ".png")
    file_name_semseg = os.path.splitext(file_name)[0] + semseg_sufix
    gt_path = os.path.join(osp.realpath(metadata.semseg_map_folder), file_name_semseg)
    gt = np.array(Image.open(gt_path), dtype=np.int64)
    gt = torch.from_numpy(gt)
    gt = resize_mask_gt(gt.unsqueeze(0), target_hw, scaled_size).squeeze(0)
    ignore = getattr(metadata, "ignore_label", 255)
    gt = torch.where(gt == ignore, torch.tensor(255, dtype=gt.dtype), gt)
    return {"segmentation": gt}


from xsam.utils.visualize import filter_instances_to_things


def _thing_contiguous_ids(metadata) -> Optional[set]:
    if metadata is None or not hasattr(metadata, "thing_dataset_id_to_contiguous_id"):
        return None
    return set(metadata.thing_dataset_id_to_contiguous_id.values())


def prepare_gt_data_detection(image_info, segmentation_output, metadata=None):
    """Detection GT：从 COCO bbox 标注构造 Instances（支持水平框/旋转框）。"""
    anns = image_info.get("gt_annotations")
    if not anns:
        return None

    thing_ids = _thing_contiguous_ids(metadata)
    if thing_ids is not None:
        anns = [ann for ann in anns if int(ann.get("category_id", -1)) in thing_ids]
    if not anns:
        return None

    target_hw = None
    instances_out = segmentation_output.get("instances")
    if instances_out is not None:
        target_hw = tuple(instances_out.image_size)
    if target_hw is None:
        height = image_info.get("height")
        width = image_info.get("width")
        if height is not None and width is not None:
            target_hw = (int(height), int(width))
    if target_hw is None:
        return None

    box_tensors = []
    classes = []
    rotated = False
    for ann in anns:
        bbox = ann.get("bbox")
        if bbox is None:
            continue
        bbox_mode = ann.get("bbox_mode", BoxMode.XYWH_ABS)
        bbox = list(bbox)
        if len(bbox) == 5:
            rotated = True
            angle = bbox[4]
            if abs(angle) <= math.pi + 1e-3:
                angle = math.degrees(angle)
            box_tensors.append([bbox[0], bbox[1], bbox[2], bbox[3], angle])
        elif len(bbox) == 4:
            xyxy = BoxMode.convert(bbox, bbox_mode, BoxMode.XYXY_ABS)
            box_tensors.append(xyxy)
        else:
            continue
        classes.append(int(ann["category_id"]))

    if not box_tensors:
        return None

    image_size = (int(target_hw[0]), int(target_hw[1]))
    instances = Instances(image_size)
    if rotated:
        instances.pred_boxes = RotatedBoxes(torch.tensor(box_tensors, dtype=torch.float32))
    else:
        instances.pred_boxes = Boxes(torch.tensor(box_tensors, dtype=torch.float32))
    instances.pred_classes = torch.tensor(classes, dtype=torch.long)
    instances.scores = torch.ones(len(classes), dtype=torch.float32)
    return {"instances": instances}


def prepare_gt_data_instance(mask_gt, class_gt, segmentation_output, scaled_size):
    """Instance seg GT：由 dataloader 的 mask_labels / class_labels 构造 Instances，供 draw_ins_seg 使用。"""
    if mask_gt is None or class_gt is None or class_gt.numel() == 0:
        return None

    target_hw = None
    instances_out = segmentation_output.get("instances")
    if instances_out is not None:
        target_hw = tuple(instances_out.image_size)

    if target_hw is None:
        pred_seg = segmentation_output.get("segmentation")
        if isinstance(pred_seg, torch.Tensor):
            target_hw = tuple(int(x) for x in pred_seg.shape[-2:])
        elif isinstance(pred_seg, (list, tuple)) and len(pred_seg) > 0:
            first = pred_seg[0]
            if isinstance(first, torch.Tensor):
                target_hw = tuple(int(x) for x in first.shape[-2:])

    if target_hw is None and isinstance(mask_gt, torch.Tensor):
        if mask_gt.ndim == 3:
            target_hw = tuple(int(x) for x in mask_gt.shape[-2:])
        elif mask_gt.ndim == 2:
            target_hw = tuple(int(x) for x in mask_gt.shape)

    if target_hw is None:
        return None

    masks = resize_mask_gt(mask_gt, target_hw, scaled_size)
    if masks.ndim == 2:
        masks = masks.unsqueeze(0)

    image_size = (int(target_hw[0]), int(target_hw[1]))
    instances = Instances(image_size)
    instances.pred_masks = masks > 0
    instances.pred_classes = class_gt.to(device=masks.device, dtype=torch.long)
    instances.scores = torch.ones(class_gt.shape[0], device=masks.device, dtype=torch.float32)
    instances.pred_boxes = BitMasks(instances.pred_masks).get_bounding_boxes()
    return {"instances": instances}


def prepare_gt_data(mask_gt, class_gt, segmentation_output, metadata, data_name, scaled_size, image_info=None):
    gt = None
    if "detection" in data_name:
        gt = prepare_gt_data_detection(image_info or {}, segmentation_output, metadata)
    elif "instance" in data_name:
        gt = prepare_gt_data_instance(mask_gt, class_gt, segmentation_output, scaled_size)
    elif "genseg" in data_name and "pan" in data_name:
        gt = prepare_gt_data_pan(mask_gt, class_gt, segmentation_output, metadata, scaled_size, image_info=image_info)
    elif "ovseg" in data_name and "pan" in data_name:
        gt = prepare_gt_data_pan(mask_gt, class_gt, segmentation_output, metadata, scaled_size, image_info=image_info)
    elif "ovseg" in data_name and "semantic" in data_name:
        gt = prepare_gt_data_semantic(segmentation_output, metadata, scaled_size, image_info)
    elif "refseg" in data_name or "reaseg" in data_name:
        gt = prepare_gt_data_refseg(mask_gt, class_gt, segmentation_output, scaled_size)

    return gt


def resolve_eval_batch_size(data_name: str, dataset, requested: int) -> Tuple[int, bool]:
    """refseg/reaseg 在模型里会把 batch 内条件拼成多类，后处理要求 num_labels==1，仅支持 batch_size=1。"""
    task_name = getattr(dataset, "task_name", None)
    force_bs1_tasks = {"refseg", "reaseg", "imgconv"}
    if task_name in force_bs1_tasks or any(t in data_name for t in ("refseg", "reaseg", "imgconv")):
        if requested > 1:
            return 1, True
        return 1, False
    return requested, False


def evaluate_dataset(
    model,
    dataset,
    evaluator,
    rank: int,
    world_size: int,
    generation_config: Optional[GenerationConfig] = None,
    stop_criteria: Optional[StoppingCriteriaList] = None,
    visualizer=None,
    batch_size: int = 1,
    num_workers: int = 4,
    save_vis: bool = True,
) -> None:
    """Evaluate model on a single dataset."""
    data_name = evaluator.data_name
    metadata = dataset.metadata
    output_ids_with_output = dataset.output_ids_with_output
    mode = "tensor" if output_ids_with_output else "predict"

    # Visualizer 在 cfg 里通常不带 metadata，默认为空 default，会导致回退去读硬编码 sota JSON
    if visualizer is not None:
        visualizer.metadata = metadata
        for _cache_attr in (
            "_category_id_to_name_cache",
            "_debug_logged_categories",
            "_debug_logged_thing_categories",
        ):
            if hasattr(visualizer, _cache_attr):
                delattr(visualizer, _cache_attr)

    dataset_image_folder = getattr(dataset, "image_folder", None)
    effective_batch_size, forced_bs1 = resolve_eval_batch_size(data_name, dataset, batch_size)
    if rank == 0 and forced_bs1:
        print_log(
            f"{data_name}: task={getattr(dataset, 'task_name', None)} 不支持 batch_size>1，"
            f"per-GPU batch 使用 {effective_batch_size}（请求值为 {batch_size}）",
            logger="current",
        )
    # #  取dataset的前10个样本进行测试
    #dataset = torch.utils.data.Subset(dataset, list(range(100)))
    # Setup dataloader
    sampler = DistributedSampler(dataset=dataset, rank=rank, num_replicas=world_size, shuffle=False)
    dataloader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        num_workers=num_workers,
        sampler=sampler,
        shuffle=False,
        collate_fn=xsam_collate_fn,
        pin_memory=False,
        drop_last=False,
    )

    # Evaluation loop
    failed_cnt = 0
    evaluator.reset()
    print_log(f"Evaluating {data_name}...", logger="current")

    for data in tqdm(dataloader, desc=f"Evaluating {data_name}", disable=rank != 0):
        success, seg_outputs, llm_outputs = process_batch(model, data, data_name, metadata, generation_config, stop_criteria, mode)
        if not success:
            failed_cnt += 1
            continue

        if save_vis and visualizer is not None and "imgconv" not in data_name:
            # Draw predictions
            image_infos = data["data_samples"].metainfo["image_infos"]
            mask_labels = data["data_samples"].mask_labels
            class_labels = data["data_samples"].class_labels
            scaled_sizes = data["data_samples"].scaled_sizes
            sampled_labels_batch = getattr(data["data_samples"], "sampled_labels", None)

            for i, (image_info, segmentation_output, mask_gt, class_gt) in enumerate(
                zip(image_infos, seg_outputs, mask_labels, class_labels)
            ):
                scaled_size = scaled_sizes[i] if scaled_sizes is not None and i < len(scaled_sizes) else scaled_sizes[0]
                file_name = image_info["file_name"]
                image = mmcv.imread(osp.join(dataset_image_folder, file_name))
                image = mmcv.imconvert(image, "bgr", "rgb")

                sample_id = image_info.get("sample_id", "")
                if "phrases" not in image_info:
                    image_info.update({"phrases": []})
                vis_output_dir = osp.join(evaluator.output_dir, "vis")
                os.makedirs(vis_output_dir, exist_ok=True)
                try:
                    seg_out_vis = dict(segmentation_output)
                    if "detection" in data_name and seg_out_vis.get("instances") is not None:
                        seg_out_vis["instances"] = filter_instances_to_things(seg_out_vis["instances"], metadata)
                    if "pan" in data_name and seg_out_vis.get("segments_info") is not None:
                        seg_out_vis["segments_info"] = _panoptic_pred_segments_info_for_vis(
                            seg_out_vis["segments_info"], metadata
                        )
                    # semantic：补上 sampled_labels，供 EvalVisualizer 做 contiguous→dataset_id 查名
                    if "semantic" in data_name and seg_out_vis.get("sampled_labels") is None:
                        if sampled_labels_batch is not None and i < len(sampled_labels_batch):
                            seg_out_vis["sampled_labels"] = sampled_labels_batch[i]
                        else:
                            d2c = getattr(metadata, "dataset_id_to_contiguous_id", None) or {}
                            if d2c:
                                cont_to_ds = {int(v): int(k) for k, v in d2c.items()}
                                seg_out_vis["sampled_labels"] = [cont_to_ds[j] for j in range(len(cont_to_ds))]
                    visualizer.draw_predictions(
                        image,
                        data_name=data_name,
                        output_file=osp.join(vis_output_dir,  f"{osp.splitext(file_name)[0]}{sample_id}.png"),
                        **image_info,
                        **seg_out_vis,
                    )
                except Exception as e:
                    print_log(f"Error visualizing {file_name}: {e}\n{traceback.format_exc()}", logger="current")
                    continue

                per_sample_labels = None
                if sampled_labels_batch is not None and i < len(sampled_labels_batch):
                    per_sample_labels = sampled_labels_batch[i]
                gt_image_info = {**image_info, "sampled_labels": per_sample_labels} if per_sample_labels is not None else image_info

                try:
                    gt = prepare_gt_data(
                        mask_gt, class_gt, segmentation_output, metadata, data_name, scaled_size, image_info=gt_image_info
                    )
                except Exception as e:
                    print_log(
                        f"Error preparing ground truth for {file_name}: {e}\n{traceback.format_exc()}",
                        logger="current",
                    )
                    continue
                if gt is None:
                    continue
                try:
                    # GT semantic 像素已是 dataset category_id，禁止再按 contiguous 反查
                    gt_vis_kwargs = dict(gt)
                    if "semantic" in data_name:
                        gt_vis_kwargs["remap_pred_to_dataset_id"] = False
                    visualizer.draw_predictions(
                        image,
                        data_name=data_name,
                        output_file=osp.join(vis_output_dir,  f"{osp.splitext(file_name)[0]}{sample_id}_gt.png"),
                        **image_info,
                        **gt_vis_kwargs,
                    )
                except Exception as e:
                    print_log(f"Error visualizing ground truth {file_name}: {e}\n{traceback.format_exc()}", logger="current")

        val_inputs = copy.deepcopy(data["data_samples"].metainfo["image_infos"])   
        if 'imgconv' in data_name:
            conversations = data["data_samples"].metainfo["conversations"]
            task_categories = data["data_samples"].metainfo.get("task_categories", None)
            for i in range(len(val_inputs)):
                val_inputs[i]["conversation"] = conversations[i]
                if task_categories is not None:
                    val_inputs[i]["task_category"] = task_categories[i]
            evaluator.process(val_inputs, llm_outputs)
        else:
            evaluator.process(val_inputs, seg_outputs)
    print_log(f"Failed number of {data_name}: {failed_cnt}", logger="current")
    evaluator.evaluate()
    print_log(f"Evaluating {data_name} done!", logger="current")


def main():
    """Main evaluation function."""
    args = parse_args()
    rank, local_rank, world_size = setup_distributed(args)

    # Load and process config
    if not osp.isfile(args.config):
        try:
            args.config = cfgs_name_path[args.config]
        except KeyError:
            raise FileNotFoundError(f"Cannot find {args.config}")

    cfg = Config.fromfile(args.config)
    set_model_resource(cfg)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    register_function(cfg._cfg_dict)
    if args.seed is not None:
        # Use args.seed
        set_random_seed(args.seed)
        print_log(
            f"Set the random seed to {args.seed}.",
            logger="current",
        )

    # Handle latest checkpoint
    if args.pth_model == "latest":
        from mmengine.runner import find_latest_checkpoint

        if osp.exists(osp.join(args.work_dir, "pytorch_model.bin")):
            args.pth_model = osp.join(args.work_dir, "pytorch_model.bin")
        else:
            args.pth_model = find_latest_checkpoint(args.work_dir)
        print_log(f"Found latest checkpoint: {args.pth_model}", logger="current")

    # Build and setup model
    model = BUILDER.build(cfg.model)
    if "llm" in cfg.model:
        model.llm.to(cfg.model.llm.torch_dtype)
    model.eval()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if world_size > 1:
        print_log(f"Model on {device}, distributed eval with {world_size} GPUs", logger="current")

    load_checkpoint(model, args.pth_model)
    stop_criteria, generation_config = setup_model_config(model, cfg)

    # Setup visualizer if available in config
    visualizer = None
    if hasattr(cfg, "visualizer") and cfg.visualizer is not None:
        try:
            visualizer = _build_visualizer_from_cfg(cfg)
            print_log("Visualizer initialized successfully", logger="current")
        except Exception as e:
            print_log(f"Warning: Could not initialize visualizer: {e}", logger="current")
            visualizer = None
    if visualizer is None:
        print_log(
            "No visualizer (missing or failed to build). PNGs under pred_data/<dataset>/vis/ will not be saved.",
            logger="current",
        )

    # Evaluate on all datasets
    assert len(cfg.val_datasets) == len(
        cfg.val_evaluators
    ), f"len(cfg.val_datasets) = {len(cfg.val_datasets)}, len(cfg.val_evaluators) = {len(cfg.val_evaluators)}"
    save_vis = not args.no_vis
    if rank == 0:
        print_log(
            f"Eval settings: batch_size={args.batch_size}, num_workers={args.num_workers}, "
            f"world_size={world_size}, save_vis={save_vis}",
            logger="current",
        )

    print_log(f"Evaluating {len(cfg.val_datasets)} datasets...", logger="current")
    for dataset_cfg, evaluator_cfg in zip(cfg.val_datasets, cfg.val_evaluators):
        try:
            dataset = BUILDER.build(dataset_cfg)
            model.postprocess_fn = dataset.postprocess_fn

            evaluator = BUILDER.build(evaluator_cfg)
            evaluator.metadata = dataset.metadata
            evaluator.output_dir = osp.join(args.work_dir, "pred_data", evaluator.data_name)
            evaluator._distributed = world_size > 1
            evaluate_dataset(
                model,
                dataset,
                evaluator,
                rank,
                world_size,
                generation_config,
                stop_criteria,
                visualizer if save_vis else None,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                save_vis=save_vis,
            )
        except Exception as e:
            print_log(f"Error evaluating {dataset_cfg.data_name}\n: {e}\n{traceback.format_exc()}", logger="current")
            continue


if __name__ == "__main__":
    main()
