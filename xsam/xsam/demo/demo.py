#!/usr/bin/env python


import argparse
import json
import os.path as osp
import random
import re
import traceback
import warnings
from typing import List

import numpy as np
import torch
import torch.nn as nn
from mmengine.config import Config, DictAction
from mmengine.runner.utils import set_random_seed
from mmengine.utils.misc import get_object_from_string
from PIL import Image
from xtuner.configs import cfgs_name_path
from xtuner.dataset.utils import expand2square
from xtuner.model.utils import traverse_dict
from xtuner.registry import BUILDER
from xtuner.tools.utils import set_model_resource
from xtuner.utils.device import get_device

from xsam.dataset.collate_fns import xsam_collate_fn
from xsam.dataset.map_fns import (
    dataset_map_fn_factory,
    gcg_seg_map_fn,
    generic_seg_map_fn,
    image_conv_map_fn,
    inter_seg_map_fn,
    ov_seg_map_fn,
    reason_seg_map_fn,
    refer_seg_map_fn,
    template_map_fn_factory,
    vgd_seg_map_fn,
)
from xsam.dataset.map_fns.template_map_fn_factory import template_map_fn
from xsam.dataset.process_fns import (
    gcg_seg_postprocess_fn,
    generic_seg_postprocess_fn,
    inter_seg_postprocess_fn,
    ov_seg_postprocess_fn,
    process_map_fn_factory,
    reason_seg_postprocess_fn,
    refer_seg_postprocess_fn,
    vgd_seg_postprocess_fn,
)
from xsam.dataset.utils.catalog import MetadataCatalog
from xsam.dataset.utils.encode import encode_fn
from xsam.engine.utils.util import split_list
from xsam.utils.checkpoint import load_checkpoint
from xsam.utils.config import setup_model_config
from xsam.utils.constants import DEFAULT_IMAGE_TOKEN, INDEX2TOKEN
from xsam.utils.logging import print_log, set_default_logging_format
from xsam.utils.palette import get_palette
from xsam.demo.genseg_pano_utils import (
    GENSEG_PANO_METADATA_NAME,
    GENSEG_PANO_VIS_DATA_NAME,
    build_metadata_from_categories,
    convert_segments_info_to_dataset_id,
    load_pano_categories,
    resolve_pano_categories_json,
)
from xsam.utils.misc import data_dict_to_device
from xsam.utils.utils import register_function
from xsam.utils.visualize import Visualizer
from xsam.utils.xtuner_patch import patch_xtuner_llama_attn_for_single_gpu

# Xtuner llama_attn_forward calls dist.get_rank(); demo runs without init_process_group.
patch_xtuner_llama_attn_for_single_gpu()

# Global setup
set_default_logging_format()
warnings.filterwarnings("ignore")

# ovseg 为开集：类别完全由用户在提示中给出，不用固定评测集 catalog 名（避免与 sota_panoptic_ovseg_val 等混淆）
OVSEG_OPEN_METADATA_NAME = "open_panoptic_ovseg_user"


def _panoptic_pred_segments_info_for_vis(segments_info, metadata, sampled_labels=None):
    """Demo 全景预测可视化：与评测侧逻辑对齐。

    open_cls 下 compute_segments 的 category_id 为「当前提示中类别顺序」下标；若提供与 DataLoader
    中 sampled_labels 同义的 dataset_id 列表（与 sampled_cats 顺序一致），则按该表映射到 COCO id；
    否则回退 metadata 的 contiguous→dataset 逆映射（与 eval_ori._panoptic_pred_segments_info_for_vis 一致）。
    """
    if segments_info is None or not isinstance(segments_info, list):
        return segments_info
    thing_ds_keys = set(getattr(metadata, "thing_dataset_id_to_contiguous_id", None) or {})
    ds_to_cont = getattr(metadata, "dataset_id_to_contiguous_id", None) or {}
    cont_to_ds = {int(v): int(k) for k, v in ds_to_cont.items()} if ds_to_cont else {}

    def _cid_to_dataset_id(cid: int) -> int:
        if sampled_labels is not None and len(sampled_labels) > 0:
            if 0 <= cid < len(sampled_labels):
                return int(sampled_labels[cid])
        if cont_to_ds:
            return int(cont_to_ds.get(cid, cid))
        return cid

    out = []
    for s in segments_info:
        t = dict(s)
        cid = t.get("category_id")
        if cid is not None:
            cid = int(cid)
            ds_cat = _cid_to_dataset_id(cid)
            t["category_id"] = ds_cat
            t["isthing"] = ds_cat in thing_ds_keys
        out.append(t)
    return out


def register_panoptic_metadata_from_coco_json(
    coco_data: dict,
    data_name: str,
    *,
    gt_json_path: str,
    panseg_map_folder=None,
    semseg_map_folder=None,
    semseg_sufix: str = ".png",
    ignore_label: int = 255,
) -> None:
    """与 GenericSegDataset._set_panoptic_metadata 一致，将 pano COCO JSON 写入 MetadataCatalog（供 genseg 推理与可视化）。"""
    cats = coco_data["categories"]
    cat_ids = sorted([cat["id"] for cat in cats])
    cat_colors = (
        [x["color"] for x in sorted(cats, key=lambda x: x["id"])]
        if cats and "color" in cats[0]
        else get_palette("random", len(cats))
    )
    dataset_id_to_contiguous_id = {x["id"]: i for i, x in enumerate(sorted(cats, key=lambda x: x["id"]))}
    cat_id_to_name = {x["id"]: x["name"] for x in cats}
    cat_id_to_color = {x["id"]: cat_colors[dataset_id_to_contiguous_id[x["id"]]] for x in cats}

    thing_cats = [x for x in cats if x.get("isthing", None) == 1]
    thing_cat_ids = [x["id"] for x in thing_cats]
    stuff_cats = [x for x in cats if x.get("isthing", None) == 0]
    stuff_cat_ids = [x["id"] for x in stuff_cats]
    thing_cat_id_to_contiguous_id = {
        cat_id: cont_id for cont_id, cat_id in enumerate(cat_ids) if cat_id in thing_cat_ids
    }
    stuff_cat_id_to_contiguous_id = {
        cat_id: cont_id for cont_id, cat_id in enumerate(cat_ids) if cat_id in stuff_cat_ids
    }
    thing_cat_id_to_name = {thing_cat_id_to_contiguous_id[x["id"]]: x["name"] for x in thing_cats}
    stuff_cat_id_to_name = {stuff_cat_id_to_contiguous_id[x["id"]]: x["name"] for x in stuff_cats}
    thing_cat_id_to_color = {
        thing_cat_id_to_contiguous_id[x["id"]]: cat_colors[thing_cat_id_to_contiguous_id[x["id"]]]
        for x in thing_cats
    }
    stuff_cat_id_to_color = {
        stuff_cat_id_to_contiguous_id[x["id"]]: cat_colors[stuff_cat_id_to_contiguous_id[x["id"]]]
        for x in stuff_cats
    }

    metadata = MetadataCatalog.get(data_name)
    metadata.set(
        gt_json=gt_json_path,
        semseg_sufix=semseg_sufix,
        panseg_map_folder=panseg_map_folder,
        semseg_map_folder=semseg_map_folder,
        data_name=data_name,
        dataset_classes=cat_id_to_name,
        dataset_colors=cat_id_to_color,
        thing_classes=thing_cat_id_to_name,
        thing_colors=thing_cat_id_to_color,
        stuff_classes=stuff_cat_id_to_name,
        stuff_colors=stuff_cat_id_to_color,
        dataset_id_to_contiguous_id=dataset_id_to_contiguous_id,
        thing_dataset_id_to_contiguous_id=thing_cat_id_to_contiguous_id,
        stuff_dataset_id_to_contiguous_id=stuff_cat_id_to_contiguous_id,
        ignore_label=ignore_label,
        label_divisor=1000,
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Single image demo for X-SAM model")
    parser.add_argument("config", help="config file name or path")
    parser.add_argument("--image", type=str, required=True, help="path to input image")
    parser.add_argument("--prompt", type=str, required=True, help="user prompt for the task_name")
    parser.add_argument("--task_name", type=str, required=True, help="task_name name (e.g., segmentation, referring)")
    parser.add_argument("--work-dir", help="directory to save logs and visualizations")
    parser.add_argument(
        "--pth_model",
        type=str,
        default=None,
        help="path to model checkpoint",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--output", type=str, default="demo_output.png", help="output image path")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override config options, format: xxx=yyy",
    )
    return parser.parse_args()


def build_from_cfg_or_module(cfg_or_mod):
    if cfg_or_mod is None:
        return None

    if isinstance(cfg_or_mod, nn.Module):
        return cfg_or_mod
    elif callable(cfg_or_mod):
        return cfg_or_mod
    elif isinstance(cfg_or_mod, dict):
        traverse_dict(cfg_or_mod)
        return BUILDER.build(cfg_or_mod)
    else:
        raise NotImplementedError


def get_phrases_ids(input_ids: torch.Tensor, pstart_token_idx: int, pend_token_idx: int) -> List[torch.Tensor]:
    """Extract phrase IDs from input IDs using start and end tokens."""
    pstart_idx = [i for i, x in enumerate(input_ids) if x == pstart_token_idx]
    pend_idx = [i + 1 for i, x in enumerate(input_ids) if x == pend_token_idx]
    phrases_ids = []
    for ps, pe in zip(pstart_idx, pend_idx):
        phrases_ids.append(input_ids[ps + 1 : pe - 1])
    return phrases_ids


def decode_phrases_ids(tokenizer, phrases_ids: List[torch.Tensor]) -> List[str]:
    """Decode phrase IDs to text."""
    phrases = []
    for phrase_id in phrases_ids:
        if (phrase_id < 0).any():
            phrase = ""
        else:
            phrase = tokenizer.decode(phrase_id).strip()
        phrases.append(phrase)
    return phrases


class XSamDemo:
    def __init__(
        self,
        cfg,
        pth_model=None,
        output_ids_with_output=True,
        max_length=4096,
        cond_type="phrase",
        pad_image_to_square=True,
        **kwargs,
    ):
        self.cfg = cfg
        self.device = get_device()
        self.cpu_device = torch.device("cpu")

        self.model = BUILDER.build(cfg.model)
        if "llm" in cfg.model:
            self.model.llm.to(cfg.model.llm.torch_dtype)
        self.model.eval()
        self.model = self.model.to(self.device)
        if pth_model is not None:
            print_log(f"Loading checkpoint from {pth_model}", logger="current")
            assert osp.exists(pth_model), f"Checkpoint file {pth_model} does not exist"
            load_checkpoint(self.model, pth_model)
        self.stop_criteria, self.generation_config = setup_model_config(self.model, cfg)

        self.tokenizer = self.model.tokenizer
        self.visualizer = build_from_cfg_or_module(cfg.visualizer)
        self.image_processor = build_from_cfg_or_module(cfg.image_processor)
        self.extra_image_processor = build_from_cfg_or_module(cfg.extra_image_processor)

        self.cond_type = cond_type
        self.output_ids_with_output = output_ids_with_output
        self.max_length = max_length
        self.pad_image_to_square = pad_image_to_square
        self.metadata = MetadataCatalog.get("default")
        self.metadata.set(ignore_label=255, label_divisor=1000)
        self.dtype = self.model.dtype

        self.task_map_fns = self.build_map_fns()
        self.template_map_fns = self.build_template_map_fns()
        self.postprocess_fns = self.build_postprocess_fn()
        self._current_classes = None  # 用于ovseg任务过滤segments

        # genseg 与 eval_ori 中 panoptic_genseg_pano_val 对齐：全量类别时用 pano JSON 注册 Metadata
        self._pano_genseg_json_path = None
        self._pano_coco_raw = None
        self._pano_full_signature = None
        if hasattr(self.cfg, "pano_data_root") and self.cfg.pano_data_root:
            pano_root = self.cfg.pano_data_root
            for cand in ("annotations_val.json", "annotations_train.json"):
                p = osp.join(pano_root, cand)
                if osp.isfile(p):
                    self._pano_genseg_json_path = p
                    break
            if self._pano_genseg_json_path:
                try:
                    with open(self._pano_genseg_json_path, "r", encoding="utf-8") as f:
                        self._pano_coco_raw = json.load(f)
                    th, st, _ = self._thing_stuff_names_from_coco_categories(self._pano_coco_raw)
                    self._pano_full_signature = (tuple(th + st), tuple(th), tuple(st))
                except Exception as e:
                    print_log(f"Could not load pano genseg categories: {e}", logger="current")
                    self._pano_genseg_json_path = None
                    self._pano_coco_raw = None
                    self._pano_full_signature = None

        try:
            self._genseg_pano_json_resolved = resolve_pano_categories_json(self.cfg)
            print_log(f"genseg pano 类别 JSON: {self._genseg_pano_json_resolved}", logger="current")
        except FileNotFoundError as e:
            self._genseg_pano_json_resolved = None
            print_log(f"genseg pano JSON 未找到，将回退旧 demo 路径: {e}", logger="current")

    def _pano_panseg_folder_for_json(self, json_path: str) -> str:
        root = self.cfg.pano_data_root
        if "annotations_val" in json_path or json_path.endswith("val_annotations.json"):
            return osp.join(root, "val", "panoptic_labels")
        return osp.join(root, "train", "panoptic_labels")

    @staticmethod
    def _thing_stuff_names_from_coco_categories(coco_data: dict):
        cats = coco_data.get("categories", [])
        if not cats:
            return None, None, None
        cats = sorted(cats, key=lambda x: x["id"])
        thing_cats = [c for c in cats if c.get("isthing", 0) == 1]
        stuff_cats = [c for c in cats if c.get("isthing", 0) == 0]
        thing_classes = [c["name"] for c in thing_cats]
        stuff_classes = [c["name"] for c in stuff_cats]
        cat_name_to_isthing = {c["name"]: c.get("isthing", 0) for c in cats}
        return thing_classes, stuff_classes, cat_name_to_isthing

    def default_genseg_prompt(self) -> str:
        """展示用 ins:/sem: 提示；实际 genseg 推理走 predict_genseg_pano 全量 pano 类别（与评测脚本一致）。"""
        if self._genseg_pano_json_resolved and osp.isfile(self._genseg_pano_json_resolved):
            _, thing_classes, stuff_classes, _ = load_pano_categories(self._genseg_pano_json_resolved)
        else:
            thing_classes, stuff_classes, _ = self._load_sota_categories("genseg")
        if not thing_classes and not stuff_classes:
            return ""
        return "ins: " + ", ".join(thing_classes) + ";\nsem: " + ", ".join(stuff_classes)

    def build_template_map_fns(self):
        template_map_fns = {
            "imgconv": dict(
                type=template_map_fn_factory,
                template=self.cfg.prompt_template,
                output_suffix=self.output_ids_with_output,
            ),
            "genseg": dict(
                type=template_map_fn_factory,
                template=self.cfg.prompt_template,
                output_suffix=self.output_ids_with_output,
            ),
            "ovseg": dict(
                type=template_map_fn_factory,
                template=self.cfg.prompt_template,
                output_suffix=self.output_ids_with_output,
            ),
            "refseg": dict(
                type=template_map_fn_factory,
                template=self.cfg.prompt_template,
                output_suffix=self.output_ids_with_output,
            ),
            "reaseg": dict(
                type=template_map_fn_factory,
                template=self.cfg.prompt_template,
                output_suffix=self.output_ids_with_output,
            ),
            "gcgseg": dict(
                type=template_map_fn_factory,
                template=self.cfg.prompt_template,
                output_suffix=False,
            ),
            "interseg": dict(
                type=template_map_fn_factory,
                template=self.cfg.prompt_template,
                output_suffix=self.output_ids_with_output,
            ),
            "vgdseg": dict(
                type=template_map_fn_factory,
                template=self.cfg.prompt_template,
                output_suffix=self.output_ids_with_output,
            ),
        }
        template_map_fns = {
            task_name: build_from_cfg_or_module(template_map_fn)
            for task_name, template_map_fn in template_map_fns.items()
        }
        return template_map_fns

    def build_map_fns(self):
        task_map_fns = {
            "imgconv": image_conv_map_fn,
            "genseg": dict(
                type=dataset_map_fn_factory,
                fn=generic_seg_map_fn,
                cond_type=self.cond_type,
            ),
            "ovseg": dict(
                type=dataset_map_fn_factory,
                fn=ov_seg_map_fn,
                cond_type=self.cond_type,
            ),
            "refseg": dict(
                type=dataset_map_fn_factory,
                fn=refer_seg_map_fn,
                cond_type=self.cond_type,
            ),
            "reaseg": dict(
                type=dataset_map_fn_factory,
                fn=reason_seg_map_fn,
                cond_type=self.cond_type,
            ),
            "gcgseg": dict(
                type=dataset_map_fn_factory,
                fn=gcg_seg_map_fn,
                cond_type=self.cond_type,
            ),
            "interseg": dict(
                type=dataset_map_fn_factory,
                fn=inter_seg_map_fn,
                cond_type=self.cond_type,
            ),
            "vgdseg": dict(
                type=dataset_map_fn_factory,
                fn=vgd_seg_map_fn,
                cond_type=self.cond_type,
            ),
        }
        task_map_fns = {
            task_name: build_from_cfg_or_module(task_map_fn) for task_name, task_map_fn in task_map_fns.items()
        }
        return task_map_fns

    def build_postprocess_fn(self):
        postprocess_fns = {
            "imgconv": None,
            "genseg": dict(
                type=process_map_fn_factory,
                fn=generic_seg_postprocess_fn,
                task_name="panoptic_genseg",
                threshold=0.0,
            ),
            "genseg(pan)": dict(
                type=process_map_fn_factory,
                fn=generic_seg_postprocess_fn,
                task_name="panoptic_genseg",
                threshold=0.5,
            ),
            "genseg(sem)": dict(
                type=process_map_fn_factory,
                fn=generic_seg_postprocess_fn,
                task_name="genseg(sem)",
            ),
            "genseg(ins)": dict(
                type=process_map_fn_factory,
                fn=generic_seg_postprocess_fn,
                task_name="genseg(ins)",
            ),
            "ovseg": dict(
                type=process_map_fn_factory,
                fn=ov_seg_postprocess_fn,
                task_name="panoptic_ovseg",
                threshold=0.0,
            ),
            "refseg": refer_seg_postprocess_fn,
            "reaseg": reason_seg_postprocess_fn,
            "gcgseg": gcg_seg_postprocess_fn,
            "interseg": inter_seg_postprocess_fn,
            "vgdseg": vgd_seg_postprocess_fn,
        }
        postprocess_fns = {
            task_name: build_from_cfg_or_module(postprocess_fn)
            for task_name, postprocess_fn in postprocess_fns.items()
        }
        return postprocess_fns

    @staticmethod
    def _uses_tensor_infer(task_name: str) -> bool:
        """与 eval_ori 一致：分割类 val 使用 output_ids_with_output=True → tensor（genseg 另有 run_genseg_pano_predict）。"""
        return task_name in ("genseg", "ovseg", "refseg", "reaseg") or "genseg" in task_name

    def _input_ids_with_output_for_task(self, task_name: str) -> bool:
        if self._uses_tensor_infer(task_name):
            return True
        return self.output_ids_with_output

    def _apply_template_map_fn(self, data_dict, task_name: str, input_ids_with_output: bool):
        if self.template_map_fns.get(task_name) is None:
            return data_dict
        if input_ids_with_output == self.output_ids_with_output:
            return self.template_map_fns[task_name](data_dict)
        template = self.cfg.prompt_template
        if isinstance(template, str):
            template = get_object_from_string(template)
        return template_map_fn(data_dict, template, output_suffix=input_ids_with_output)

    def _get_input_ids(
        self,
        data_dict,
        task_name,
        with_image_token=True,
        next_needs_bos_token=False,
        input_ids_with_output=None,
    ):
        if self.tokenizer is None:
            return data_dict

        if input_ids_with_output is None:
            input_ids_with_output = self._input_ids_with_output_for_task(task_name)

        if self.task_map_fns.get(task_name) is not None:
            data_dict = self.task_map_fns[task_name](data_dict, input_ids_with_output)
        data_dict = self._apply_template_map_fn(data_dict, task_name, input_ids_with_output)
        if self.tokenizer is not None:
            data_dict = encode_fn(
                data_dict,
                self.tokenizer,
                self.max_length,
                input_ids_with_output,
                with_image_token,
                next_needs_bos_token,
            )
        return data_dict

    def _get_cond_ids(self, data_dict):
        if self.tokenizer is None:
            return data_dict

        input_ids = data_dict["input_ids"]
        cond_ids = [-1] * len(input_ids)
        pstart_idx = [i for i, x in enumerate(input_ids) if x == self.model.pstart_token_idx]
        pend_idx = [i for i, x in enumerate(input_ids) if x == self.model.pend_token_idx]
        cls_idx = [i for i, x in enumerate(input_ids) if x == self.model.cls_token_idx]

        if len(pstart_idx) == 0 and len(pend_idx) == 0 and len(cls_idx) == 0:
            return data_dict

        if self.cond_type in ["phrase", "all"]:
            for i, (ps, pe) in enumerate(zip(pstart_idx, pend_idx)):
                cond_ids[ps : pe + 1] = [i] * (pe - ps + 1)
        if self.cond_type in ["cls", "all"]:
            for i, ci in enumerate(cls_idx):
                cond_ids[ci] = i

        data_dict["cond_ids"] = cond_ids
        return data_dict

    def _get_phrases_ids(self, input_ids):
        pstart_idx = [i for i, x in enumerate(input_ids) if x == self.model.pstart_token_idx]
        pend_idx = [i + 1 for i, x in enumerate(input_ids) if x == self.model.pend_token_idx]
        phrases_ids = []
        for ps, pe in zip(pstart_idx, pend_idx):
            phrases_ids.append(input_ids[ps + 1 : pe - 1])
        return phrases_ids

    def _get_seg_ids(self, data_dict):
        if self.tokenizer is None:
            return data_dict

        input_ids = data_dict["input_ids"]
        seg_ids = [-1] * len(input_ids)

        seg_idx = [i for i, x in enumerate(input_ids) if x == self.model.seg_token_idx]
        for i, idx in enumerate(seg_idx):
            seg_ids[idx] = i

        data_dict["seg_ids"] = seg_ids
        return data_dict

    def _get_vgd_labels(self, data_dict):
        vprompt_masks = data_dict.get("vprompt_masks", None)
        if vprompt_masks is None:
            return data_dict

        class_labels = [i for i in range(len(vprompt_masks))]
        sampled_labels = [i for i in range(len(vprompt_masks))]
        contiguous_labels = [i for i in range(len(vprompt_masks))]

        data_dict["class_labels"] = torch.tensor(class_labels, dtype=torch.int64)
        data_dict["sampled_labels"] = sampled_labels
        data_dict["contiguous_labels"] = contiguous_labels
        return data_dict

    def _load_sota_categories(self, task_name="genseg"):
        """Load SOTA dataset categories from annotations.json"""
        # Try to get data path from config
        data_paths_to_try = []

        # 优先 pano（与 xsam_base_mixed_finetune_all 中 panoptic_genseg_pano_val 一致）
        if hasattr(self.cfg, "pano_data_root") and self.cfg.pano_data_root:
            pano_root = self.cfg.pano_data_root
            data_paths_to_try.append(osp.join(pano_root, "annotations_val.json"))
            data_paths_to_try.append(osp.join(pano_root, "annotations_train.json"))

        # Try to get from config if available
        if hasattr(self.cfg, 'genseg_data_root'):
            genseg_data_root = self.cfg.genseg_data_root
            data_paths_to_try.append(osp.join(genseg_data_root, "sota/train_annotations.json"))
            data_paths_to_try.append(osp.join(genseg_data_root, "sota/val_annotations.json"))
            data_paths_to_try.append(osp.join(genseg_data_root, "sota/train/train_annotations.json"))
            data_paths_to_try.append(osp.join(genseg_data_root, "sota/val/val_annotations.json"))
        
        # ovseg 为开集任务，不从标注 JSON 拉取 thing/stuff；此处仅为 genseg 等保留 gen_seg / 通用路径

        # Try common paths
        common_paths = [
            "./datas/gen_seg_data/sota/train_annotations.json",
            "./datas/gen_seg_data/sota/val_annotations.json",
            "./datas/gen_seg_data/sota/train/train_annotations.json",
            "./datas/gen_seg_data/sota/val/val_annotations.json",
            "./datas/ov_seg_data/sota/train/train_annotations.json",
            "./datas/ov_seg_data/sota/val/val_annotations.json",
            "./datas/sota/train_annotations.json",
            "./datas/sota/val_annotations.json",
        ]
        data_paths_to_try.extend(common_paths)
        
        # Try to load annotations.json
        for data_path in data_paths_to_try:
            if osp.exists(data_path):
                try:
                    with open(data_path, "r", encoding="utf-8") as f:
                        coco_data = json.load(f)
                    thing_classes, stuff_classes, cat_name_to_isthing = self._thing_stuff_names_from_coco_categories(
                        coco_data
                    )
                    if thing_classes is not None and stuff_classes is not None:
                        return thing_classes, stuff_classes, cat_name_to_isthing
                except Exception as e:
                    print_log(f"Failed to load categories from {data_path}: {e}", logger="current")
                    continue
        
        # Fallback: return None to indicate failure
        return None, None, None

    def _get_classes_from_prompt(self, prompt, task_name):
        if task_name == "ovseg":
            # ovseg：开集全景，由用户在提示中显式给出 thing / stuff（或逗号列表全部视为 stuff）
            # 1) 推荐: "thing: a, b; stuff: c, d"
            # 2) 仅逗号列表: 全部当作 stuff（若需 thing，请用格式 1）
            thing_match = re.search(r"thing:\s*([^;]+)", prompt, re.IGNORECASE)
            stuff_match = re.search(r"stuff:\s*([^;]+)", prompt, re.IGNORECASE)

            if thing_match or stuff_match:
                thing_classes = (
                    [x.strip() for x in thing_match.group(1).split(",") if len(x.strip()) > 0] if thing_match else []
                )
                stuff_classes = (
                    [x.strip() for x in stuff_match.group(1).split(",") if len(x.strip()) > 0] if stuff_match else []
                )
                if not thing_classes and not stuff_classes:
                    raise ValueError("ovseg 请在提示中用 thing: / stuff: 指定至少一类（开集，由用户定义）")
            else:
                classes = [x.strip() for x in prompt.split(",") if len(x.strip()) > 0]
                assert len(classes) > 0, "Please provide at least one class for ovseg"
                thing_classes = []
                stuff_classes = classes
            
            # 保持用户输入的类别顺序
            # thing_classes和stuff_classes已经按照用户输入顺序排列
            all_classes = thing_classes + stuff_classes
            assert len(all_classes) > 0, "Please provide at least one class for ovseg"
            
            return (all_classes, thing_classes, stuff_classes), task_name
        elif task_name == "genseg":
            # genseg: 如果没有提供prompt，使用SOTA数据集的所有默认类别
            if not prompt or not prompt.strip():
                # 从SOTA数据集的annotations.json加载类别
                thing_classes, stuff_classes, _ = self._load_sota_categories("genseg")
                if thing_classes is None or stuff_classes is None:
                    raise ValueError("Failed to load SOTA dataset categories. Please ensure SOTA dataset annotations.json is available.")
            else:
                # 如果提供了prompt，尝试解析
                ins_match = re.search(r"ins:\s*([^;\n]+)", prompt)
                sem_match = re.search(r"sem:\s*([^;\n]+)", prompt)

                thing_classes = [x.strip() for x in ins_match.group(1).split(",") if len(x.strip()) > 0] if ins_match else []
                stuff_classes = [x.strip() for x in sem_match.group(1).split(",") if len(x.strip()) > 0] if sem_match else []
                
                # 如果解析失败，使用SOTA默认类别
                if not thing_classes and not stuff_classes:
                    thing_classes, stuff_classes, _ = self._load_sota_categories("genseg")
                    if thing_classes is None or stuff_classes is None:
                        raise ValueError("Failed to load SOTA dataset categories. Please ensure SOTA dataset annotations.json is available.")
            
            # 保持类别顺序，不要打乱，确保文本和mask的对应关系正确
            # thing_classes在前，stuff_classes在后，保持与annotations.json中的顺序一致
            all_classes = thing_classes + stuff_classes
            assert len(all_classes) > 0, "Please provide at least one thing or stuff class"
            # genseg任务使用panoptic_genseg作为后处理任务名
            task_name_postprocess = "genseg"
            return (all_classes, thing_classes, stuff_classes), task_name_postprocess
        elif "genseg" in task_name:
            # 处理genseg(pan), genseg(ins), genseg(sem)等变体
            ins_match = re.search(r"ins:\s*([^;\n]+)", prompt)
            sem_match = re.search(r"sem:\s*([^;\n]+)", prompt)

            thing_classes = [x.strip() for x in ins_match.group(1).split(",") if len(x.strip()) > 0] if ins_match else []
            stuff_classes = [x.strip() for x in sem_match.group(1).split(",") if len(x.strip()) > 0] if sem_match else []
            
            # 保持类别顺序，不要打乱，确保文本和mask的对应关系正确
            all_classes = thing_classes + stuff_classes
            assert len(all_classes) > 0, "Please provide at least one thing or stuff class"
            if len(thing_classes) > 0 and len(stuff_classes) > 0:
                task_name = "genseg(pan)"
            elif len(thing_classes) > 0 and len(stuff_classes) == 0:
                task_name = "genseg(ins)"
            elif len(thing_classes) == 0 and len(stuff_classes) > 0:
                task_name = "genseg(sem)"
            return (all_classes, thing_classes, stuff_classes), task_name
        else:
            return ([], [], []), task_name

    def _process_prompt(self, prompt, task_name, classes=None):
        if task_name == "imgconv":
            example = {
                "conversations": [
                    {"from": "human", "value": DEFAULT_IMAGE_TOKEN + prompt},
                    {"from": "gpt", "value": ""},
                ]
            }
        elif "genseg" in task_name:
            example = {
                "sampled_cats": classes[0],
                "caption": None,
            }
        elif task_name == "ovseg":
            example = {
                "sampled_cats": classes[0],  # ovseg也使用sampled_cats
                "caption": None,
            }
        elif task_name == "refseg":
            example = {
                "sampled_sents": [prompt],
            }
        elif task_name == "reaseg":
            example = {
                "sampled_sents": [prompt],
                "explain": None,
                "is_sentence": True,
            }
        elif task_name == "gcgseg":
            example = {}
        elif task_name == "interseg":
            # TODO: add interseg example
            example = {
                "sampled_labels": [0],
            }
        elif task_name == "vgdseg":
            # TODO: add vgdseg example
            example = {
                "sampled_labels": [0],
            }
        else:
            raise ValueError(f"Unsupported task_name: {task_name}")

        return example

    def _process_image(self, image):
        if isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        image = np.array(pil_image)
        height, width = image.shape[:2]
        _image_info = {
            "height": height,
            "width": width,
            "image_size": (height, width),
        }
        image_info = {
            "image_info": _image_info,
            "image_size": (height, width),
        }
        return image_info

    def _process_data_dict(self, data_dict):
        data_dict["image_file"] = None
        pil_image = data_dict["pil_image"]
        if self.image_processor is not None:
            image = pil_image
            # 如果 image_processor 是 SiglipProcessor，需要使用内部的 image_processor 属性
            actual_image_processor = self.image_processor
            if hasattr(self.image_processor, 'image_processor'):
                actual_image_processor = self.image_processor.image_processor
            
            if self.pad_image_to_square:
                # 获取 image_mean，优先使用 actual_image_processor 的，否则使用外层的
                if hasattr(actual_image_processor, 'image_mean'):
                    image_mean = actual_image_processor.image_mean
                elif hasattr(self.image_processor, 'image_mean'):
                    image_mean = self.image_processor.image_mean
                else:
                    # 默认值（SigLIP 的标准均值）
                    image_mean = [0.5, 0.5, 0.5]
                image = expand2square(pil_image, tuple(int(x * 255) for x in image_mean))
            
            # 使用实际的图像处理器处理图像
            if hasattr(actual_image_processor, 'preprocess'):
                image = actual_image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            else:
                image = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            data_dict["pixel_values"] = image
        if self.extra_image_processor is not None:
            seg_output = self.extra_image_processor.preprocess(
                pil_image, condition_maps=data_dict["vprompt_masks"], return_tensors="pt"
            )
            data_dict["seg_pixel_values"] = seg_output["pixel_values"][0]
            data_dict["scaled_size"] = tuple(seg_output["scaled_sizes"][0].tolist())
            data_dict["vprompt_masks"] = seg_output.get("vprompt_masks", None)

        data_dict.update(self._get_vgd_labels(data_dict))
        task_name = data_dict["task_name"]
        data_dict.update(
            self._get_input_ids(
                data_dict,
                task_name,
                with_image_token=True,
                input_ids_with_output=self._input_ids_with_output_for_task(task_name),
            )
        )
        data_dict.update(self._get_cond_ids(data_dict))
        data_dict.update(self._get_seg_ids(data_dict))

        return data_dict

    def _process_input_dict(self, data_dict):
        input_dict = xsam_collate_fn([data_dict])
        input_dict = data_dict_to_device(input_dict, device=self.device, dtype=self.dtype)
        data_dict = input_dict["data_dict"]
        data_samples = input_dict["data_samples"]
        data_dict.pop("labels", None)
        data_dict.pop("position_ids", None)
        data_dict.pop("attention_mask", None)

        return data_dict, data_samples

    def _decode_phrases_ids(self, phrases_ids):
        phrases = []
        for phrase_id in phrases_ids:
            if (phrase_id < 0).any():
                phrase = ""
            else:
                phrase = self.tokenizer.decode(phrase_id).strip()
            phrases.append(phrase)
        return phrases

    def _decode_input_ids(self, input_ids):
        input_ids = split_list(input_ids, INDEX2TOKEN.keys())
        text = ""
        for ids in input_ids:
            if len(ids) == 1 and ids[0] in INDEX2TOKEN:
                text += INDEX2TOKEN[ids[0]]
            else:
                text += self.tokenizer.decode(ids)
        ignore_tokens = ["<image>\n", "<p> ", "</p> ", "<|user|>", "<|assistant|>", "<|end|>"]
        for ignore_token in ignore_tokens:
            text = text.replace(ignore_token, "")
        return text

    def _genseg_sampled_labels_for_vis(self, metadata, all_class_names):
        """按提示词里类别顺序构造与评测 sampled_labels 等价的 dataset_id 列表，供 pred 可视化映射。"""
        if not all_class_names:
            return None
        d2c = getattr(metadata, "dataset_id_to_contiguous_id", None) or {}
        if isinstance(d2c, dict) and d2c and all(int(k) == int(v) for k, v in d2c.items()) and len(d2c) == len(all_class_names):
            return list(range(len(all_class_names)))

        ds_classes = getattr(metadata, "dataset_classes", None) or {}
        if not isinstance(ds_classes, dict) or len(ds_classes) == 0:
            return None

        def _norm(x):
            return " ".join(str(x).strip().lower().replace("_", " ").split())

        norm_to_ids = {}
        for did, name in ds_classes.items():
            norm_to_ids.setdefault(_norm(name), []).append(int(did))

        out = []
        for raw in all_class_names:
            ids = norm_to_ids.get(_norm(raw))
            if not ids:
                return None
            out.append(ids[0])
        return out

    def _set_metadata(self, task_name, classes=None):
        MetadataCatalog.reset()
        # 全量 pano genseg：与 eval_ori 中 panoptic_genseg_pano_val + Visualizer 一致
        if task_name == "genseg" and classes is not None and self._pano_full_signature and self._pano_coco_raw:
            sig = (tuple(classes[0]), tuple(classes[1]), tuple(classes[2]))
            if sig == self._pano_full_signature:
                panseg_dir = self._pano_panseg_folder_for_json(self._pano_genseg_json_path)
                register_panoptic_metadata_from_coco_json(
                    self._pano_coco_raw,
                    "panoptic_genseg_pano_val",
                    gt_json_path=self._pano_genseg_json_path,
                    panseg_map_folder=panseg_dir if osp.isdir(panseg_dir) else None,
                )
                return MetadataCatalog.get("panoptic_genseg_pano_val")

        # 使用与eval.py一致的数据名称（子集 genseg 或其它任务）
        if task_name == "genseg":
            data_name = "sota_panoptic_genseg_val"
        elif task_name == "ovseg":
            data_name = OVSEG_OPEN_METADATA_NAME
        else:
            data_name = task_name

        metadata = MetadataCatalog.get(data_name)
        metadata.set(
            label_divisor=1000,
            ignore_label=255,
            data_name=data_name,
        )

        if task_name == "ovseg" and classes is not None:
            # ovseg：开集，metadata 仅反映当前用户输入的 thing/stuff，不绑定任何固定 val 集名称
            all_classes, thing_classes, stuff_classes = classes
            # 调试信息：打印类别顺序
            print_log(f"OVSeg all classes order: {all_classes}", logger="current")
            print_log(f"OVSeg thing_classes: {thing_classes}", logger="current")
            print_log(f"OVSeg stuff_classes: {stuff_classes}", logger="current")
            
            # 构建索引映射：all_classes中的索引到thing/stuff的映射
            thing_indices = [all_classes.index(c) for c in thing_classes if c in all_classes]
            stuff_indices = [all_classes.index(c) for c in stuff_classes if c in all_classes]
            
            # 设置dataset_classes：将contiguous_id映射到类别名称（可视化器优先使用这个）
            # 对于ovseg，contiguous_id就是category_id（0, 1, 2, ...）
            dataset_classes = {i: c for i, c in enumerate(all_classes)}
            
            # 调试信息：验证dataset_classes设置
            print_log(f"OVSeg dataset_classes: {dataset_classes}", logger="current")
            
            metadata.set(
                dataset_id_to_contiguous_id={i: i for i, _ in enumerate(all_classes)},
                thing_dataset_id_to_contiguous_id={i: i for i in thing_indices},
                stuff_dataset_id_to_contiguous_id={i: i for i in stuff_indices},
                thing_classes={i: c for i, c in enumerate(all_classes) if i in thing_indices},
                stuff_classes={i: c for i, c in enumerate(all_classes) if i in stuff_indices},
                dataset_classes=dataset_classes,  # 关键：设置dataset_classes，让可视化器优先使用这个
            )
            
            # 验证metadata设置
            print_log(f"OVSeg metadata.dataset_classes: {metadata.dataset_classes if hasattr(metadata, 'dataset_classes') else 'NOT SET'}", logger="current")
        elif task_name == "genseg" and classes is not None:
            all_classes, thing_classes, stuff_classes = classes
            dataset_classes = {i: c for i, c in enumerate(all_classes)}
            metadata.set(
                dataset_id_to_contiguous_id={i: i for i, _ in enumerate(all_classes)},
                thing_dataset_id_to_contiguous_id={i: i for i, c in enumerate(all_classes) if c in thing_classes},
                stuff_dataset_id_to_contiguous_id={i: i for i, c in enumerate(all_classes) if c in stuff_classes},
                thing_classes={i: c for i, c in enumerate(all_classes) if c in thing_classes},
                stuff_classes={i: c for i, c in enumerate(all_classes) if c in stuff_classes},
                dataset_classes=dataset_classes,
            )
        elif "genseg" in task_name and classes is not None:
            all_classes, thing_classes, stuff_classes = classes
            dataset_classes = {i: c for i, c in enumerate(all_classes)}
            metadata.set(
                dataset_id_to_contiguous_id={i: i for i, _ in enumerate(all_classes)},
                thing_dataset_id_to_contiguous_id={i: i for i, c in enumerate(all_classes) if c in thing_classes},
                stuff_dataset_id_to_contiguous_id={i: i for i, c in enumerate(all_classes) if c in stuff_classes},
                thing_classes={i: c for i, c in enumerate(all_classes) if c in thing_classes},
                stuff_classes={i: c for i, c in enumerate(all_classes) if c in stuff_classes},
                dataset_classes=dataset_classes,
            )

        return metadata

    def run_genseg_pano_predict(self, pil_image, threshold=0.0, pano_json=None):
        """与 xsam/tools/predict_genseg_pano.py 单图通路完全一致（tensor + pano 全类别）。"""
        pano_json = pano_json or self._genseg_pano_json_resolved
        if not pano_json or not osp.isfile(pano_json):
            pano_json = resolve_pano_categories_json(self.cfg, pano_json)

        all_classes, thing_classes, stuff_classes, anno = load_pano_categories(pano_json)
        metadata = build_metadata_from_categories(GENSEG_PANO_METADATA_NAME, anno["categories"])
        classes = (all_classes, thing_classes, stuff_classes)

        post_fn = self.postprocess_fns["genseg"]
        if hasattr(post_fn, "keywords") and isinstance(post_fn.keywords, dict):
            post_fn.keywords["threshold"] = threshold
        self.model.postprocess_fn = post_fn

        prev_output_flag = self.output_ids_with_output
        self.output_ids_with_output = True
        try:
            data_dict = {"pil_image": pil_image, "vprompt_masks": None, "task_name": "genseg"}
            data_dict.update(self._process_prompt("", "genseg", classes))
            data_dict.update(self._process_image(pil_image))
            data_dict.update(self._process_data_dict(data_dict))
            seg_conversation = data_dict.get("conversation")

            input_dict = xsam_collate_fn([data_dict])
            input_dict = data_dict_to_device(input_dict, device=self.device, dtype=self.dtype)
            model_inputs = input_dict["data_dict"]
            data_samples = input_dict["data_samples"]
            model_inputs.pop("labels", None)
            model_inputs.pop("position_ids", None)
            model_inputs.pop("attention_mask", None)
        finally:
            self.output_ids_with_output = prev_output_flag

        llm_input = ""
        generation_output = ""
        if seg_conversation:
            turn = seg_conversation[0] if isinstance(seg_conversation, list) else seg_conversation
            llm_input = (turn.get("input") or "").replace(DEFAULT_IMAGE_TOKEN, "<image>").strip()
            generation_output = (turn.get("output") or "").strip()

        self.model.eval()
        with torch.no_grad():
            _, seg_outputs = self.model(
                model_inputs,
                data_samples,
                mode="tensor",
                metadata=metadata,
                generation_config=self.generation_config,
                stopping_criteria=self.stop_criteria,
                do_postprocess=True,
                do_loss=False,
            )

        if seg_outputs is None or len(seg_outputs) == 0:
            print_log("genseg pano: empty seg_outputs from model", logger="current")
            return llm_input, generation_output, None

        seg_out = seg_outputs[0]
        segments_info_ds = convert_segments_info_to_dataset_id(seg_out.get("segments_info"), metadata)
        print_log(
            f"genseg pano: {len(segments_info_ds)} segments after postprocess (threshold={threshold})",
            logger="current",
        )
        if not segments_info_ds:
            return llm_input, generation_output, None

        image = np.array(pil_image.convert("RGB"))
        vis_seg_out = dict(seg_out)
        vis_seg_out["segments_info"] = segments_info_ds
        visualizer = Visualizer(metadata=metadata)
        visualized_image = visualizer.draw_predictions(
            image,
            data_name=GENSEG_PANO_VIS_DATA_NAME,
            **vis_seg_out,
        )
        result_image = visualized_image.get_image()
        if result_image is None:
            return llm_input, generation_output, None
        return llm_input, generation_output, result_image

    def run_on_image(self, image, prompt, task_name, vprompt_masks=None, **kwargs):
        # 在每次推理前重置状态，确保从干净状态开始
        self._current_classes = None

        if task_name == "genseg":
            pano_json = kwargs.pop("pano_categories_json", None)
            threshold = kwargs.pop("threshold", 0.0)
            return self.run_genseg_pano_predict(image, threshold=threshold, pano_json=pano_json)

        uses_tensor = self._uses_tensor_infer(task_name)
        mode = "tensor" if uses_tensor or self.output_ids_with_output else "predict"
        data_dict = {"pil_image": image, "vprompt_masks": vprompt_masks, "task_name": task_name}

        classes, task_name_postprocess = self._get_classes_from_prompt(prompt, task_name)
        # 确保后处理函数存在
        if task_name_postprocess not in self.postprocess_fns:
            raise ValueError(f"Postprocess function not found for task: {task_name_postprocess}")
        self.model.postprocess_fn = self.postprocess_fns[task_name_postprocess]
        # 设置metadata（所有任务都需要）
        metadata = self._set_metadata(task_name, classes if task_name in ["ovseg", "genseg"] or "genseg" in task_name else None)
        # 保存classes用于后续过滤（ovseg任务需要）
        self._current_classes = classes if task_name == "ovseg" else None
        data_dict.update(self._process_prompt(prompt, task_name, classes))
        data_dict.update(self._process_image(image))
        data_dict.update(self._process_data_dict(data_dict))
        seg_conversation = data_dict.get("conversation")
        data_dict, data_samples = self._process_input_dict(data_dict)
        input_ids = data_dict["input_ids"]

        if self._uses_tensor_infer(task_name):
            # 与 eval / predict_genseg_pano 一致；demo 默认 score_thr=0.5 会滤掉全部 mask 导致只见原图
            kwargs["threshold"] = 0.0

        # 与评测 DataLoader 中 sampled_labels 对齐：open_cls 下 pred 的 class 下标对应该顺序的 COCO id
        vis_sampled_labels = None
        if data_samples is not None and getattr(data_samples, "sampled_labels", None) is not None:
            sl = data_samples.sampled_labels
            if isinstance(sl, (list, tuple)) and len(sl) > 0 and sl[0] is not None:
                vis_sampled_labels = list(sl[0])
        if vis_sampled_labels is None and "genseg" in task_name and classes is not None and len(classes) >= 1 and classes[0]:
            vis_sampled_labels = self._genseg_sampled_labels_for_vis(metadata, classes[0])

        # 确保模型处于eval模式（防止某些层在训练模式下行为不同）
        self.model.eval()
        
        # 清理data_dict中的中间变量，释放内存
        if "inputs_embeds" in data_dict:
            data_dict["inputs_embeds"] = data_dict["inputs_embeds"].contiguous()
        
        with torch.no_grad():
            try:
                llm_outputs, seg_outputs = self.model(
                    data_dict,
                    data_samples,
                    mode=mode,
                    metadata=metadata,
                    generation_config=self.generation_config,
                    stopping_criteria=self.stop_criteria,
                    do_postprocess=True,
                    do_loss=False,
                    **kwargs,
                )
                # 立即清理中间变量
                del data_dict
                if data_samples is not None:
                    del data_samples
            except torch.cuda.OutOfMemoryError as e:
                print_log(f"CUDA OOM in {task_name} prediction: {e}", logger="current")
                # OOM时强制清理
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                return None, None, None
            except Exception as e:
                print_log(f"Error in {task_name} prediction: {e}\n{traceback.format_exc()}", logger="current")
                torch.cuda.empty_cache()  # 出错时也清理缓存
                torch.cuda.synchronize()
                return None, None, None

        input_id_list = input_ids[0].tolist()
        if uses_tensor and seg_conversation:
            turn = seg_conversation[0] if isinstance(seg_conversation, list) else seg_conversation
            llm_input = (turn.get("input") or "").replace(DEFAULT_IMAGE_TOKEN, "<image>").strip()
            generation_output = (turn.get("output") or "").strip()
            output_ids = input_ids
        elif hasattr(llm_outputs, "sequences"):
            llm_input = self._decode_input_ids(input_id_list)
            output_ids = llm_outputs.sequences
            input_length = input_ids[0].shape[0]
            output_length = output_ids[0].shape[0]

            generated_length = output_length - input_length
            if generated_length < 10:
                print_log(
                    f"Warning: {task_name} generated only {generated_length} tokens "
                    f"(input: {input_length}, output: {output_length}). "
                    f"This may indicate early stopping. Full output: {self.tokenizer.decode(output_ids[0], skip_special_tokens=False)[:200]}",
                    logger="current",
                )

            if output_length > input_length:
                generated_ids = output_ids[0][input_length:]
                generation_output = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            else:
                full_output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
                if llm_input and full_output.startswith(llm_input):
                    generation_output = full_output[len(llm_input) :].strip()
                else:
                    llm_input_stripped = llm_input.strip()
                    full_output_stripped = full_output.strip()
                    if llm_input_stripped and full_output_stripped.startswith(llm_input_stripped):
                        generation_output = full_output_stripped[len(llm_input_stripped) :].strip()
                    else:
                        generation_output = full_output_stripped

                if not generation_output:
                    print_log(
                        f"Warning: No new content generated. Input length: {input_length}, "
                        f"Output length: {output_length}, Input: {repr(llm_input)}, "
                        f"Full output: {repr(full_output)}",
                        logger="current",
                    )
        else:
            llm_input = self._decode_input_ids(input_id_list)
            output_ids = input_ids
            generation_output = ""
        
        # 清理特殊标记
        generation_output = generation_output.replace("<|end|>", "").replace("<p> ", "<p>").replace("</p> ", "</p>")
        if task_name != "gcgseg":
            generation_output = generation_output.replace("<p>", "").replace("</p>", "")

        input_phrases = []
        output_phrases = []
        if hasattr(self.model, "pstart_token_idx") and hasattr(self.model, "pend_token_idx"):
            input_phrases_ids = self._get_phrases_ids(input_ids[0])
            input_phrases = self._decode_phrases_ids(input_phrases_ids)

        if hasattr(self.model, "pstart_token_idx") and hasattr(self.model, "pend_token_idx"):
            output_phrases_ids = self._get_phrases_ids(output_ids[0])
            output_phrases = self._decode_phrases_ids(output_phrases_ids)
        phrases = output_phrases or input_phrases

        print_log(f"Sample output of {task_name}:\n" f"{llm_input + generation_output}\n", logger="current")
        
        # 对于对话任务（imgconv），可能没有分割输出，跳过可视化（这是正常的）
        if seg_outputs is None or len(seg_outputs) == 0:
            if task_name != "imgconv":
                print_log(
                    f"Warning: {task_name} returned empty seg_outputs (None or len=0). "
                    f"genseg/ovseg 请确认分数阈值为 0（与评测一致）。",
                    logger="current",
                )
            # 清理LLM输出以释放内存
            if 'llm_outputs' in locals():
                del llm_outputs
            torch.cuda.empty_cache()
            return llm_input, generation_output, None
        
        # 尝试可视化（仅当有分割输出时）
        if self.visualizer is not None:
            self.visualizer.metadata = metadata
            # 对于ovseg任务，清除可视化器的JSON缓存，强制使用metadata中的类别
            if task_name == "ovseg":
                if hasattr(self.visualizer, '_category_id_to_name_cache'):
                    self.visualizer._category_id_to_name_cache = {}
            try:
                # 使用与eval.py一致的data_name
                if task_name == "genseg":
                    vis_data_name = getattr(metadata, "data_name", "panoptic_genseg_pano_val")
                elif task_name == "ovseg":
                    vis_data_name = OVSEG_OPEN_METADATA_NAME
                else:
                    vis_data_name = task_name_postprocess
                
                # 对于ovseg任务，过滤掉不在用户输入类别列表中的segments
                seg_output_for_vis = seg_outputs[0].copy() if isinstance(seg_outputs[0], dict) else seg_outputs[0]
                if task_name == "ovseg" and self._current_classes is not None and "segments_info" in seg_output_for_vis:
                    all_classes, thing_classes, stuff_classes = self._current_classes
                    valid_category_ids = set(range(len(all_classes)))  # 只允许用户输入的类别索引
                    
                    # 过滤segments_info，只保留有效的category_id
                    original_segments_info = seg_output_for_vis["segments_info"]
                    filtered_segments_info = []
                    filtered_segmentation = seg_output_for_vis["segmentation"].clone() if torch.is_tensor(seg_output_for_vis["segmentation"]) else seg_output_for_vis["segmentation"].copy()
                    
                    # 创建category_id到新segment_id的映射
                    valid_segment_id = 1
                    old_segment_id_to_new = {}
                    
                    for seg_info in original_segments_info:
                        category_id = seg_info.get("category_id", -1)
                        old_segment_id = seg_info.get("id", 0)
                        
                        if category_id in valid_category_ids:
                            # 这是一个有效的类别
                            if old_segment_id not in old_segment_id_to_new:
                                old_segment_id_to_new[old_segment_id] = valid_segment_id
                                valid_segment_id += 1
                            
                            new_seg_info = seg_info.copy()
                            new_seg_info["id"] = old_segment_id_to_new[old_segment_id]
                            filtered_segments_info.append(new_seg_info)
                        else:
                            # 无效的类别，从segmentation中移除（设为0）
                            if torch.is_tensor(filtered_segmentation):
                                filtered_segmentation[filtered_segmentation == old_segment_id] = 0
                            else:
                                import numpy as np
                                filtered_segmentation = np.array(filtered_segmentation)
                                filtered_segmentation[filtered_segmentation == old_segment_id] = 0
                    
                    # 更新seg_output_for_vis
                    seg_output_for_vis["segments_info"] = filtered_segments_info
                    seg_output_for_vis["segmentation"] = filtered_segmentation
                    
                    # 调试信息
                    print_log(f"OVSeg: Filtered {len(original_segments_info)} -> {len(filtered_segments_info)} segments", logger="current")
                    print_log(f"OVSeg valid category_ids: {sorted(valid_category_ids)}", logger="current")
                    print_log(f"OVSeg filtered category_ids: {[s.get('category_id') for s in filtered_segments_info]}", logger="current")
                    print_log(f"OVSeg filtered category_names: {[all_classes[s.get('category_id')] if s.get('category_id') < len(all_classes) else 'UNKNOWN' for s in filtered_segments_info]}", logger="current")
                
                if "pan" in vis_data_name and seg_output_for_vis.get("segments_info") is not None:
                    seg_output_for_vis["segments_info"] = _panoptic_pred_segments_info_for_vis(
                        seg_output_for_vis["segments_info"], metadata, sampled_labels=vis_sampled_labels
                    )
                    n_seg = len(seg_output_for_vis["segments_info"])
                    print_log(f"{task_name}: {n_seg} segments for visualization (threshold={kwargs.get('threshold', 'n/a')})", logger="current")

                visualized_image = self.visualizer.draw_predictions(
                    image,
                    data_name=vis_data_name,
                    phrases=phrases,
                    **seg_output_for_vis,
                )
                # visualized_image.save("visualized_image.png")
                result_image = visualized_image.get_image()
                # 清理中间变量释放内存
                del visualized_image
                if 'seg_output_for_vis' in locals():
                    del seg_output_for_vis
                if result_image is None:
                    print_log(f"Warning: {task_name} visualization returned None image", logger="current")
                    torch.cuda.empty_cache()
                    return llm_input, generation_output, None
                # 清理LLM输出和分割输出以释放内存
                if 'llm_outputs' in locals():
                    del llm_outputs
                if 'seg_outputs' in locals():
                    del seg_outputs
                torch.cuda.empty_cache()
                return llm_input, generation_output, result_image
            except Exception as e:
                error_msg = f"Error in {task_name} visualization: {e}\n{traceback.format_exc()}"
                print_log(error_msg, logger="current")
                # 可视化失败时也清理GPU缓存
                torch.cuda.empty_cache()
                return llm_input, generation_output, None
        else:
            return llm_input, generation_output, None


def main():
    """Main demo function for single image processing."""
    args = parse_args()

    # Validate input image exists
    if not osp.exists(args.image):
        raise FileNotFoundError(f"Input image not found: {args.image}")

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
    if args.seed is not None:
        set_random_seed(args.seed)
        print_log(f"Set the random seed to {args.seed}.", logger="current")
    register_function(cfg._cfg_dict)

    # Handle latest checkpoint
    if args.pth_model == "latest":
        from mmengine.runner import find_latest_checkpoint

        if args.work_dir and osp.exists(osp.join(args.work_dir, "pytorch_model.bin")):
            args.pth_model = osp.join(args.work_dir, "pytorch_model.bin")
        elif args.work_dir:
            args.pth_model = find_latest_checkpoint(args.work_dir)
        else:
            raise ValueError("work_dir must be specified when using 'latest' checkpoint")
        print_log(f"Found latest checkpoint: {args.pth_model}", logger="current")

    # Create demo instance
    demo = XSamDemo(cfg, args.pth_model, output_ids_with_output=False)

    pil_image = Image.open(args.image)
    demo.run_on_image(pil_image, args.prompt, args.task_name, None)


if __name__ == "__main__":
    main()
