#!/usr/bin/env python

import argparse
import json
import os
import os.path as osp
import sys
from typing import Dict, List, Tuple

import mmcv
import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.runner.utils import set_random_seed
from PIL import Image
from xtuner.configs import cfgs_name_path
from xtuner.registry import BUILDER
from xtuner.tools.utils import set_model_resource
from xtuner.utils.device import get_device

from xsam.dataset.collate_fns import xsam_collate_fn
from xsam.dataset.utils.catalog import MetadataCatalog
from xsam.demo.demo import XSamDemo
from xsam.utils.checkpoint import load_checkpoint
from xsam.utils.logging import print_log, set_default_logging_format
from xsam.utils.misc import data_dict_to_device
from xsam.utils.utils import register_function
from xsam.utils.visualize import Visualizer


current_dir = osp.dirname(osp.abspath(__file__))
project_root = osp.dirname(osp.dirname(osp.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
xsam_dir = osp.join(project_root, "xsam")
if xsam_dir not in sys.path:
    sys.path.insert(0, xsam_dir)

set_default_logging_format()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch predict panoptic genseg for an image folder")
    parser.add_argument("config", help="config file name or path")
    parser.add_argument("--pth_model", type=str, required=True, help="path to model checkpoint")
    parser.add_argument("--image-dir", type=str, required=True, help="input image directory")
    parser.add_argument("--output-dir", type=str, required=True, help="output root directory")
    parser.add_argument(
        "--pano-categories-json",
        type=str,
        default="./assets/pano/annotations_val.json",
        help="pano categories json copied into this project",
    )
    parser.add_argument("--threshold", type=float, default=0.0, help="postprocess threshold")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--max-images", type=int, default=-1, help="max images to run, -1 for all")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override config options, format: xxx=yyy",
    )
    return parser.parse_args()


def load_pano_categories(pano_json_path: str) -> Tuple[List[str], List[str], List[str], Dict]:
    with open(pano_json_path, "r", encoding="utf-8") as f:
        anno = json.load(f)
    cats = anno.get("categories", [])
    if not cats:
        raise ValueError(f"categories is empty in {pano_json_path}")
    # Align with GenericSegDataset eval path:
    # sampled_labels/sample_cats are generated from category ids sorted by id.
    cats_by_id = sorted(cats, key=lambda x: int(x["id"]))
    all_classes = [c["name"] for c in cats_by_id]
    thing = [c["name"] for c in cats_by_id if int(c.get("isthing", 0)) == 1]
    stuff = [c["name"] for c in cats_by_id if int(c.get("isthing", 0)) == 0]
    return all_classes, thing, stuff, {"categories": cats}


def build_metadata_from_categories(data_name: str, cats: List[Dict]):
    cat_ids = [int(c["id"]) for c in cats]
    dataset_id_to_contiguous_id = {int(c["id"]): i for i, c in enumerate(cats)}
    thing_dataset_id_to_contiguous_id = {
        int(c["id"]): dataset_id_to_contiguous_id[int(c["id"])] for c in cats if int(c.get("isthing", 0)) == 1
    }
    stuff_dataset_id_to_contiguous_id = {
        int(c["id"]): dataset_id_to_contiguous_id[int(c["id"])] for c in cats if int(c.get("isthing", 0)) == 0
    }
    thing_classes = {
        dataset_id_to_contiguous_id[int(c["id"])]: c["name"] for c in cats if int(c.get("isthing", 0)) == 1
    }
    stuff_classes = {
        dataset_id_to_contiguous_id[int(c["id"])]: c["name"] for c in cats if int(c.get("isthing", 0)) == 0
    }
    dataset_classes = {int(c["id"]): c["name"] for c in cats}

    metadata = MetadataCatalog.get(data_name)
    metadata.set(
        data_name=data_name,
        label_divisor=1000,
        ignore_label=255,
        dataset_id_to_contiguous_id=dataset_id_to_contiguous_id,
        thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id,
        stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id,
        thing_classes=thing_classes,
        stuff_classes=stuff_classes,
        dataset_classes=dataset_classes,
    )
    return metadata


def is_image_file(name: str) -> bool:
    ext = osp.splitext(name.lower())[1]
    return ext in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def to_numpy_seg(seg):
    if isinstance(seg, torch.Tensor):
        return seg.detach().cpu().numpy().astype(np.int32)
    return np.asarray(seg, dtype=np.int32)


def convert_segments_info_to_dataset_id(segments_info, metadata):
    if segments_info is None:
        return []
    cont_to_ds = {int(v): int(k) for k, v in metadata.dataset_id_to_contiguous_id.items()}
    thing_ds_ids = set(metadata.thing_dataset_id_to_contiguous_id.keys())
    out = []
    for s in segments_info:
        t = dict(s)
        cid = int(t.get("category_id", -1))
        ds_cid = cont_to_ds.get(cid, cid)
        t["category_id"] = ds_cid
        t["isthing"] = ds_cid in thing_ds_ids
        out.append(t)
    return out


def segmentation_ids_to_category_map(
    seg_map: np.ndarray,
    segments_info: List[Dict],
    background_category_id: int = 0,
) -> np.ndarray:
    """将 segment id 图转为与 annotations 中 categories[].id 一致的逐像素类别 id。"""
    out = np.full(seg_map.shape, background_category_id, dtype=np.int32)
    for s in segments_info:
        sid = int(s["id"])
        cid = int(s["category_id"])
        out[seg_map == sid] = cid
    return out


def save_category_id_png(path: str, category_map: np.ndarray) -> None:
    """类别 id 单通道 PNG：id∈[0,255] 用 L，否则用 I（32-bit）。"""
    cm = category_map.astype(np.int32)
    if cm.min() >= 0 and cm.max() <= 255:
        Image.fromarray(cm.astype(np.uint8), mode="L").save(path)
    else:
        Image.fromarray(cm, mode="I").save(path)


def main():
    args = parse_args()
    if args.seed is not None:
        set_random_seed(args.seed)

    if not osp.isfile(args.config):
        if args.config in cfgs_name_path:
            args.config = cfgs_name_path[args.config]
        else:
            raise FileNotFoundError(f"Cannot find config: {args.config}")

    if not osp.exists(args.pth_model):
        raise FileNotFoundError(f"checkpoint not found: {args.pth_model}")
    if not osp.isdir(args.image_dir):
        raise NotADirectoryError(f"image dir not found: {args.image_dir}")
    if not osp.isfile(args.pano_categories_json):
        raise FileNotFoundError(f"pano categories json not found: {args.pano_categories_json}")

    cfg = Config.fromfile(args.config)
    set_model_resource(cfg)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    register_function(cfg._cfg_dict)

    all_classes, thing_classes, stuff_classes, anno = load_pano_categories(args.pano_categories_json)
    metadata = build_metadata_from_categories("panoptic_genseg_pano_predict", anno["categories"])

    os.makedirs(args.output_dir, exist_ok=True)
    pred_root = osp.join(args.output_dir, "pred_data", "panoptic_genseg_pano_predict")
    vis_dir = osp.join(pred_root, "vis")
    panoptic_png_dir = osp.join(pred_root, "panoptic_png")
    mask_npy_dir = osp.join(pred_root, "mask_npy")
    mask2_dir = osp.join(pred_root, "mask2")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(panoptic_png_dir, exist_ok=True)
    os.makedirs(mask_npy_dir, exist_ok=True)
    os.makedirs(mask2_dir, exist_ok=True)

    # Keep inference path consistent with eval_ori.py:
    # genseg in eval uses output_ids_with_output=True and mode='tensor'
    demo = XSamDemo(cfg, pth_model=None, output_ids_with_output=True)
    model = demo.model
    load_checkpoint(model, args.pth_model)
    model.eval()
    model = model.to(get_device())
    demo.model = model

    post_fn = demo.postprocess_fns["genseg"]
    if hasattr(post_fn, "keywords") and isinstance(post_fn.keywords, dict):
        post_fn.keywords["threshold"] = args.threshold
    demo.model.postprocess_fn = post_fn
    visualizer = Visualizer(metadata=metadata)

    image_files = sorted([n for n in os.listdir(args.image_dir) if is_image_file(n)])
    if args.max_images is not None and args.max_images > 0:
        image_files = image_files[: args.max_images]
    if not image_files:
        raise ValueError(f"no image files found in {args.image_dir}")

    summary = []
    for idx, img_name in enumerate(image_files):
        img_path = osp.join(args.image_dir, img_name)
        pil_image = Image.open(img_path).convert("RGB")

        data_dict = {"pil_image": pil_image, "vprompt_masks": None, "task_name": "genseg"}
        classes = (all_classes, thing_classes, stuff_classes)
        data_dict.update(demo._process_prompt("", "genseg", classes))
        data_dict.update(demo._process_image(pil_image))
        data_dict.update(demo._process_data_dict(data_dict))
        input_dict = xsam_collate_fn([data_dict])
        input_dict = data_dict_to_device(input_dict, device=model.device, dtype=model.dtype)
        model_inputs = input_dict["data_dict"]
        data_samples = input_dict["data_samples"]

        with torch.no_grad():
            _, seg_outputs = model(
                model_inputs,
                data_samples,
                mode="tensor",
                metadata=metadata,
                generation_config=demo.generation_config,
                stopping_criteria=demo.stop_criteria,
                do_postprocess=True,
                do_loss=False,
            )
        if seg_outputs is None or len(seg_outputs) == 0:
            print_log(f"[{idx+1}/{len(image_files)}] failed: {img_name}", logger="current")
            continue

        seg_out = seg_outputs[0]
        seg_map = to_numpy_seg(seg_out["segmentation"])
        segments_info_ds = convert_segments_info_to_dataset_id(seg_out.get("segments_info"), metadata)

        stem = osp.splitext(img_name)[0]
        np.save(osp.join(mask_npy_dir, f"{stem}.npy"), seg_map)

        from panopticapi.utils import id2rgb

        seg_rgb = id2rgb(seg_map)
        Image.fromarray(seg_rgb).save(osp.join(panoptic_png_dir, f"{stem}.png"))

        category_map = segmentation_ids_to_category_map(seg_map, segments_info_ds, background_category_id=0)
        np.save(osp.join(mask2_dir, f"{stem}.npy"), category_map)
        save_category_id_png(osp.join(mask2_dir, f"{stem}.png"), category_map)

        image = mmcv.imread(img_path)
        image = mmcv.imconvert(image, "bgr", "rgb")
        vis_seg_out = dict(seg_out)
        vis_seg_out["segments_info"] = segments_info_ds
        visualizer.draw_predictions(
            image,
            data_name="panoptic_genseg_pano_val",
            output_file=osp.join(vis_dir, f"{stem}.png"),
            **vis_seg_out,
        )

        summary.append(
            {
                "image_file": img_name,
                "mask_npy": f"mask_npy/{stem}.npy",
                "mask2_category_id_npy": f"mask2/{stem}.npy",
                "mask2_category_id_png": f"mask2/{stem}.png",
                "panoptic_png": f"panoptic_png/{stem}.png",
                "vis_png": f"vis/{stem}.png",
                "segments_info": segments_info_ds,
            }
        )
        print_log(f"[{idx+1}/{len(image_files)}] done: {img_name}", logger="current")

    with open(osp.join(pred_root, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(osp.join(pred_root, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"total_images={len(image_files)}\n")
        f.write(f"success={len(summary)}\n")
        f.write(f"output_dir={pred_root}\n")

    print_log(f"Predict done. outputs: {pred_root}", logger="current")


if __name__ == "__main__":
    main()
