#!/usr/bin/env python

import argparse
import copy
import json
import os
import os.path as osp
import sys
from typing import Dict, List, Tuple

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmengine.config import Config, DictAction
from mmengine.runner.utils import set_random_seed
from PIL import Image
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from xtuner.configs import cfgs_name_path
from xtuner.tools.utils import set_model_resource

from xsam.dataset.collate_fns import xsam_collate_fn
from xsam.demo.demo import OVSEG_OPEN_METADATA_NAME, XSamDemo, _panoptic_pred_segments_info_for_vis
from xsam.utils.checkpoint import load_checkpoint
from xsam.utils.dist import setup_distributed
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

PRED_DATA_NAME = "panoptic_ovseg_predict"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch predict open-vocabulary panoptic ovseg for an image folder")
    parser.add_argument("config", help="config file name or path")
    parser.add_argument("--pth_model", type=str, required=True, help="path to model checkpoint")
    parser.add_argument("--image-dir", type=str, required=True, help="input image directory")
    parser.add_argument("--output-dir", type=str, required=True, help="output root directory")
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help=(
            "open-vocabulary class prompt, e.g. "
            "'thing: person, car; stuff: road, sky' or comma-only list (all stuff)"
        ),
    )
    parser.add_argument("--threshold", type=float, default=0.0, help="postprocess threshold")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--max-images", type=int, default=-1, help="max images to run, -1 for all")
    parser.add_argument("--batch-size", type=int, default=1, help="inference batch size per GPU")
    parser.add_argument("--num-workers", type=int, default=4, help="dataloader num_workers per GPU")
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="distributed launcher type",
    )
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override config options, format: xxx=yyy",
    )
    return parser.parse_args()


def is_image_file(name: str) -> bool:
    ext = osp.splitext(name.lower())[1]
    return ext in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class OvsegImageFolderDataset(Dataset):
    """Image folder dataset for ovseg batch inference."""

    def __init__(self, image_dir: str, image_files: List[str], demo: XSamDemo, classes: Tuple):
        self.image_dir = image_dir
        self.image_files = image_files
        self.demo = demo
        self.classes = classes

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict:
        img_name = self.image_files[idx]
        img_path = osp.join(self.image_dir, img_name)
        pil_image = Image.open(img_path).convert("RGB")

        data_dict: Dict = {"pil_image": pil_image, "vprompt_masks": None, "task_name": "ovseg"}
        data_dict.update(self.demo._process_prompt("", "ovseg", self.classes))
        data_dict.update(self.demo._process_image(pil_image))
        data_dict.update(self.demo._process_data_dict(data_dict))
        data_dict["image_file"] = img_name
        return data_dict


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
    out = np.full(seg_map.shape, background_category_id, dtype=np.int32)
    for s in segments_info:
        sid = int(s["id"])
        cid = int(s["category_id"])
        out[seg_map == sid] = cid
    return out


def save_category_id_png(path: str, category_map: np.ndarray) -> None:
    cm = category_map.astype(np.int32)
    if cm.min() >= 0 and cm.max() <= 255:
        Image.fromarray(cm.astype(np.uint8), mode="L").save(path)
    else:
        Image.fromarray(cm, mode="I").save(path)


def filter_ovseg_seg_output(seg_out: Dict, classes: Tuple) -> Dict:
    """Keep only segments whose category_id is in user-defined open classes."""
    all_classes, _, _ = classes
    valid_category_ids = set(range(len(all_classes)))
    seg_output = copy.deepcopy(seg_out)
    if "segments_info" not in seg_output:
        return seg_output

    original_segments_info = seg_output["segments_info"]
    filtered_segments_info = []
    seg = seg_output["segmentation"]
    if torch.is_tensor(seg):
        filtered_segmentation = seg.clone()
    else:
        filtered_segmentation = np.array(seg, copy=True)

    valid_segment_id = 1
    old_segment_id_to_new = {}

    for seg_info in original_segments_info:
        category_id = int(seg_info.get("category_id", -1))
        old_segment_id = int(seg_info.get("id", 0))
        if category_id in valid_category_ids:
            if old_segment_id not in old_segment_id_to_new:
                old_segment_id_to_new[old_segment_id] = valid_segment_id
                valid_segment_id += 1
            new_seg_info = dict(seg_info)
            new_seg_info["id"] = old_segment_id_to_new[old_segment_id]
            filtered_segments_info.append(new_seg_info)
        else:
            if torch.is_tensor(filtered_segmentation):
                filtered_segmentation[filtered_segmentation == old_segment_id] = 0
            else:
                filtered_segmentation[filtered_segmentation == old_segment_id] = 0

    seg_output["segments_info"] = filtered_segments_info
    seg_output["segmentation"] = filtered_segmentation
    return seg_output


def main():
    args = parse_args()

    rank, local_rank, world_size = setup_distributed(args)
    is_main = rank == 0

    if args.seed is not None:
        set_random_seed(args.seed + rank)

    if not osp.isfile(args.config):
        if args.config in cfgs_name_path:
            args.config = cfgs_name_path[args.config]
        else:
            raise FileNotFoundError(f"Cannot find config: {args.config}")

    if not osp.exists(args.pth_model):
        raise FileNotFoundError(f"checkpoint not found: {args.pth_model}")
    if not osp.isdir(args.image_dir):
        raise NotADirectoryError(f"image dir not found: {args.image_dir}")
    if not args.prompt or not args.prompt.strip():
        raise ValueError("ovseg requires --prompt with at least one class (thing:/stuff: or comma list)")

    cfg = Config.fromfile(args.config)
    set_model_resource(cfg)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    register_function(cfg._cfg_dict)

    demo = XSamDemo(cfg, pth_model=None, output_ids_with_output=True)
    classes, _ = demo._get_classes_from_prompt(args.prompt, "ovseg")
    metadata = demo._set_metadata("ovseg", classes)

    if is_main:
        all_classes, thing_classes, stuff_classes = classes
        print_log(f"OVSeg prompt: {args.prompt}", logger="current")
        print_log(f"OVSeg all_classes ({len(all_classes)}): {all_classes}", logger="current")
        print_log(f"OVSeg thing ({len(thing_classes)}): {thing_classes}", logger="current")
        print_log(f"OVSeg stuff ({len(stuff_classes)}): {stuff_classes}", logger="current")

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
    pred_root = osp.join(args.output_dir, "pred_data", PRED_DATA_NAME)
    vis_dir = osp.join(pred_root, "vis")
    panoptic_png_dir = osp.join(pred_root, "panoptic_png")
    mask_npy_dir = osp.join(pred_root, "mask_npy")
    mask2_dir = osp.join(pred_root, "mask2")
    if is_main:
        os.makedirs(vis_dir, exist_ok=True)
        os.makedirs(panoptic_png_dir, exist_ok=True)
        os.makedirs(mask_npy_dir, exist_ok=True)
        os.makedirs(mask2_dir, exist_ok=True)
        with open(osp.join(pred_root, "ovseg_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(args.prompt.strip() + "\n")
            f.write(f"all_classes={classes[0]}\n")
            f.write(f"thing_classes={classes[1]}\n")
            f.write(f"stuff_classes={classes[2]}\n")
    if world_size > 1:
        dist.barrier()

    model = demo.model
    load_checkpoint(model, args.pth_model)
    model.eval()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    demo.model = model

    post_fn = demo.postprocess_fns["ovseg"]
    if hasattr(post_fn, "keywords") and isinstance(post_fn.keywords, dict):
        post_fn.keywords["threshold"] = args.threshold
    demo.model.postprocess_fn = post_fn

    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    visualizer = Visualizer(metadata=metadata)
    if hasattr(visualizer, "_category_id_to_name_cache"):
        visualizer._category_id_to_name_cache = {}

    image_files = sorted([n for n in os.listdir(args.image_dir) if is_image_file(n)])
    if args.max_images is not None and args.max_images > 0:
        image_files = image_files[: args.max_images]
    if not image_files:
        raise ValueError(f"no image files found in {args.image_dir}")

    dataset = OvsegImageFolderDataset(args.image_dir, image_files, demo, classes)

    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        shuffle = False
    else:
        sampler = None
        shuffle = False

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=xsam_collate_fn,
        pin_memory=False,
        drop_last=False,
    )

    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    vis_sampled_labels = list(range(len(classes[0])))

    summary = []
    total = len(image_files)

    for batch in tqdm(dataloader, desc=f"[rank {rank}] ovseg predicting", disable=not is_main):
        batch = data_dict_to_device(batch, device=device, dtype=raw_model.dtype)
        model_inputs = batch["data_dict"]
        data_samples = batch["data_samples"]
        batch_img_files = data_samples.image_files

        with torch.no_grad():
            _, seg_outputs = raw_model(
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
            for img_name in batch_img_files:
                print_log(f"failed: {img_name}", logger="current")
            continue

        for img_name, seg_out in zip(batch_img_files, seg_outputs):
            seg_out = filter_ovseg_seg_output(seg_out, classes)
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

            img_path = osp.join(args.image_dir, img_name)
            image = mmcv.imread(img_path)
            image = mmcv.imconvert(image, "bgr", "rgb")
            vis_seg_out = dict(seg_out)
            vis_seg_out["segments_info"] = _panoptic_pred_segments_info_for_vis(
                segments_info_ds, metadata, sampled_labels=vis_sampled_labels
            )
            visualizer.metadata = metadata
            visualizer.draw_predictions(
                image,
                data_name=OVSEG_OPEN_METADATA_NAME,
                output_file=osp.join(vis_dir, f"{stem}.png"),
                **vis_seg_out,
            )

            summary.append(
                {
                    "image_file": img_name,
                    "prompt": args.prompt.strip(),
                    "all_classes": classes[0],
                    "thing_classes": classes[1],
                    "stuff_classes": classes[2],
                    "mask_npy": f"mask_npy/{stem}.npy",
                    "mask2_category_id_npy": f"mask2/{stem}.npy",
                    "mask2_category_id_png": f"mask2/{stem}.png",
                    "panoptic_png": f"panoptic_png/{stem}.png",
                    "vis_png": f"vis/{stem}.png",
                    "segments_info": segments_info_ds,
                }
            )
            print_log(f"done: {img_name}", logger="current")

    if world_size > 1:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, summary)
        if is_main:
            merged: List[Dict] = []
            for sub in gathered:
                merged.extend(sub)
            order = {n: i for i, n in enumerate(image_files)}
            merged.sort(key=lambda x: order.get(x["image_file"], len(image_files)))
            seen: set = set()
            deduped: List[Dict] = []
            for item in merged:
                if item["image_file"] not in seen:
                    seen.add(item["image_file"])
                    deduped.append(item)
            summary = deduped

    if is_main:
        with open(osp.join(pred_root, "predictions.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        with open(osp.join(pred_root, "summary.txt"), "w", encoding="utf-8") as f:
            f.write(f"total_images={total}\n")
            f.write(f"success={len(summary)}\n")
            f.write(f"prompt={args.prompt.strip()}\n")
            f.write(f"output_dir={pred_root}\n")
        print_log(f"OVSeg predict done. outputs: {pred_root}", logger="current")

    if world_size > 1:
        dist.barrier()


if __name__ == "__main__":
    main()
