import json
import os
import os.path as osp
import random
import tempfile

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils

from ..structures import BoxMode
from ..utils.logging import print_log
from ..utils.palette import get_palette
from .generic_seg_dataset import GenericSegDataset
from .utils.catalog import MetadataCatalog
from .utils.coco import COCO
from .utils.detection import normalize_detection_annotation


class OVSegDataset(GenericSegDataset):
    def __init__(self, *args, label_file=None, label_shift=0, **kwargs):
        super().__init__(*args, label_file=label_file, label_shift=label_shift, **kwargs)

    def custom_init(self, **kwargs):
        super().custom_init(**kwargs)
        self.label_file = kwargs.get("label_file", None)
        self.label_shift = kwargs.get("label_shift", 0)

    def _set_semantic_metadata(self, coco_data, **kwargs):
        metadata = MetadataCatalog.get(f"{self.data_name}")
        cats = coco_data["categories"]
        # OVSeg 的 prompt 类别顺序使用 sorted(cat_ids)；这里的 contiguous 映射必须一致，
        # 否则会出现稳定的一对一类别错位（如 A 全部显示为 B）。
        sorted_cats = sorted(cats, key=lambda x: x["id"])
        cat_colors = (
            [x["color"] for x in sorted_cats]
            if "color" in cats[0]
            else get_palette("random", len(cats))
        )
        dataset_id_to_contiguous_id = {x["id"]: i for i, x in enumerate(sorted_cats)}
        cat_id_to_name = {x["id"]: x["name"] for x in cats}
        cat_id_to_color = {x["id"]: cat_colors[dataset_id_to_contiguous_id[x["id"]]] for x in cats}

        metadata.set(
            gt_json=self.data_path,
            label_file=self.label_file,
            semseg_sufix=self.semseg_sufix,
            label_shift=self.label_shift,
            semseg_map_folder=self.semseg_map_folder,
            data_name=self.data_name,
            dataset_classes=cat_id_to_name,
            dataset_colors=cat_id_to_color,
            stuff_classes=cat_id_to_name,
            stuff_colors=cat_id_to_color,
            dataset_id_to_contiguous_id=dataset_id_to_contiguous_id,
            stuff_dataset_id_to_contiguous_id=dataset_id_to_contiguous_id,
            ignore_label=self.ignore_label,
            label_divisor=1000,
        )
        self._metadata = metadata

    def _load_ann_data(self):
        def _format_caption(caption):
            caption = caption.strip().capitalize()
            if not caption.endswith("."):
                caption = caption + "."
            return caption

        def _load_label_file():
            invalid_names = ["invalid_class_id", "background"]
            with open(self.label_file, "r") as f:
                lines = f.read().splitlines()
            categories = []
            for line in lines:
                id, name = line.split(":")
                name = name.split(",")[0]
                if name in invalid_names:
                    continue
                categories.append({"id": int(id), "name": name})
            return categories

        def _sample_cat_ids(cat_ids, sample_num=-1):
            if sample_num == -1:
                sample_num = len(cat_ids)
            cat_ids = random.sample(cat_ids, min(len(cat_ids), sample_num))
            return cat_ids

        if self.label_file is not None and self.data_path is None:
            cats = _load_label_file()
            coco_data = {
                "categories": cats,
            }
        else:
            with open(self.data_path, "r") as f:
                coco_data = json.load(f)

            cap_coco_api = None
            if self.caption_data_path is not None:
                cap_coco_api = COCO(self.caption_data_path)
            cats = coco_data["categories"]

        cat_ids = sorted([cat["id"] for cat in cats])
        cat_ids2names = {cat["id"]: cat["name"] for cat in cats}

        rets = []
        if "panoptic" in self.data_name:
            coco_data["images"] = sorted(coco_data["images"], key=lambda x: x["id"])
            coco_data["annotations"] = sorted(coco_data["annotations"], key=lambda x: x["image_id"])
            for _img_info, _ann_info in zip(coco_data["images"], coco_data["annotations"]):
                img_id = _img_info["id"]
                assert _img_info["id"] == _ann_info["image_id"]
                seg_map_path = _img_info["file_name"].replace(".jpg", ".png")

                if cap_coco_api is not None:
                    cap_ann_ids = cap_coco_api.getAnnIds(imgIds=[img_id])
                    cap_anns = cap_coco_api.loadAnns(cap_ann_ids)
                    caption = _format_caption(random.choice(cap_anns).get("caption", []))
                else:
                    caption = None

                segments_info = _ann_info["segments_info"]
                if len(segments_info) == 0:
                    self.woann_cnt += 1
                    continue

                # random sample cat_ids to shuffle the order
                sampled_cat_ids, sampled_segments_info = self._sample_cats(cat_ids, segments_info)
                sampled_cat_names = [cat_ids2names[cat_id] for cat_id in sampled_cat_ids]

                img_info = {
                    "file_name": _img_info["file_name"],
                    "image_id": _img_info["id"],
                    "height": _img_info["height"],
                    "width": _img_info["width"],
                }

                rets.append(
                    {
                        "image_id": _img_info["id"],
                        "image_file": _img_info["file_name"],
                        "image_size": (_img_info["height"], _img_info["width"]),
                        "caption": caption,
                        "seg_map": seg_map_path,
                        "segments_info": sampled_segments_info,
                        "sampled_cats": sampled_cat_names,
                        "sampled_labels": sampled_cat_ids,
                        "image_info": img_info,
                    }
                )
        elif "semantic" in self.data_name and self.semseg_map_folder is not None:
            # Prefer COCO image list when data_path is set (e.g. Potsdam tiles named 2_13_0_0_512_512.png).
            if self.data_path is not None and coco_data.get("images"):
                image_items = sorted(coco_data["images"], key=lambda x: x["id"])
            else:
                image_items = []
                for img_file in sorted(os.listdir(self.image_folder)):
                    img_path = osp.join(self.image_folder, img_file)
                    pil_image = Image.open(img_path)
                    width, height = pil_image.size
                    stem = osp.splitext(img_file)[0]
                    try:
                        img_id = int(stem)
                    except ValueError:
                        img_id = len(image_items) + 1
                    image_items.append(
                        {
                            "file_name": img_file,
                            "id": img_id,
                            "height": height,
                            "width": width,
                        }
                    )

            for _img_info in image_items:
                img_file = _img_info["file_name"]
                img_id = _img_info["id"]
                height = _img_info["height"]
                width = _img_info["width"]

                if len(cat_ids) > self.sample_num:
                    semseg_stem = osp.splitext(img_file)[0]
                    semseg_map = Image.open(osp.join(self.semseg_map_folder, semseg_stem + self.semseg_sufix))
                    semseg_map = np.array(semseg_map, dtype=np.uint32)
                    pos_cat_ids = np.unique(semseg_map[semseg_map != self.ignore_label]).tolist()
                    neg_cat_ids = sorted(list(set(cat_ids) - set(pos_cat_ids)))
                    neg_cat_ids = _sample_cat_ids(neg_cat_ids, self.sample_num - len(pos_cat_ids))
                    sampled_cat_ids = sorted(pos_cat_ids + neg_cat_ids)
                else:
                    sampled_cat_ids = cat_ids

                sampled_cat_names = [cat_ids2names[cat_id] for cat_id in sampled_cat_ids]
                img_info = {
                    "file_name": img_file,
                    "image_id": img_id,
                    "height": height,
                    "width": width,
                }
                rets.append(
                    {
                        "image_id": img_info["image_id"],
                        "image_file": img_info["file_name"],
                        "image_size": (img_info["height"], img_info["width"]),
                        "image_info": img_info,
                        "sampled_cats": sampled_cat_names,
                        "sampled_labels": sampled_cat_ids,
                        "caption": None,
                    }
                )
        elif "detection" in self.data_name or "instance" in self.data_name:
            require_segmentation = "instance" in self.data_name
            thing_dataset_ids = None
            if "detection" in self.data_name:
                thing_dataset_ids = {c["id"] for c in cats if c.get("isthing", 1) == 1}
            coco_api = COCO(dataset=coco_data)
            img_ids = sorted(coco_api.getImgIds())
            for img_id in img_ids:
                _img_info = coco_api.loadImgs([img_id])[0]
                ann_ids = coco_api.getAnnIds(imgIds=[img_id])
                _anns = coco_api.loadAnns(ann_ids)

                if cap_coco_api is not None and random.random() < 0.5:
                    cap_ann_ids = cap_coco_api.getAnnIds(imgIds=[img_id])
                    cap_anns = cap_coco_api.loadAnns(cap_ann_ids)
                    caption = _format_caption(random.choice(cap_anns).get("caption", []))
                else:
                    caption = None

                if len(_anns) == 0:
                    self.woann_cnt += 1
                    continue

                anns = []
                for ann in _anns:
                    if thing_dataset_ids is not None and ann["category_id"] not in thing_dataset_ids:
                        continue
                    normalized = normalize_detection_annotation(ann, require_segmentation=require_segmentation)
                    if normalized is not None:
                        anns.append(normalized)

                if len(anns) == 0:
                    self.woann_cnt += 1
                    continue

                # random sample cat_ids to shuffle the order
                sampled_cat_ids, sampled_anns = self._sample_cats(cat_ids, anns)
                sampled_cat_names = [cat_ids2names[cat_id] for cat_id in sampled_cat_ids]

                img_info = {
                    "file_name": _img_info["file_name"],
                    "image_id": _img_info["id"],
                    "height": _img_info["height"],
                    "width": _img_info["width"],
                }
                if "detection" in self.data_name:
                    img_info["gt_annotations"] = sampled_anns

                rets.append(
                    {
                        "image_id": _img_info["id"],
                        "image_file": _img_info["file_name"],
                        "image_size": (_img_info["height"], _img_info["width"]),
                        "caption": caption,
                        "annotations": sampled_anns,
                        "sampled_cats": sampled_cat_names,
                        "sampled_labels": sampled_cat_ids,
                        "image_info": img_info,
                    }
                )
        else:
            raise ValueError(f"Invalid dataset type: {self.data_name}")

        if self.data_mode == "eval" and ("instance" in self.data_name or "detection" in self.data_name):
            base_temp = tempfile.gettempdir()
            cache_dir = osp.join(base_temp, "xsam_cache")
            os.makedirs(cache_dir, exist_ok=True)
            temp_dir = tempfile.mkdtemp(dir=cache_dir)
            print_log(f"Writing {self.data_name} gt_json to {temp_dir}...", logger="current")
            temp_file = osp.join(temp_dir, f"{self.data_name}.json")
            with open(temp_file, "w") as f:
                json.dump(rets, f)
            self._set_metadata(coco_data, gt_json=temp_file)
        else:
            self._set_metadata(coco_data)

        del coco_data
        return rets

    def _decode_mask(self, data_dict):
        if ("semantic" in self.data_name or "detection" in self.data_name) and self.data_mode != "train":
            img_info = data_dict["image_info"]
            height, width = img_info["height"], img_info["width"]
            mask_labels = torch.zeros((0, height, width))
            class_labels = torch.zeros((0,), dtype=torch.int64)
            data_dict.update(
                {
                    "mask_labels": mask_labels,
                    "class_labels": class_labels,
                }
            )
        else:
            super()._decode_mask(data_dict)

        return data_dict
