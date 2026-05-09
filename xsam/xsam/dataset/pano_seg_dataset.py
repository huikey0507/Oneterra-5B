"""
Pano 训练集：与 GenSegDataset._set_panoptic_metadata 使用同一套 contiguous 规则。

categories 在 JSON 中的顺序与 GenericSegDataset 中
dataset_id_to_contiguous_id = {x["id"]: i for i, x in enumerate(cats)} 一致，
避免「仅按标注出现过的 cat 排序」与 val/evaluator 的 metadata 不一致。
"""
import json
import os

import numpy as np
import torch

from xsam.dataset import GenericSegDataset


class PanoSegDataset(GenericSegDataset):
    """Pano 数据集：类别 ID 映射与 GenSeg 的 panoptic metadata 对齐。"""

    def custom_init(self, **kwargs):
        super().custom_init(**kwargs)
        self._init_category_mapping()

    def _init_category_mapping(self):
        """与 GenericSegDataset 一致：按 annotations JSON 中 categories 列表顺序分配 contiguous。"""
        with open(self.data_path, "r") as f:
            coco_data = json.load(f)

        cats = coco_data.get("categories", [])
        if not cats:
            raise ValueError(f"PanoSegDataset: {self.data_path} 缺少 categories，无法构建与 eval 一致的类别索引")

        self.cat_id_mapping = {int(c["id"]): i for i, c in enumerate(cats)}
        self.reverse_mapping = {i: int(c["id"]) for i, c in enumerate(cats)}

        print(
            f"PanoSegDataset: 类别映射与 GenSeg metadata 对齐，共 {len(self.cat_id_mapping)} 个 category（含 id=0 若存在）"
        )

    def _decode_mask(self, data_dict):
        if "panoptic" in self.data_name:
            segments_info = data_dict.get("segments_info", None)
            seg_map_path = data_dict.get("seg_map", None)
            if seg_map_path is None:
                height, width = data_dict["image_size"]
                mask_labels = torch.zeros((0, height, width))
                class_labels = torch.zeros((0,))
            else:
                from panopticapi.utils import rgb2id
                from PIL import Image

                seg_map = Image.open(os.path.join(self.panseg_map_folder, seg_map_path)).convert("RGB")
                seg_map = rgb2id(np.array(seg_map))

                mask_labels = []
                class_labels = []
                for segment_info in segments_info:
                    cat_id = segment_info["category_id"]
                    if not segment_info["iscrowd"]:
                        if cat_id == 0:
                            continue
                        if cat_id in self.cat_id_mapping:
                            mapped_cat_id = self.cat_id_mapping[cat_id]
                            mask = seg_map == segment_info["id"]
                            class_labels.append(mapped_cat_id)
                            mask_labels.append(mask)

                if len(mask_labels) == 0:
                    mask_labels = torch.zeros((0, seg_map.shape[-2], seg_map.shape[-1]))
                    class_labels = torch.zeros((0,), dtype=torch.int64)
                else:
                    mask_labels = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in mask_labels])
                    class_labels = torch.tensor(np.array(class_labels), dtype=torch.int64)

            del data_dict["segments_info"]
            del data_dict["seg_map"]
            data_dict.update(
                {
                    "mask_labels": mask_labels,
                    "class_labels": class_labels,
                }
            )
        else:
            return super()._decode_mask(data_dict)

        return data_dict
