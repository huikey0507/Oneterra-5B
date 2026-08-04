"""
评测专用可视化（eval_ori_batch.py 使用）。

与 xsam.utils.visualize.Visualizer 分离，避免改动影响 demo / app_021 / xsam_demo_021.sh。
"""

from __future__ import annotations

import matplotlib.colors as mplc
import numpy as np
import torch

from xsam.structures import RotatedBoxes
from xsam.utils.rbox_vis import (
    is_rbox_detection,
    pred_instances_to_rbox_for_draw,
    rbox_instances_for_draw,
)

from .visualize import ColorMode, Visualizer, _OFF_WHITE, _is_instance_val_data_name, filter_instances_to_things


class EvalVisualizer(Visualizer):
    """批量评测可视化：支持 Potsdam 等 OVSeg 全类 prompt + semantic/instance 命名。"""

    def draw_gen_seg(self, data_name, **kwargs):
        is_semantic = "semantic" in data_name or (
            "sem" in data_name and "ins" not in data_name and "panoptic" not in data_name
        )
        is_detection = "detection" in data_name
        is_instance = ("instance" in data_name and not is_detection) or (
            "ins" in data_name and "semantic" not in data_name and "panoptic" not in data_name and not is_detection
        )
        is_panoptic = "panoptic" in data_name or "pan" in data_name

        if is_semantic:
            return self.draw_sem_seg(**kwargs)
        if is_detection or is_instance:
            return self.draw_ins_seg(data_name=data_name, **kwargs)
        if is_panoptic:
            return self.draw_pan_seg(**kwargs)
        raise ValueError(f"Unsupported task: {data_name}")

    def draw_sem_seg(self, segmentation, area_threshold=None, alpha=0.8, **kwargs):
        if isinstance(segmentation, torch.Tensor):
            segmentation = segmentation.numpy()
        # pred=contiguous 下标；GT=dataset category_id。
        # 有 sampled_labels / metadata 时把 pred 映到 dataset_id，再查名字与调色板。
        sampled_labels = kwargs.pop("sampled_labels", None)
        remap_pred_to_dataset_id = kwargs.pop("remap_pred_to_dataset_id", None)
        if sampled_labels is None and remap_pred_to_dataset_id is None:
            d2c = getattr(self.metadata, "dataset_id_to_contiguous_id", None) or {}
            if d2c:
                cont_to_ds = {int(v): int(k) for k, v in d2c.items()}
                # 仅当像素值落在 contiguous 范围时自动反查（避免误伤已是 dataset_id 的 GT）
                uniq = [int(x) for x in np.unique(segmentation)]
                if uniq and max(uniq) < len(cont_to_ds) and all(u in cont_to_ds for u in uniq if u >= 0):
                    remap_pred_to_dataset_id = True
                    sampled_labels = [cont_to_ds[i] for i in range(len(cont_to_ds))]
        if sampled_labels is not None and remap_pred_to_dataset_id is not False:
            if isinstance(sampled_labels, torch.Tensor):
                sampled_labels = sampled_labels.tolist()
            out = np.array(segmentation, copy=True)
            for cid in np.unique(segmentation):
                cid = int(cid)
                if 0 <= cid < len(sampled_labels):
                    out[segmentation == cid] = int(sampled_labels[cid])
            segmentation = out
        labels, areas = np.unique(segmentation, return_counts=True)
        sorted_idxs = np.argsort(-areas).tolist()
        labels = labels[sorted_idxs]
        stuff_palette = getattr(self.metadata, "stuff_colors", None) or getattr(
            self.metadata, "dataset_colors", None
        )
        for label in labels:
            if label < 0:
                continue
            pal_id = self._contiguous_id_for_palette(int(label))
            try:
                fallback_seed = int(label)
            except (TypeError, ValueError):
                fallback_seed = abs(hash(str(label))) % (2**32)
            mask_color = self._get_palette_color(stuff_palette, pal_id, fallback_seed=fallback_seed)
            binary_mask = (segmentation == label).astype(np.uint8)
            text = self._get_category_name(int(label))
            self.draw_binary_mask(
                binary_mask,
                color=mask_color,
                edge_color=_OFF_WHITE,
                text=text,
                alpha=alpha,
                area_threshold=area_threshold,
            )
        return self.output

    def draw_ins_seg(self, instances, jittering: bool = True, data_name=None, **kwargs):
        from .visualize import GenericMask, _create_text_labels

        if _is_instance_val_data_name(data_name):
            instances = filter_instances_to_things(instances, self.metadata)
        instances = instances.to(self.cpu_device)
        scores = instances.scores if instances.has("scores") else None
        classes = instances.pred_classes.tolist() if instances.has("pred_classes") else None
        labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
        keypoints = instances.pred_keypoints if instances.has("pred_keypoints") else None

        use_rbox = is_rbox_detection(data_name, self.metadata) and bool(
            data_name and "detection" in data_name
        )

        if use_rbox:
            if instances.has("pred_masks") and len(instances) > 0:
                # pred：由 mask 拟合 OBB，角度已在 rbox_vis 中转为 CCW 度
                boxes = pred_instances_to_rbox_for_draw(instances)
            elif instances.has("pred_boxes") and isinstance(instances.pred_boxes, RotatedBoxes):
                # GT：prepare_gt_data_detection 存的是 OpenCV 角，画之前取反
                boxes = rbox_instances_for_draw(instances.pred_boxes)
            else:
                boxes = instances.pred_boxes if instances.has("pred_boxes") else None
            masks = None
        else:
            # 水平框 / instance：保持原有逻辑
            boxes = instances.pred_boxes if instances.has("pred_boxes") else None
            draw_masks = not (data_name and "detection" in data_name)
            if draw_masks and instances.has("pred_masks"):
                masks = np.asarray(instances.pred_masks)
                masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
            else:
                masks = None

        if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
            colors = (
                [self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in classes]
                if jittering
                else [tuple(mplc.to_rgb([x / 255 for x in self.metadata.thing_colors[c]])) for c in classes]
            )
            alpha = 0.8
        else:
            colors = None
            alpha = 0.5

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.reset_image(
                self._create_grayscale_image(
                    (instances.pred_masks.any(dim=0) > 0).numpy() if instances.has("pred_masks") else None
                )
            )
            alpha = 0.3

        self.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )
        return self.output
