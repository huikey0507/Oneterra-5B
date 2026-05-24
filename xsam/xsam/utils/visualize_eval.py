"""
评测专用可视化（eval_ori_batch.py 使用）。

与 xsam.utils.visualize.Visualizer 分离，避免改动影响 demo / app_021 / xsam_demo_021.sh。
"""

from __future__ import annotations

import numpy as np
import torch

from .visualize import ColorMode, Visualizer, _OFF_WHITE


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

        instances = instances.to(self.cpu_device)
        boxes = instances.pred_boxes if instances.has("pred_boxes") else None
        scores = instances.scores if instances.has("scores") else None
        classes = instances.pred_classes.tolist() if instances.has("pred_classes") else None
        labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
        keypoints = instances.pred_keypoints if instances.has("pred_keypoints") else None

        draw_masks = not (data_name and "detection" in data_name)
        if draw_masks and instances.has("pred_masks"):
            masks = np.asarray(instances.pred_masks)
            masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
        else:
            masks = None

        if self._instance_mode == ColorMode.SEGMENTATION:
            color_palette = getattr(self.metadata, "thing_colors", None) or getattr(
                self.metadata, "dataset_colors", None
            )
            colors = []
            for c in classes:
                pal_id = self._contiguous_id_for_palette(c)
                try:
                    fallback_seed = int(c)
                except (TypeError, ValueError):
                    fallback_seed = abs(hash(str(c))) % (2**32)
                base = self._get_palette_color(color_palette, pal_id, fallback_seed=fallback_seed)
                colors.append(self._jitter(base) if jittering else base)
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
