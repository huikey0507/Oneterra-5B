"""Rotated-bbox visualization helpers (eval only).

OpenCV / mmcv use clockwise angles; Detectron2 ``RotatedBoxes`` and ``draw_rotated_box_with_label``
use counter-clockwise degrees. These helpers are only for drawing — do not use in mAP eval.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from xsam.structures import RotatedBoxes


def is_rbox_detection(data_name: Optional[str], metadata=None) -> bool:
    """True when detection eval/visualization should use oriented boxes."""
    if metadata is not None:
        gt_bbox_type = getattr(metadata, "gt_bbox_type", None)
        if gt_bbox_type == "rbox":
            return True
        if gt_bbox_type == "hbox":
            return False
    if data_name and "dior_r" in data_name:
        return True
    return False


def opencv_angle_to_draw_degrees(angle) -> float:
    """OpenCV/mmcv angle (rad or deg) -> Detectron2 drawer CCW degrees."""
    angle = float(angle)
    if abs(angle) <= math.pi + 1e-3:
        angle = math.degrees(angle)
    return -angle


def _mask_to_opencv_xywha(mask) -> Optional[tuple]:
    if cv2 is None:
        return None
    mask_uint8 = np.ascontiguousarray(mask.astype(np.uint8))
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 1:
        return None
    (cx, cy), (w, h), angle_deg = cv2.minAreaRect(cnt)
    if w <= 0 or h <= 0:
        return None
    return float(cx), float(cy), float(w), float(h), float(np.deg2rad(angle_deg))


def mask_to_rbox_for_draw(mask):
    """Binary mask -> [cx, cy, w, h, angle_deg] for ``draw_rotated_box_with_label``."""
    xywha = _mask_to_opencv_xywha(mask)
    if xywha is None:
        return None
    cx, cy, w, h, angle = xywha
    return [cx, cy, w, h, opencv_angle_to_draw_degrees(angle)]


def rbox_instances_for_draw(boxes) -> RotatedBoxes:
    """Flip OpenCV-style stored angles to Detectron2 CCW degrees (GT vis)."""
    if boxes is None or not isinstance(boxes, RotatedBoxes):
        return boxes
    tensor = boxes.tensor.clone()
    tensor[:, 4] = -tensor[:, 4]
    return RotatedBoxes(tensor)


def pred_instances_to_rbox_for_draw(instances):
    """Fit drawable OBB from pred masks; fallback to axis-aligned hbox as 0-deg rbox."""
    boxes = instances.pred_boxes if instances.has("pred_boxes") else None
    if not instances.has("pred_masks") or len(instances) == 0:
        return boxes

    rboxes = []
    fallback = boxes.tensor.numpy() if boxes is not None else None
    for i, mask in enumerate(np.asarray(instances.pred_masks)):
        rb = mask_to_rbox_for_draw(mask)
        if rb is not None:
            rboxes.append(rb)
            continue
        if fallback is not None and i < len(fallback) and len(fallback[i]) == 4:
            x0, y0, x1, y1 = map(float, fallback[i])
            rboxes.append([(x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, 1.0), max(y1 - y0, 1.0), 0.0])
    if not rboxes:
        return boxes
    return RotatedBoxes(torch.tensor(rboxes, dtype=torch.float32))
