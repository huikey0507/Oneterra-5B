import numpy as np
import pycocotools.mask as mask_util

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from xsam.structures import BoxMode

ANGLE_EPS = np.deg2rad(0.1)


def infer_annotation_bbox_type(annotation):
    """Infer whether a single GT annotation uses horizontal or rotated bbox."""
    bbox_mode = annotation.get("bbox_mode")
    bbox = annotation.get("bbox")
    if bbox is None:
        return None

    bbox = np.asarray(bbox, dtype=np.float32).reshape(-1)
    if bbox_mode == BoxMode.XYWHA_ABS:
        return "rbox"
    if bbox_mode == BoxMode.XYWH_ABS:
        if bbox.shape[0] == 4:
            return "hbox"
        if bbox.shape[0] == 5:
            angle = bbox[4]
            if abs(angle) > np.pi:
                angle = np.deg2rad(angle)
            return "rbox" if abs(angle) > ANGLE_EPS else "hbox"

    if bbox.shape[0] == 4:
        return "hbox"
    if bbox.shape[0] == 5:
        angle = bbox[4]
        if abs(angle) > np.pi:
            angle = np.deg2rad(angle)
        return "rbox" if abs(angle) > ANGLE_EPS else "hbox"
    return None


def detect_gt_bbox_type(gt_anns, metadata=None):
    """Detect whether GT uses horizontal or rotated bounding boxes.

    Priority:
    1. metadata.gt_bbox_type in {"hbox", "rbox"}
    2. Majority vote over all instance annotations with explicit bbox
    3. Default to "hbox"
    """
    if metadata is not None:
        gt_bbox_type = getattr(metadata, "gt_bbox_type", None)
        if gt_bbox_type in ("hbox", "rbox"):
            return gt_bbox_type

    counts = {"hbox": 0, "rbox": 0}
    for gt_item in gt_anns:
        for ann in gt_item.get("annotations", []):
            ann_type = infer_annotation_bbox_type(ann)
            if ann_type is not None:
                counts[ann_type] += 1

    if counts["rbox"] > counts["hbox"]:
        return "rbox"
    if counts["hbox"] > 0:
        return "hbox"
    return "hbox"


def _mask_to_rbox(mask):
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
    return np.array([cx, cy, w, h, np.deg2rad(angle_deg)], dtype=np.float32)


def _xywh_to_rbox(bbox):
    x, y, w, h = bbox[:4]
    if w <= 0 or h <= 0:
        return None
    return np.array([x + w / 2.0, y + h / 2.0, w, h, 0.0], dtype=np.float32)


def _bbox_array_to_rbox(bbox, bbox_mode=BoxMode.XYWH_ABS):
    bbox = np.asarray(bbox, dtype=np.float32).reshape(-1)
    if bbox.shape[0] == 5:
        cx, cy, w, h, angle = bbox.tolist()
        if abs(angle) > np.pi:
            angle = np.deg2rad(angle)
        if w <= 0 or h <= 0:
            return None
        return np.array([cx, cy, w, h, angle], dtype=np.float32)
    if bbox.shape[0] == 4:
        bbox_xywh = bbox
        if bbox_mode != BoxMode.XYWH_ABS:
            bbox_xywh = np.array(
                BoxMode.convert(bbox.tolist(), bbox_mode, BoxMode.XYWH_ABS),
                dtype=np.float32,
            )
        return _xywh_to_rbox(bbox_xywh)
    return None


def _decode_annotation_mask(annotation, height, width):
    segmentation = annotation.get("segmentation")
    if segmentation is None:
        return None
    if isinstance(segmentation, dict):
        return mask_util.decode(segmentation)
    rles = mask_util.frPyObjects(segmentation, height, width)
    return mask_util.decode(rles)


def _annotation_to_rbox(annotation, height, width):
    """Convert GT annotation to rotated box; prefer explicit bbox over mask."""
    bbox = annotation.get("bbox")
    if bbox is not None:
        rbox = _bbox_array_to_rbox(bbox, annotation.get("bbox_mode", BoxMode.XYWH_ABS))
        if rbox is not None:
            return rbox

    mask = _decode_annotation_mask(annotation, height, width)
    if mask is not None:
        return _mask_to_rbox(mask)
    return None


def _prediction_to_rbox(prediction, height, width):
    """Convert prediction to rotated box; prefer mask fit for segmentation outputs."""
    segmentation = prediction.get("segmentation")
    if segmentation is not None:
        mask = mask_util.decode(segmentation)
        rbox = _mask_to_rbox(mask)
        if rbox is not None:
            return rbox

    bbox = prediction.get("bbox")
    if bbox is None:
        return None
    return _bbox_array_to_rbox(bbox, BoxMode.XYWH_ABS)


def build_rbox_eval_inputs(gt_anns, predictions, num_classes, category_id_to_contiguous):
    pred_by_image = {item["image_id"]: item.get("instances", []) for item in predictions}
    det_results = []
    annotations = []

    for gt_item in gt_anns:
        image_info = gt_item.get("image_info", {})
        height = image_info.get("height")
        width = image_info.get("width")
        image_id = gt_item["image_id"]

        gt_bboxes = []
        gt_labels = []
        for ann in gt_item.get("annotations", []):
            if height is None or width is None:
                continue
            rbox = _annotation_to_rbox(ann, height, width)
            if rbox is None:
                continue
            dataset_cat_id = ann["category_id"]
            if dataset_cat_id not in category_id_to_contiguous:
                continue
            gt_bboxes.append(rbox)
            gt_labels.append(category_id_to_contiguous[dataset_cat_id])

        if gt_bboxes:
            annotations.append(
                {
                    "bboxes": np.stack(gt_bboxes, axis=0),
                    "labels": np.asarray(gt_labels, dtype=np.int64),
                }
            )
        else:
            annotations.append({"bboxes": np.zeros((0, 5), dtype=np.float32), "labels": np.zeros((0,), dtype=np.int64)})

        per_class_dets = [[] for _ in range(num_classes)]
        for pred in pred_by_image.get(image_id, []):
            if height is None or width is None:
                continue
            rbox = _prediction_to_rbox(pred, height, width)
            if rbox is None:
                continue
            cat_id = pred["category_id"]
            if cat_id < 0 or cat_id >= num_classes:
                continue
            score = float(pred.get("score", 1.0))
            per_class_dets[cat_id].append(np.append(rbox, score))

        per_class_results = []
        for class_dets in per_class_dets:
            if class_dets:
                per_class_results.append(np.stack(class_dets, axis=0))
            else:
                per_class_results.append(np.zeros((0, 6), dtype=np.float32))
        det_results.append(per_class_results)

    return det_results, annotations
