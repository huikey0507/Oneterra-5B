from typing import Any, Dict, Optional

from pycocotools import mask as mask_utils

from xsam.structures import BoxMode


def normalize_detection_annotation(ann: Dict[str, Any], require_segmentation: bool = False) -> Optional[Dict[str, Any]]:
    """Normalize a COCO annotation for detection datasets.

    Args:
        ann: Raw COCO annotation dict.
        require_segmentation: If True, drop annotations without valid segmentation
            (instance segmentation). If False, bbox-only annotations are kept.
    """
    if int(ann.get("iscrowd", 0)) != 0:
        return None
    if "bbox" not in ann:
        return None

    bbox = ann["bbox"]
    if len(bbox) == 5:
        ann["bbox_mode"] = BoxMode.XYWHA_ABS
    else:
        ann["bbox_mode"] = ann.get("bbox_mode", BoxMode.XYWH_ABS)

    segmentation = ann.get("segmentation")
    if segmentation is None:
        if require_segmentation:
            return None
        ann.pop("segmentation", None)
        return ann

    if isinstance(segmentation, dict):
        if isinstance(segmentation.get("counts"), list):
            segmentation = mask_utils.frPyObjects(segmentation["counts"], *segmentation["size"])
        if isinstance(segmentation.get("counts"), bytes):
            segmentation["counts"] = segmentation["counts"].decode("utf-8")
    else:
        segmentation = [poly for poly in segmentation if len(poly) % 2 == 0 and len(poly) >= 6]
        if len(segmentation) == 0:
            if require_segmentation:
                return None
            ann.pop("segmentation", None)
            return ann

    ann["segmentation"] = segmentation
    return ann
