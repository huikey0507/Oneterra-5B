from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import TensorType

from ...utils.process import sem_seg_postprocess
from .refer_seg_process_fn import refer_seg_postprocess_fn

# 全分辨率上对 num_queries 条 mask 做插值+sigmoid 的 float32 元素量上限。
# 超过则对本样本走 query-first（EarthReason 6k+ 原图）；未超过则与 refer_seg 单样本逻辑一致。
_DEFAULT_MAX_FULLRES_MASK_ELEMENTS = 2_000_000_000


def _refer_seg_postprocess_one(
    mask_pred,
    mask_cls,
    image_size,
    scaled_size,
    mask_threshold: float,
) -> Dict[str, TensorType]:
    mask_pred = sem_seg_postprocess(mask_pred, scaled_size, image_size[0], image_size[1])
    mask_prob = mask_pred.sigmoid()
    scores = F.softmax(mask_cls, dim=-1)[:, :-1]
    top_score, top_index = scores.max(dim=0)
    mask_pred = mask_pred[top_index]
    mask_prob = mask_pred.sigmoid()
    segmentation = torch.full((image_size[0], image_size[1]), 255, dtype=torch.long, device=mask_pred.device)
    segmentation[mask_prob[0] > mask_threshold] = 1
    return {
        "segmentation": segmentation,
        "segments_info": {
            "id": 0,
            "label_id": 0,
            "was_fused": False,
            "score": round(top_score.item(), 6),
        },
    }


def _reason_seg_postprocess_one_large(
    mask_pred,
    mask_cls,
    image_size,
    scaled_size,
    mask_threshold: float,
) -> Dict[str, TensorType]:
    scores = F.softmax(mask_cls, dim=-1)[:, :-1]
    top_score, top_index = scores.max(dim=0)
    mask_pred = mask_pred[top_index : top_index + 1]
    mask_pred = sem_seg_postprocess(mask_pred, scaled_size, image_size[0], image_size[1])
    mask_prob = mask_pred.sigmoid()
    segmentation = torch.full((image_size[0], image_size[1]), 255, dtype=torch.long, device=mask_pred.device)
    segmentation[mask_prob[0] > mask_threshold] = 1
    return {
        "segmentation": segmentation,
        "segments_info": {
            "id": 0,
            "label_id": 0,
            "was_fused": False,
            "score": round(top_score.item(), 6),
        },
    }


def reason_seg_postprocess_fn(
    outputs,
    image_sizes,
    scaled_sizes: Optional[List[TensorType]] = None,
    mask_threshold: float = 0.5,
    max_fullres_mask_elements: int = _DEFAULT_MAX_FULLRES_MASK_ELEMENTS,
    **kwargs,
) -> List[Dict]:
    class_queries_logits = outputs.class_queries_logits
    masks_queries_logits = outputs.masks_queries_logits
    scaled_sizes = scaled_sizes if scaled_sizes is not None else image_sizes

    batch_size = class_queries_logits.shape[0]
    num_labels = class_queries_logits.shape[-1] - 1
    assert num_labels == 1

    if batch_size == 1:
        i = 0
        h, w = int(image_sizes[i][0]), int(image_sizes[i][1])
        nq = int(masks_queries_logits.shape[1])
        if nq * h * w <= max_fullres_mask_elements:
            return refer_seg_postprocess_fn(
                outputs,
                image_sizes,
                scaled_sizes=scaled_sizes,
                mask_threshold=mask_threshold,
                **kwargs,
            )

    results: List[Dict[str, TensorType]] = []
    for i in range(batch_size):
        mask_pred = masks_queries_logits[i]
        mask_cls = class_queries_logits[i]
        image_size = image_sizes[i]
        scaled_size = scaled_sizes[i]
        h, w = int(image_size[0]), int(image_size[1])
        nq = int(mask_pred.shape[0])

        if nq * h * w <= max_fullres_mask_elements:
            results.append(
                _refer_seg_postprocess_one(mask_pred, mask_cls, image_size, scaled_size, mask_threshold)
            )
        else:
            results.append(
                _reason_seg_postprocess_one_large(mask_pred, mask_cls, image_size, scaled_size, mask_threshold)
            )
    return results
