import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import TensorType

from xsam.structures import BitMasks, Instances
from xsam.utils.logging import print_log

from ...utils.mask import convert_segmentation_to_rle
from ...utils.process import compute_segments, remove_low_and_no_objects, sem_seg_postprocess


_OVSEG_DEBUG_PRINT_COUNT = 0
_OVSEG_DEBUG_MAX_PRINT = 20


def _filter_detection_instances(instances, score_thr: float, nms_thr: float = 0.5):
    """按分数阈值 + 逐类 NMS 去重，避免 Mask2Former 多 query 重复框。"""
    if instances is None or len(instances) == 0:
        return instances
    # pred_boxes 来自 BitMasks.get_bounding_boxes()，默认在 CPU；scores 可能在 GPU
    instances = instances.to("cpu")
    keep = instances.scores >= score_thr
    if not keep.any():
        return instances[keep]
    instances = instances[keep]
    if len(instances) == 0:
        return instances

    try:
        from torchvision.ops import batched_nms
    except ImportError:
        return instances

    boxes = instances.pred_boxes.tensor
    scores = instances.scores
    classes = instances.pred_classes
    keep_idx = batched_nms(boxes, scores, classes, nms_thr)
    return instances[keep_idx]


def generic_seg_postprocess_fn(
    outputs,
    image_sizes,
    task_name: str = "panoptic_genseg",
    scaled_sizes: Optional[List[TensorType]] = None,
    threshold: float = 0.5,
    mask_threshold: float = 0.5,
    overlap_mask_area_threshold: float = 0.8,
    **kwargs,
) -> List[Dict]:

    def _semantic_genseg_postprocess(outputs, image_sizes, scaled_sizes, sampled_labels=None, **kwargs):
        # [batch_size, num_queries, num_classes+1]
        class_queries_logits = outputs.class_queries_logits
        # [batch_size, num_queries, height, width]
        masks_queries_logits = outputs.masks_queries_logits
        scaled_sizes = scaled_sizes if scaled_sizes is not None else image_sizes

        batch_size = class_queries_logits.shape[0]
        mask_classes = class_queries_logits.softmax(dim=-1)[:, :, :-1]
        mask_probs = masks_queries_logits.sigmoid()
        segmentations = torch.einsum("bqc,bqhw->bchw", mask_classes, mask_probs).cpu()

        # Loop over items in batch size
        results: List[Dict[str, TensorType]] = []

        for i in range(batch_size):
            image_size = image_sizes[i]
            scaled_size = scaled_sizes[i]
            segmentation = segmentations[i]

            segmentation = sem_seg_postprocess(segmentation, scaled_size, image_size[0], image_size[1])
            segmentation = segmentation.argmax(dim=0)

            results.append(
                {
                    "segmentation": segmentation,
                    "sampled_labels": sampled_labels[i] if sampled_labels is not None else None,
                }
            )

        return results

    def _instance_genseg_postprocess(
        outputs,
        image_sizes,
        scaled_sizes,
        threshold,
        return_coco_annotation: Optional[bool] = False,
        return_binary_maps: Optional[bool] = False,
        **kwargs,
    ):
        if return_coco_annotation and return_binary_maps:
            raise ValueError("return_coco_annotation and return_binary_maps can not be both set to True.")

        # [batch_size, num_queries, num_classes+1]
        class_queries_logits = outputs.class_queries_logits
        # [batch_size, num_queries, height, width]
        masks_queries_logits = outputs.masks_queries_logits

        device = masks_queries_logits.device
        batch_size = class_queries_logits.shape[0]
        num_classes = class_queries_logits.shape[-1] - 1
        num_queries = class_queries_logits.shape[-2]
        sampled_labels = kwargs.get("sampled_labels", None)

        # Loop over items in batch size
        results: List[Dict[str, TensorType]] = []

        for i in range(batch_size):
            mask_pred = masks_queries_logits[i]
            mask_cls = class_queries_logits[i]
            image_size = image_sizes[i]
            scaled_size = scaled_sizes[i]

            mask_pred = sem_seg_postprocess(mask_pred, scaled_size, image_size[0], image_size[1])

            scores = F.softmax(mask_cls, dim=-1)[:, :-1]
            labels = torch.arange(num_classes, device=device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)

            scores_per_image, topk_indices = scores.flatten(0, 1).topk(num_queries, sorted=False)
            labels_per_image = labels[topk_indices]

            topk_indices = torch.div(topk_indices, num_classes, rounding_mode="floor")
            mask_pred = mask_pred[topk_indices]
            pred_masks = (mask_pred > 0).float()

            # Calculate average mask prob
            mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * pred_masks.flatten(1)).sum(1) / (
                pred_masks.flatten(1).sum(1) + 1e-6
            )
            pred_scores = scores_per_image * mask_scores_per_image
            pred_classes = labels_per_image

            segmentation = torch.zeros((image_size[0], image_size[1])) - 1

            instance_maps, segments = [], []
            current_segment_id = 0
            for j in range(num_queries):
                score = pred_scores[j].item()

                if not torch.all(pred_masks[j] == 0) and score >= threshold:
                    segmentation[pred_masks[j] == 1] = current_segment_id
                    segments.append(
                        {
                            "id": current_segment_id,
                            "label_id": pred_classes[j].item(),
                            "was_fused": False,
                            "score": round(score, 6),
                        }
                    )
                    current_segment_id += 1
                    instance_maps.append(pred_masks[j])

            # Return segmentation map in run-length encoding (RLE) format
            if return_coco_annotation:
                segmentation = convert_segmentation_to_rle(segmentation)

            # Return a concatenated tensor of binary instances maps
            if return_binary_maps and len(instance_maps) != 0:
                segmentation = torch.stack(instance_maps, dim=0)

            # Return the instances for d2
            keep = pred_scores >= threshold
            instances = Instances(image_size)
            instances.pred_masks = pred_masks[keep]
            instances.scores = pred_scores[keep]
            instances.pred_classes = pred_classes[keep]
            instances.pred_boxes = BitMasks(pred_masks[keep]).get_bounding_boxes()

            results.append(
                {
                    "segmentation": segmentation,
                    "segments_info": segments,
                    "instances": instances,
                    "sampled_labels": sampled_labels[i] if sampled_labels is not None else None,
                }
            )
        return results

    def _remap_prompt_labels_to_contiguous(pred_labels, sampled_label, metadata):
        """Map prompt-local class indices to dataset contiguous ids via sampled_labels.

        For full-catalog eval this is identity. For eval_pos_cat_only prompts it is required
        so PQ / stuff fusion still use global contiguous indices.
        """
        if sampled_label is None or metadata is None:
            return pred_labels
        d2c = getattr(metadata, "dataset_id_to_contiguous_id", None) or {}
        if not d2c:
            return pred_labels
        if isinstance(sampled_label, torch.Tensor):
            sampled_label = sampled_label.tolist()
        mapped = []
        for lab in pred_labels.tolist():
            lab = int(lab)
            if lab < 0 or lab >= len(sampled_label):
                mapped.append(lab)
                continue
            ds_id = int(sampled_label[lab])
            mapped.append(int(d2c.get(ds_id, lab)))
        return torch.tensor(mapped, device=pred_labels.device, dtype=pred_labels.dtype)

    def _panoptic_genseg_postprocess(
        outputs,
        image_sizes,
        scaled_sizes,
        threshold,
        mask_threshold,
        overlap_mask_area_threshold,
        label_ids_to_fuse,
        **kwargs,
    ):
        def _debug_ovseg_confusion(mask_cls_logits, task_name, metadata):
            global _OVSEG_DEBUG_PRINT_COUNT
            if _OVSEG_DEBUG_PRINT_COUNT >= _OVSEG_DEBUG_MAX_PRINT:
                return
            if os.environ.get("OVSEG_DEBUG_CONFUSION", "0") != "1":
                return
            if "ovseg" not in str(task_name).lower():
                return
            if metadata is None:
                return
            if not hasattr(metadata, "dataset_id_to_contiguous_id") or not hasattr(metadata, "dataset_classes"):
                return

            d2c = metadata.dataset_id_to_contiguous_id or {}
            ds_classes = metadata.dataset_classes or {}
            c2d = {cont_id: ds_id for ds_id, cont_id in d2c.items()}

            def _find_contiguous_id_by_name(target_name):
                target = target_name.strip().lower()
                for ds_id, name in ds_classes.items():
                    if str(name).strip().lower() == target:
                        return d2c.get(ds_id, None)
                return None

            lv_cid = _find_contiguous_id_by_name("large vehicle")
            bf_cid = _find_contiguous_id_by_name("baseball field")
            if lv_cid is None or bf_cid is None:
                return

            probs = F.softmax(mask_cls_logits, dim=-1)[:, :-1]  # [num_queries, num_classes]
            pred_scores, pred_labels = probs.max(-1)
            topk_scores, topk_ids = probs.topk(k=min(5, probs.shape[-1]), dim=-1)

            # 优先看被预测成 baseball field 的 query，排查是否 large vehicle 分数被系统性压制
            focus_indices = (pred_labels == bf_cid).nonzero(as_tuple=False).flatten().tolist()
            if len(focus_indices) == 0:
                focus_indices = torch.topk(pred_scores, k=min(3, pred_scores.shape[0])).indices.tolist()
            else:
                focus_indices = focus_indices[:3]

            for qi in focus_indices:
                lv_score = probs[qi, lv_cid].item()
                bf_score = probs[qi, bf_cid].item()
                pred_cid = int(pred_labels[qi].item())
                pred_ds = c2d.get(pred_cid, pred_cid)
                pred_name = ds_classes.get(pred_ds, f"category_{pred_ds}")
                rank_items = []
                for s, cid in zip(topk_scores[qi].tolist(), topk_ids[qi].tolist()):
                    ds_id = c2d.get(int(cid), int(cid))
                    cls_name = ds_classes.get(ds_id, f"category_{ds_id}")
                    rank_items.append(f"{cls_name}:{s:.3f}")
                print_log(
                    f"[OVSEG_DEBUG] q={qi} pred={pred_name} "
                    f"baseball_field={bf_score:.3f} large_vehicle={lv_score:.3f} "
                    f"top5={rank_items}",
                    logger="current",
                )
                _OVSEG_DEBUG_PRINT_COUNT += 1
                if _OVSEG_DEBUG_PRINT_COUNT >= _OVSEG_DEBUG_MAX_PRINT:
                    break

        # label_ids_to_fuse is the stuff_class_contiguous_ids
        if label_ids_to_fuse is None:
            print_log("`label_ids_to_fuse` unset. No instance will be fused.", logger="current")
            label_ids_to_fuse = set()

        # [batch_size, num_queries, num_classes+1]
        class_queries_logits = outputs.class_queries_logits
        # [batch_size, num_queries, height, width]
        masks_queries_logits = outputs.masks_queries_logits
        scaled_sizes = scaled_sizes if scaled_sizes is not None else image_sizes
        metadata = kwargs.get("metadata", None)
        sampled_labels = kwargs.get("sampled_labels", None)

        batch_size = class_queries_logits.shape[0]
        num_labels = class_queries_logits.shape[-1] - 1

        # Loop over items in batch size
        results: List[Dict[str, TensorType]] = []

        for i in range(batch_size):
            mask_pred = masks_queries_logits[i]
            mask_cls = class_queries_logits[i]
            image_size = image_sizes[i]
            scaled_size = scaled_sizes[i]

            _debug_ovseg_confusion(mask_cls, kwargs.get("task_name", ""), metadata)

            mask_pred = sem_seg_postprocess(mask_pred, scaled_size, image_size[0], image_size[1])

            mask_prob = mask_pred.sigmoid()
            # the last class is __background__
            scores = F.softmax(mask_cls, dim=-1)[:, :-1]
            pred_score, pred_label = scores.max(-1)

            mask_probs_item, pred_scores_item, pred_labels_item = remove_low_and_no_objects(
                mask_prob, pred_score, pred_label, threshold, num_labels
            )

            # No mask found
            if mask_probs_item.shape[0] <= 0:
                height, width = image_size if image_sizes is not None else mask_probs_item.shape[1:]
                # Official evaluation script uses 0 for VOID label.
                segmentation = torch.zeros((height, width), device=mask_pred.device, dtype=torch.long)
                results.append({"segmentation": segmentation, "segments_info": []})
                continue

            sampled_label_i = None
            if sampled_labels is not None and i < len(sampled_labels):
                sampled_label_i = sampled_labels[i]
            pred_labels_item = _remap_prompt_labels_to_contiguous(
                pred_labels_item, sampled_label_i, metadata
            )

            # Get segmentation map and segment information of batch item
            target_size = image_size if image_sizes is not None else None
            segmentation, segments_info = compute_segments(
                mask_probs=mask_probs_item,
                pred_scores=pred_scores_item,
                pred_labels=pred_labels_item,
                mask_threshold=mask_threshold,
                overlap_mask_area_threshold=overlap_mask_area_threshold,
                label_ids_to_fuse=label_ids_to_fuse,
                target_size=target_size,
            )

            results.append(
                {
                    "segmentation": segmentation,
                    "segments_info": segments_info,
                    "sampled_labels": sampled_label_i,
                }
            )

        return results

    if "pan" in task_name:
        metadata = kwargs.get("metadata", None)
        label_ids_to_fuse = None
        if metadata is not None and hasattr(metadata, "stuff_dataset_id_to_contiguous_id"):
            label_ids_to_fuse = metadata.stuff_dataset_id_to_contiguous_id.values()
        return _panoptic_genseg_postprocess(
            outputs,
            image_sizes,
            scaled_sizes,
            threshold,
            mask_threshold,
            overlap_mask_area_threshold,
            label_ids_to_fuse,
            **kwargs,
        )
    elif "sem" in task_name:
        sampled_labels = kwargs.pop("sampled_labels", None)
        return _semantic_genseg_postprocess(outputs, image_sizes, scaled_sizes, sampled_labels=sampled_labels)
    elif "ins" in task_name:
        return_coco_annotation = kwargs.pop("return_coco_annotation", True)
        return_binary_maps = kwargs.pop("return_binary_maps", False)
        return _instance_genseg_postprocess(
            outputs,
            image_sizes,
            scaled_sizes,
            threshold,
            return_coco_annotation,
            return_binary_maps,
            **kwargs,
        )
    elif "det" in task_name:
        return_coco_annotation = kwargs.pop("return_coco_annotation", True)
        return_binary_maps = kwargs.pop("return_binary_maps", False)
        nms_threshold = kwargs.pop("nms_threshold", 0.5)
        # detection 默认至少 0.05 分，避免 threshold=0 时保留大量 0% 重复框
        score_thr = threshold if threshold > 0 else kwargs.pop("min_score", 0.05)
        results = _instance_genseg_postprocess(
            outputs,
            image_sizes,
            scaled_sizes,
            0.0,
            return_coco_annotation,
            return_binary_maps,
            **kwargs,
        )
        for result in results:
            if result.get("instances") is not None:
                result["instances"] = _filter_detection_instances(
                    result["instances"], score_thr=score_thr, nms_thr=nms_threshold
                )
        return results
    else:
        raise ValueError(f"Task name {task_name} not supported")
