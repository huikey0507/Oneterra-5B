import contextlib
import io
import itertools
import json
import os
import os.path as osp
import tempfile
from typing import Optional

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_util
from tabulate import tabulate

from xsam.utils.logging import print_log

from ...dataset.utils.catalog import MetadataCatalog
from ...dataset.utils.coco import COCO
from ..utils import comm
from ..utils.map import convert_to_coco_json, create_small_table, derive_coco_results, evaluate_predictions_on_coco, instances_to_coco_json
from ..utils.pq import pq_compute, print_panoptic_results
from ..utils.rbox_eval import build_rbox_eval_inputs, detect_gt_bbox_type
from .base_seg_evaluator import BaseSegEvaluator
from .eval_map import eval_rbbox_map_coco_metrics


class GenericSegEvaluator(BaseSegEvaluator):
    def __init__(
        self,
        data_name: str = "panoptic_genseg",
        output_dir: Optional[str] = None,
        distributed: bool = True,
        show_categories: bool = False,
    ):
        """
        Args:
            metadata: metadata of the dataset
            output_dir: output directory to save results for evaluation.
        """
        self._data_name = data_name
        self._distributed = distributed
        self._metadata = MetadataCatalog.get(data_name)
        self._output_dir = output_dir
        self._show_categories = show_categories
        self._cpu_device = torch.device("cpu")
        if self._output_dir is not None:
            os.makedirs(self._output_dir, exist_ok=True)

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        self._metadata = value
        self._dataset_name = self.data_name
        self._num_classes = len(self._metadata.dataset_id_to_contiguous_id)
        self._contiguous_id_to_dataset_id = {v: k for k, v in self._metadata.dataset_id_to_contiguous_id.items()}
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            self._thing_contiguous_id_to_dataset_id = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
        if hasattr(self._metadata, "stuff_dataset_id_to_contiguous_id"):
            self._stuff_contiguous_id_to_dataset_id = {
                v: k for k, v in self._metadata.stuff_dataset_id_to_contiguous_id.items()
            }

    @property
    def output_dir(self):
        return self._output_dir

    @output_dir.setter
    def output_dir(self, value):
        self._output_dir = value
        if self._output_dir is not None:
            os.makedirs(self._output_dir, exist_ok=True)

    @property
    def data_name(self):
        return self._data_name

    def reset(self):
        self._conf_matrix = np.zeros((self._num_classes + 1, self._num_classes + 1), dtype=np.int64)
        self._predictions = []

    def _convert_category_id(self, segment_info):
        isthing = segment_info.pop("isthing", None)
        if isthing is None:
            # the model produces panoptic category id directly. No more conversion needed
            return segment_info
        if isthing is True:
            segment_info["category_id"] = self._thing_contiguous_id_to_dataset_id[segment_info["category_id"]]
        else:
            segment_info["category_id"] = self._stuff_contiguous_id_to_dataset_id[segment_info["category_id"]]
        return segment_info

    def _encode_json_sem_seg(self, sem_seg, input_file_name):
        """
        Convert semantic segmentation to COCO stuff format with segments encoded as RLEs.
        See http://cocodataset.org/#format-results
        """
        json_list = []
        for label in np.unique(sem_seg):
            if self._contiguous_id_to_dataset_id is not None:
                assert (
                    label in self._contiguous_id_to_dataset_id
                ), "Label {} is not in the metadata info for {}".format(label, self._dataset_name)
                dataset_id = self._contiguous_id_to_dataset_id[label]
            else:
                dataset_id = int(label)
            mask = (sem_seg == label).astype(np.uint8)
            mask_rle = mask_util.encode(np.array(mask[:, :, None], order="F"))[0]
            mask_rle["counts"] = mask_rle["counts"].decode("utf-8")
            json_list.append({"file_name": input_file_name, "category_id": dataset_id, "segmentation": mask_rle})
        return json_list

    def semantic_process(self, inputs, outputs):
        """Semantic mIoU：pred/GT 统一到 dataset contiguous（0..N-1）再进混淆矩阵。

        - pred：后处理 argmax 为 prompt 下标；若有 sampled_labels，先映到 dataset_id 再映 contiguous。
        - GT：seg_labels 约定为原始 category_id（可为空洞编号，如 pano 的 0=background）；
          经 dataset_id_to_contiguous_id 映到 contiguous。未知 id / ignore_label → ignore 槽。
        """
        gt_semseg_folder = osp.realpath(self._metadata.semseg_map_folder)
        semseg_sufix = self._metadata.semseg_sufix if hasattr(self._metadata, "semseg_sufix") else ".png"
        label_shift = self._metadata.label_shift if hasattr(self._metadata, "label_shift") else 0
        d2c = {int(k): int(v) for k, v in (self._metadata.dataset_id_to_contiguous_id or {}).items()}
        ignore_label = self._metadata.ignore_label

        for input, output in zip(inputs, outputs):
            segmentation = output["segmentation"].to(self._cpu_device)
            sampled_labels = output.get("sampled_labels", None)
            pred = np.array(segmentation, dtype=np.int64)

            # prompt-contiguous → global contiguous（勿把 dataset_id 直接当矩阵下标）
            if sampled_labels is not None:
                if isinstance(sampled_labels, torch.Tensor):
                    sampled_labels = sampled_labels.tolist()
                remapped = np.full_like(pred, self._num_classes)
                for ul in np.unique(pred).tolist():
                    ul = int(ul)
                    if ul < 0 or ul >= len(sampled_labels):
                        continue
                    ds_id = int(sampled_labels[ul]) - label_shift
                    cont_id = d2c.get(ds_id)
                    if cont_id is not None:
                        remapped[pred == ul] = cont_id
                pred = remapped

            file_name = input["file_name"]
            file_name_semseg = os.path.splitext(file_name)[0] + semseg_sufix
            gt = np.array(Image.open(os.path.join(gt_semseg_folder, file_name_semseg)), dtype=np.int64)

            # dataset category_id（可空洞）→ contiguous；其余先置 ignore
            gt_contig = np.full(gt.shape, self._num_classes, dtype=np.int64)
            for ds_id, cont_id in d2c.items():
                gt_contig[gt == ds_id] = cont_id
            if ignore_label is not None:
                gt_contig[gt == ignore_label] = self._num_classes

            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt_contig.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

            self._predictions.extend(self._encode_json_sem_seg(pred, input["file_name"]))

    def instance_process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(instances, input["image_id"])
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            if len(prediction) > 1:
                self._predictions.append(prediction)

    def detection_process(self, inputs, outputs):
        """Collect bbox predictions for pure object detection evaluation."""
        self.instance_process(inputs, outputs)

    def _remap_coco_detection_results(self, predictions):
        dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
        all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
        num_classes = len(all_contiguous_ids)
        assert min(all_contiguous_ids) == 0 and max(all_contiguous_ids) == num_classes - 1

        reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        new_coco_results = []
        dropped_invalid_category = 0
        for result in coco_results:
            category_id = result["category_id"]
            if category_id not in reverse_id_mapping:
                dropped_invalid_category += 1
                continue
            result = dict(result)
            result["category_id"] = reverse_id_mapping[category_id]
            new_coco_results.append(result)

        if dropped_invalid_category > 0:
            print_log(
                f"{self.data_name}: dropped {dropped_invalid_category} detection predictions with "
                f"category_id not in thing contiguous mapping [0, {num_classes - 1}].",
                logger="current",
            )
        if len(coco_results) > 0 and len(new_coco_results) == 0:
            print_log(
                f"{self.data_name}: all {len(coco_results)} detection predictions were filtered before eval; "
                "check postprocess class indices vs metadata.thing_dataset_id_to_contiguous_id.",
                logger="current",
            )
        return new_coco_results, num_classes, dataset_id_to_contiguous_id

    def _thing_class_names(self):
        thing_classes = self._metadata.get("thing_classes")
        if thing_classes is None:
            return None
        if isinstance(thing_classes, dict):
            return [thing_classes[k] for k in sorted(thing_classes.keys())]
        return list(thing_classes)

    def detection_evaluate(self, predictions):
        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)
            file_path = os.path.join(self._output_dir, "predictions.json")
            print_log(f"Writing {self.data_name} predictions to {self._output_dir}...", logger="current")
            with open(file_path, "w") as f:
                json.dump(predictions, f)

        with open(osp.realpath(self._metadata.gt_json), "r") as f:
            gt_anns = json.load(f)

        thing_contiguous = set(self._metadata.thing_dataset_id_to_contiguous_id.values())
        for gt_item in gt_anns:
            gt_item["annotations"] = [
                ann for ann in gt_item.get("annotations", []) if int(ann.get("category_id", -1)) in thing_contiguous
            ]

        new_coco_results, num_classes, dataset_id_to_contiguous_id = self._remap_coco_detection_results(predictions)
        gt_bbox_type = detect_gt_bbox_type(gt_anns, self._metadata)
        class_names = self._thing_class_names()
        print_log(
            f"{self.data_name}: detected GT bbox type = {gt_bbox_type} "
            f"({'COCO horizontal bbox' if gt_bbox_type == 'hbox' else 'rotated bbox'} eval will be used)",
            logger="current",
        )

        if len(new_coco_results) == 0:
            det_table = create_small_table({"mAP": float("nan"), "AP50": float("nan"), "AP75": float("nan")})
            det_title = (
                "Object Detection (horizontal bbox)"
                if gt_bbox_type == "hbox"
                else "Object Detection (rotated bbox)"
            )
            return f"=== {det_title} ===\n{det_table}"

        if gt_bbox_type == "hbox":
            print_log(f"Trying to convert '{self.data_name}' to COCO format...", logger="current")
            cache_path = osp.join(self._output_dir, f"{self.data_name}_coco_format.json")
            convert_to_coco_json(self.data_name, cache_path, gt_anns, allow_cached=False)
            coco_api = COCO(cache_path)
            bbox_coco_eval = evaluate_predictions_on_coco(coco_api, new_coco_results, "bbox")
            det_table = derive_coco_results(
                bbox_coco_eval, "bbox", class_names=class_names, show_categories=self._show_categories
            )
        else:
            category_id_to_contiguous = {
                dataset_id: contiguous_id for dataset_id, contiguous_id in dataset_id_to_contiguous_id.items()
            }
            predictions_remapped = []
            for gt_item in gt_anns:
                image_id = gt_item["image_id"]
                instances = [r for r in new_coco_results if r["image_id"] == image_id]
                predictions_remapped.append({"image_id": image_id, "instances": instances})
            det_results, rbbox_annotations = build_rbox_eval_inputs(
                gt_anns,
                predictions_remapped,
                num_classes,
                category_id_to_contiguous,
            )
            det_table = eval_rbbox_map_coco_metrics(
                det_results,
                rbbox_annotations,
                dataset=class_names,
                show_categories=self._show_categories,
            )

        det_title = (
            "Object Detection (horizontal bbox)"
            if gt_bbox_type == "hbox"
            else "Object Detection (rotated bbox)"
        )
        return f"=== {det_title} ===\n{det_table}"

    def panoptic_process(self, inputs, outputs):
        from panopticapi.utils import id2rgb

        for input, output in zip(inputs, outputs):
            segmentation, segments_info = (
                output["segmentation"],
                output["segments_info"],
            )
            segmentation = segmentation.to(self._cpu_device)
            segmentation = np.array(segmentation, dtype=int)
            if segments_info is None:
                # If "segments_info" is None, we assume "segmentation" is a
                # H*W int32 image storing the panoptic_id in the format of
                # category_id * label_divisor + instance_id. We reserve -1 for
                # VOID label, and add 1 to segmentation since the official
                # evaluation script uses 0 for VOID label.
                label_divisor = self._metadata.label_divisor
                segments_info = []
                for panoptic_label in np.unique(segmentation):
                    if panoptic_label == -1:
                        # VOID region.
                        continue
                    pred_class = panoptic_label // label_divisor
                    isthing = pred_class in self._metadata.thing_dataset_id_to_contiguous_id.values()
                    segments_info.append(
                        {
                            "id": int(panoptic_label) + 1,
                            "category_id": int(pred_class),
                            "isthing": bool(isthing),
                        }
                    )
                # Official evaluation script uses 0 for VOID label.
                segmentation += 1

            file_name = os.path.basename(input["file_name"])
            file_name_png = os.path.splitext(file_name)[0] + ".png"
            with io.BytesIO() as out:
                Image.fromarray(id2rgb(segmentation)).save(out, format="PNG")
                segments_info = [self._convert_category_id(x) for x in segments_info]
                self._predictions.append(
                    {
                        "image_id": input["image_id"],
                        "file_name": file_name_png,
                        "png_string": out.getvalue(),
                        "segments_info": segments_info,
                    }
                )

    def semantic_evaluate(self, predictions):
        gt_json = osp.realpath(self._metadata.gt_json) if self._metadata.gt_json is not None else None
        if gt_json is not None:
            with tempfile.TemporaryDirectory(prefix="semantic_eval") as pred_dir:
                with open(gt_json, "r") as f:
                    json_data = json.load(f)
                json_data["annotations"] = predictions

                output_dir = self._output_dir or pred_dir
                print_log(f"Writing {self.data_name} predictions to {output_dir}...", logger="current")
                predictions_json = os.path.join(output_dir, "predictions.json")
                with open(predictions_json, "w") as f:
                    json.dump(json_data, f)
        else:
            print_log("Ground truth JSON file is not provided, skipping annotation writing.", logger="current")

        acc = np.full(self._num_classes, np.nan, dtype=np.float32)
        iou = np.full(self._num_classes, np.nan, dtype=np.float32)
        precision = np.full(self._num_classes, np.nan, dtype=np.float32)
        recall = np.full(self._num_classes, np.nan, dtype=np.float32)
        f1 = np.full(self._num_classes, np.nan, dtype=np.float32)
        tp = self._conf_matrix.diagonal()[:-1].astype(np.float32)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(np.float32)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(np.float32)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        recall[acc_valid] = acc[acc_valid]
        precision[acc_valid] = np.where(
            pos_pred[acc_valid] > 0,
            tp[acc_valid] / pos_pred[acc_valid],
            0.0,
        )
        f1[acc_valid] = np.where(
            (precision[acc_valid] + recall[acc_valid]) > 0,
            2 * precision[acc_valid] * recall[acc_valid] / (precision[acc_valid] + recall[acc_valid]),
            0.0,
        )
        union = pos_gt + pos_pred - tp
        iou_valid = np.logical_and(acc_valid, union > 0)
        iou[iou_valid] = tp[iou_valid] / union[iou_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[iou_valid]) / np.sum(iou_valid)
        fiou = np.sum(iou[iou_valid] * class_weights[iou_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)
        mprecision = np.sum(precision[acc_valid]) / np.sum(acc_valid)
        mrecall = macc
        mf1 = np.sum(f1[acc_valid]) / np.sum(acc_valid)

        data = []
        headers = ["Metric", "Value (%)"]
        data.extend(
            [
                ["mIoU", f"{100 * miou:.2f}"],
                ["fwIoU", f"{100 * fiou:.2f}"],
                ["mACC", f"{100 * macc:.2f}"],
                ["pACC", f"{100 * pacc:.2f}"],
                ["mF1", f"{100 * mf1:.2f}"],
                ["mPrecision", f"{100 * mprecision:.2f}"],
                ["mRecall", f"{100 * mrecall:.2f}"],
            ]
        )

        def _format_pct(value):
            return f"{100 * value:.2f}" if np.isfinite(value) else "nan"

        def _class_name(contiguous_id):
            dataset_id = self._contiguous_id_to_dataset_id.get(contiguous_id, contiguous_id)
            if hasattr(self._metadata, "dataset_classes") and self._metadata.dataset_classes:
                return self._metadata.dataset_classes.get(dataset_id, str(dataset_id))
            return str(contiguous_id)

        for i in range(self._num_classes):
            name = _class_name(i)
            data.extend(
                [
                    [f"IoU-{name}", _format_pct(iou[i])],
                    [f"ACC-{name}", _format_pct(acc[i])],
                ]
            )
            if self._show_categories:
                data.extend(
                    [
                        [f"F1-{name}", _format_pct(f1[i])],
                        [f"Precision-{name}", _format_pct(precision[i])],
                        [f"Recall-{name}", _format_pct(recall[i])],
                    ]
                )

        table = tabulate(
            data,
            headers=headers,
            tablefmt="outline",
            floatfmt=".2f",
            stralign="center",
            numalign="center",
        )
        return table

    def instance_evaluate(self, predictions):
        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)
            file_path = os.path.join(self._output_dir, "predictions.json")
            print_log(f"Writing {self.data_name} predictions to {self._output_dir}...", logger="current")
            with open(file_path, "w") as f:
                json.dump(predictions, f)

        with open(osp.realpath(self._metadata.gt_json), "r") as f:
            gt_anns = json.load(f)

        print_log(f"Trying to convert '{self.data_name}' to COCO format...", logger="current")
        cache_path = osp.join(self._output_dir, f"{self.data_name}_coco_format.json")
        convert_to_coco_json(self.data_name, cache_path, gt_anns, allow_cached=False)
        coco_api = COCO(cache_path)

        new_coco_results, _, _ = self._remap_coco_detection_results(predictions)

        coco_eval = (
            evaluate_predictions_on_coco(
                coco_api,
                new_coco_results,
                "segm",
            )
            if len(new_coco_results) > 0
            else None
        )
        segm_table = derive_coco_results(
            coco_eval, "segm", class_names=self._metadata.get("thing_classes"), show_categories=self._show_categories
        )
        return f"=== Instance Segmentation (mask) ===\n{segm_table}"

    def panoptic_evaluate(self, predictions):
        # PanopticApi requires local files
        gt_json = osp.realpath(self._metadata.gt_json)
        gt_panseg_folder = osp.realpath(self._metadata.panseg_map_folder)

        with tempfile.TemporaryDirectory(prefix="panoptic_eval") as pred_dir:
            for p in predictions:
                with open(os.path.join(pred_dir, p["file_name"]), "wb") as f:
                    f.write(p.pop("png_string"))

            with open(gt_json, "r") as f:
                json_data = json.load(f)
            json_data["annotations"] = predictions

            output_dir = self._output_dir or pred_dir
            predictions_json = os.path.join(output_dir, "predictions.json")
            print_log(f"Writing {self.data_name} predictions to {output_dir}...", logger="current")
            with open(predictions_json, "w") as f:
                json.dump(json_data, f)

            with contextlib.redirect_stdout(io.StringIO()):
                pq_res = pq_compute(
                    gt_json,
                    osp.realpath(predictions_json),
                    gt_folder=gt_panseg_folder,
                    pred_folder=pred_dir,
                )

        table = print_panoptic_results(pq_res)
        return table

    def process(self, inputs, outputs):
        if "panoptic" in self.data_name:
            self.panoptic_process(inputs, outputs)
        elif "semantic" in self.data_name:
            self.semantic_process(inputs, outputs)
        elif "detection" in self.data_name:
            self.detection_process(inputs, outputs)
        elif "instance" in self.data_name:
            self.instance_process(inputs, outputs)
        else:
            raise ValueError(f"Unknown dataset name: {self.data_name}")

    def evaluate(self):
        if self._distributed:
            comm.synchronize()

            conf_matrix_list = comm.gather(self._conf_matrix, dst=0)
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}

            self._conf_matrix = np.zeros_like(self._conf_matrix)
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix
        else:
            predictions = self._predictions

        if "panoptic" in self.data_name:
            table = self.panoptic_evaluate(predictions)
        elif "semantic" in self.data_name:
            table = self.semantic_evaluate(predictions)
        elif "detection" in self.data_name:
            table = self.detection_evaluate(predictions)
        elif "instance" in self.data_name:
            table = self.instance_evaluate(predictions)
        else:
            raise ValueError(f"Unknown dataset name: {self.data_name}")

        print_log(f"{self.data_name} evaluation results:\n{table}", logger="current")

        print(type(table))

        with tempfile.TemporaryDirectory(prefix="panoptic_eval") as pred_dir:
            output_dir = self._output_dir or pred_dir
            summary_file = os.path.join(output_dir, "summary.txt")
            with open(summary_file,"w") as f:
                f.write(table)


