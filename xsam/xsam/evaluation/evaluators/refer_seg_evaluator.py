import itertools
import json
import logging
import os
import os.path as osp
import traceback
from typing import List, Optional

import numpy as np

from xsam.utils.logging import print_log

from ...dataset.utils.catalog import MetadataCatalog
from ...dataset.utils.mask import calculate_iou, decode_mask, encode_mask
from ..utils import comm
from ..utils.iou import IouStat
from .base_seg_evaluator import BaseSegEvaluator


class ReferSegEvaluator(BaseSegEvaluator):

    def __init__(
        self,
        data_name: str = "refseg",
        cat_names: Optional[List[str]] = ["ignore", "refer"],
        output_dir: Optional[str] = None,
        distributed: bool = True,
    ):
        """
        Args:
            metadata: metadata of the dataset
            output_dir: output directory to save results for evaluation.
        """
        self._distributed = distributed
        self._data_name = data_name
        self._metadata = MetadataCatalog.get(data_name)
        self._output_dir = output_dir
        self.iou_stat = IouStat(cat_names=cat_names)

        if self._output_dir is not None:
            os.makedirs(self._output_dir, exist_ok=True)

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        self._metadata = value

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
        self._predictions = []

    # follow segmentation evaluation
    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            pred_mask, segments_info = (
                output["segmentation"],
                output["segments_info"],
            )
            pred_mask = pred_mask.cpu().numpy()
            pred_mask[pred_mask == self._metadata.ignore_label] = 0
            pred_mask = pred_mask.astype(np.uint8)
            file_name = os.path.basename(input["file_name"])
            self._predictions.append(
                {
                    "image_id": input["image_id"],
                    # Some datasets may miss sample_id in image_info; use 0 to keep key stable.
                    "sample_id": input.get("sample_id", 0),
                    "file_name": file_name,
                    "pred_mask": encode_mask(pred_mask),
                    "segments_info": segments_info,
                }
            )

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        print_log(f"Evaluating {self.data_name} with {len(predictions)} predictions...", logger="current")
        if len(predictions) == 0:
            logging.warning(f"{self.__class__.__name__} did not receive valid predictions.")
            return {}

        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)
            file_path = os.path.join(self._output_dir, "predictions.json")
            print_log(f"Writing {self.data_name} predictions to {self._output_dir}...", logger="current")
            with open(file_path, "w") as f:
                json.dump(predictions, f)
            print_log(f"Predictions saved to: {file_path}", logger="current")

        gt_json = osp.realpath(self._metadata.gt_json)
        try:
            results = self._eval_predictions(predictions, gt_json)
            return results
        except Exception as e:
            error_msg = (
                f"Failed to evaluate {self.data_name}. "
                f"gt_json={gt_json}, num_predictions={len(predictions)}. "
                f"Error: {e}"
            )
            print_log(error_msg, logger="current", level="ERROR")
            if self._output_dir:
                error_file = os.path.join(self._output_dir, "evaluation_error.json")
                with open(error_file, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "data_name": self.data_name,
                            "error": str(e),
                            "traceback": traceback.format_exc(),
                            "gt_json": gt_json,
                            "num_predictions": len(predictions),
                        },
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )
                print_log(f"Evaluation error details saved to: {error_file}", logger="current")
            return {"task": "refseg", "data_name": self.data_name, "error": str(e)}

    def _eval_predictions(self, predictions, gt_json):
        with open(gt_json, "r") as f:
            gt_anns = json.load(f)

        id2ann_map = {
            f"{data['image_id']}{data['image_info'].get('sample_id', 0)}": data["annotations"] for data in gt_anns
        }
        matched_cnt = 0
        missing_gt_cnt = 0

        for pred in predictions:
            image_id = pred["image_id"]
            sample_id = pred.get("sample_id", 0)
            pred_mask = pred["pred_mask"]
            height, width = pred_mask["size"]
            pred_mask = decode_mask(pred_mask, height, width)

            # segmentation is polygon
            gt_key = f"{image_id}{sample_id}"
            if gt_key not in id2ann_map:
                missing_gt_cnt += 1
                continue
            gt_mask = id2ann_map[gt_key][0]["segmentation"]
            gt_mask = decode_mask(gt_mask, height, width)

            intersection, union, _ = calculate_iou(pred_mask, gt_mask, 2, self._metadata.ignore_label)
            self.iou_stat.update(intersection, union, n=1)
            matched_cnt += 1

        if missing_gt_cnt > 0:
            print_log(
                f"{self.data_name}: {missing_gt_cnt} predictions cannot find matching GT by (image_id, sample_id).",
                logger="current",
                level="WARNING",
            )
        if matched_cnt == 0:
            raise ValueError(
                f"No valid matched prediction/gt pairs. predictions={len(predictions)}, gt_items={len(id2ann_map)}"
            )

        self.iou_stat.average()
        print_log(f"{self.data_name} evaluation results:\n{self.iou_stat}", logger="current")
        
        # 返回评估结果
        results = {
            "task": "refseg",
            "data_name": self._data_name,
            "num_samples": len(predictions),
            "ciou": self.iou_stat.ciou.tolist() if hasattr(self.iou_stat.ciou, 'tolist') else self.iou_stat.ciou,
            "giou": self.iou_stat.giou.tolist() if hasattr(self.iou_stat.giou, 'tolist') else self.iou_stat.giou,
        }

        # 保存评估结果到文件
        res = {}
        for i, cat_name in enumerate(self.iou_stat.cat_names):
            res[cat_name] = {
                "cIoU": float(self.iou_stat.ciou[i]),
                "gIoU": float(self.iou_stat.giou[i]),
                **{f"Pr@{t:.1f}": float(self.iou_stat.pr[i, j]) for j, t in enumerate(self.iou_stat.thresholds)},
            }
        if self._output_dir:
            print_log(f"Writing {self.data_name} evaluation results to {self._output_dir}...", logger="current")
            results_file = os.path.join(self._output_dir, "summary.json")
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump({"summary": res}, f, indent=2, ensure_ascii=False)
            print_log(f"Evaluation results saved to: {results_file}", logger="current")

        return results
