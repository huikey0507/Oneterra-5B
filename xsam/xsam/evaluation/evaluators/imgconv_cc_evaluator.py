import itertools
import json
import os
import os.path as osp
from typing import Optional, Dict, List, Tuple

from xsam.utils.logging import print_log

from ..utils import comm
from .base_seg_evaluator import BaseSegEvaluator

# -------------------------
# COCO-style metrics via pycocoevalcap
# -------------------------
from .eval_complex_comprehension import evaluation_metrics_CC

class ImgConvCCEvaluator(BaseSegEvaluator):
    """评估器用于 imgconv（complex comprehension）任务。
    """

    def __init__(
        self,
        data_name: str = "imgconv_cc",
        output_dir: Optional[str] = None,
        distributed: bool = True,
        metrics: Optional[List[str]] = None,
    ):
        self._data_name = data_name
        self._distributed = distributed
        self._output_dir = output_dir
        self._metadata = None

        if self._output_dir is not None:
            os.makedirs(self._output_dir, exist_ok=True)

        self.reset()

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
        self._references = []
        self._questions = []
        self._image_files = []
        self._task_categories = []

    def _extract_question_ground_truth(self, val_input):
        if not isinstance(val_input, dict):
            return None, None
        conversation = val_input.get("conversation", [])
        if not conversation:
            return None, None
        gt = conversation[-1].get("output", "")
        question = conversation[-1].get("input", "")
        return (question or "").strip(), (gt or "").strip()

    def process(self, val_inputs, llm_outputs):
        if isinstance(val_inputs, list) and len(val_inputs) > 0:
            for val_input, llm_output in zip(val_inputs, llm_outputs):
                question, gt_answer = self._extract_question_ground_truth(val_input)
                image_file = val_input.get("image_file", "")
                task_category = val_input.get("task_category", "")

                self._references.append(gt_answer if gt_answer else "")
                self._predictions.append((llm_output or ""))
                self._questions.append(question or "")
                self._image_files.append(image_file or "")
                self._task_categories.append(task_category or "")


    def _to_float(self, x) -> float:
        try:
            return float(x)
        except Exception:
            return 0.0

    # =========================
    # Evaluate
    # =========================
    def evaluate(self):

        # --------- 1) 分布式收集 ---------
        if self._distributed:
            comm.synchronize()

            local_items = []
            n = min(
                len(getattr(self, "_predictions", [])),
                len(getattr(self, "_references", [])),
                len(getattr(self, "_questions", [])),
                len(getattr(self, "_image_files", [])),
                len(getattr(self, "_task_categories", [])),
            )
            for i in range(n):
                local_items.append({
                    "image_file": self._image_files[i],
                    "question": self._questions[i],
                    "prediction": self._predictions[i],
                    "reference": self._references[i],
                    "task_category": self._task_categories[i],
                })

            gathered = comm.gather(local_items, dst=0)
            if not comm.is_main_process():
                return {}

            merged_items = list(itertools.chain(*gathered)) if gathered is not None else []
        else:
            merged_items = []
            n = min(
                len(getattr(self, "_predictions", [])),
                len(getattr(self, "_references", [])),
                len(getattr(self, "_questions", [])),
                len(getattr(self, "_image_files", [])),
                len(getattr(self, "_task_categories", [])),
            )
            for i in range(n):
                merged_items.append({
                    "image_file": self._image_files[i],
                    "question": self._questions[i],
                    "prediction": self._predictions[i],
                    "reference": self._references[i],
                    "task_category": self._task_categories[i],
                })

        # --------- 2) 空预测保护 ---------
        if len(merged_items) == 0:
            print_log("Warning: No predictions to evaluate", logger="current")
            return {"task": "imgconv", "data_name": self._data_name, "status": "no_predictions"}

        # --------- 3) 组织输出 JSON（全量）---------
        predictions_json = []
        for sample_id, item in enumerate(merged_items):
            pred = (item.get("prediction") or "").strip()
            ref = (item.get("reference") or "").strip()
            question = (item.get("question") or "").strip()
            image_file = item.get("image_file") or ""
            task_category = item.get("task_category") or ""
            predictions_json.append({
                "sample_id": sample_id,
                "image_file": image_file,
                "task_category": task_category,
                "question": question,
                "prediction": pred,
                "reference": ref,
            })


        results = {
            "task": "imgconv_cc",
            "data_name": self._data_name,
            "num_samples": len(merged_items),
        }

        # --------- 4) 评估 ---------
        eval_results = evaluation_metrics_CC(merged_items)
        results.update(eval_results)
        
        # --------- 8) 打印 ---------
        print_log(f"\n{'='*80}", logger="current")
        print_log(f"ImgConv Complex Comprehension Evaluation Results for {self._data_name}", logger="current")
        print_log(f"{'='*80}", logger="current")
        print_log(f"Number of samples: {results['num_samples']}", logger="current")
        print_log(f"{'='*80}\n", logger="current")
        print_log(f"Overall Metrics:", logger="current")
        for metric_name, metric_value in eval_results.items():
            if isinstance(metric_value, float):
                print_log(f"  {metric_name}: {metric_value:.4f}", logger="current")
            else:
                print_log(f"  {metric_name}: {metric_value}", logger="current")
        print_log(f"{'='*80}\n", logger="current")

        # --------- 9) 落盘 ---------
        if self._output_dir is not None:
            os.makedirs(self._output_dir, exist_ok=True)

            predictions_file = osp.join(self._output_dir, "predictions.json")
            print_log(f"Writing {self._data_name} predictions to {self._output_dir}...", logger="current")
            with open(predictions_file, "w", encoding="utf-8") as f:
                json.dump(predictions_json, f, indent=2, ensure_ascii=False)
            print_log(f"Predictions saved to: {predictions_file}", logger="current")

            results_file = osp.join(self._output_dir, "imgconv_evaluation_results.json")
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump({"summary": results}, f, indent=2, ensure_ascii=False)
            print_log(f"evaluation results saved to: {results_file}", logger="current")

        return results
