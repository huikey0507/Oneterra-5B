from collections import defaultdict
import itertools
import json
import os
import os.path as osp
from typing import Optional, Dict, List, Tuple

from tqdm import tqdm

from xsam.utils.logging import print_log

from ..utils import comm
from .base_seg_evaluator import BaseSegEvaluator


def calculate_precision_recall(gt_class, pred_class):
    gt_rels = set(gt_class)
    pred_rels = set(pred_class)
    # Calculate the number of true positives (tp), false positives (fp), and false negatives (fn)
    tp = len(gt_rels & pred_rels)
    fp = len(pred_rels - gt_rels)
    fn = len(gt_rels - pred_rels)
    # Calculate precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall

def calculate_tpfpfn(gt_class, pred_class):
    gt_rels = set(gt_class)
    pred_rels = set(pred_class)
    # Calculate the number of true positives (tp), false positives (fp), and false negatives (fn)
    tp = len(gt_rels & pred_rels)
    fp = len(pred_rels - gt_rels)
    fn = len(gt_rels - pred_rels)
    return tp, fp, fn

def calculate_PRF1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1



def evaluation_metrics(merged_items):
    correct_single=0
    incorrect_single=0
    count = 0
    tp_total = 0
    fp_total = 0
    fn_total = 0
    for item in tqdm(merged_items):
        question_text = item['question']
        if question_text.endswith("Answer in one word or a short phrase."):
            mode = "single"
        elif question_text.endswith("Answer with all applicable classes separated by commas."):
            mode = "multi"
        
        gt=item['reference'].lower()
        if mode == "single":
            if gt==item['prediction'].lower():
                correct_single=correct_single+1
            else:
                incorrect_single=incorrect_single+1

        elif mode == "multi":
            gt_obj = [label.strip() for label in gt.split(",")]
            answer_obj = [an.strip() for an in item['prediction'].lower().split(",")]
            tp, fp, fn = calculate_tpfpfn(gt_obj, answer_obj)
            tp_total+=tp
            fp_total+=fp
            fn_total+=fn
            count += 1
            
    result_dict = {}
    if (correct_single+incorrect_single)>0:
        scene_acc = correct_single/(correct_single+incorrect_single)
        result_dict['scene_acc'] = scene_acc
    
    precision_total, recall_total, f1_total = calculate_PRF1(tp_total, fp_total, fn_total)
    
    result_dict['obj_precision'] = precision_total
    result_dict['obj_recall'] = recall_total
    result_dict['obj_f1'] = f1_total
    return result_dict


class ImgConvMLSCEvaluator(BaseSegEvaluator):
    """评估器用于 imgconv（multi-label scene classification）任务。

    支持指标：
      - accuracy（本地 exact match）
    """

    def __init__(
        self,
        data_name: str = "imgconv",
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

                self._references.append(gt_answer if gt_answer else "")
                self._predictions.append((llm_output or ""))
                self._questions.append(question or "")
                self._image_files.append(image_file or "")

    # =========================
    # Accuracy
    # =========================
    def _normalize_answer(self, s: str) -> str:
        import re
        s = (s or "").strip().lower()
        s = re.sub(r"[^\w\s]", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

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
                len(getattr(self, "_image_files", [])),            )
            for i in range(n):
                local_items.append({
                    "image_file": self._image_files[i],
                    "question": self._questions[i],
                    "prediction": self._predictions[i],
                    "reference": self._references[i],
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
            )
            for i in range(n):
                merged_items.append({
                    "image_file": self._image_files[i],
                    "question": self._questions[i],
                    "prediction": self._predictions[i],
                    "reference": self._references[i],
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
            entry = {
                "sample_id": sample_id,
                "image_file": image_file,
                "question": question,
                "prediction": pred,
                "reference": ref,
            }
            predictions_json.append(entry)
        
        # --------- 4) 计算指标 ---------
        results = evaluation_metrics(merged_items)
        results['num_samples'] = len(merged_items)

        # --------- 5) 打印 ---------
        print_log(f"\n{'='*80}", logger="current")
        print_log(f"ImgConv Evaluation Results for {self._data_name}", logger="current")
        print_log(f"{'='*80}", logger="current")
        print_log(results, logger="current")
        
        print_log(f"{'='*80}\n", logger="current")

        # --------- 6) 落盘 ---------
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
            print_log(f"Detailed evaluation results saved to: {results_file}", logger="current")

        return results
