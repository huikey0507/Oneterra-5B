from collections import defaultdict
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
try:
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.spice.spice import Spice

    COCOEVAL_AVAILABLE = True
except Exception as e:
    COCOEVAL_AVAILABLE = False
    PTBTokenizer = None
    Bleu = Meteor = Rouge = Cider = Spice = None
    print_log(f"Warning: pycocoevalcap not available ({e}). Non-accuracy metrics will be skipped.", logger="current")


class ImgConvEvaluator(BaseSegEvaluator):
    """评估器用于 imgconv（图像对话/VQA）任务。

    支持指标：
      - accuracy（本地 exact match）
      - bleu1/bleu2/bleu3/bleu4（pycocoevalcap）
      - rougeL（pycocoevalcap；不再支持 rouge1/rouge2）
      - meteor（pycocoevalcap）
      - cider（pycocoevalcap）
      - spice（pycocoevalcap，可选）
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

        # COCO tokenizer
        self._coco_tokenizer = PTBTokenizer() if COCOEVAL_AVAILABLE else None

        self._metrics = self._validate_metrics(metrics)

        self.reset()

    def _validate_metrics(self, metrics: Optional[List[str]]) -> List[str]:
        """
        - 兼容：
          * bleu == bleu4
          * rouge == rougeL
        - rouge1/rouge2 不再计算：若出现则忽略
        """
        if metrics is None:
            # 默认加上 spice 也可以；如果你不想默认算 spice，可删掉 "spice"
            metrics = ["accuracy", "bleu4", "rougeL", "meteor", "cider"]

        metrics = [m.strip() for m in list(metrics)]
        metrics = [m.lower() for m in metrics if m]

        # aliases
        metrics = ["bleu4" if m == "bleu" else m for m in metrics]
        metrics = ["rougel" if m == "rouge" else m for m in metrics]  # rouge -> rougeL

        # 禁用 rouge1/rouge2
        filtered = []
        dropped_rouge12 = []
        for m in metrics:
            if m in ("rouge1", "rouge2"):
                dropped_rouge12.append(m)
                continue
            filtered.append(m)
        if dropped_rouge12:
            print_log(
                f"Warning: {dropped_rouge12} are deprecated/disabled; only rougeL is supported now. They will be ignored.",
                logger="current",
            )
        metrics = filtered

        valid = {
            "accuracy",
            "bleu1", "bleu2", "bleu3", "bleu4",
            "rougel",   # will normalize to rougeL
            "meteor",
            "cider",
            "spice",
        }
        unknown = [m for m in metrics if m not in valid]
        if unknown:
            raise ValueError(f"Unknown metrics: {unknown}. Valid metrics: {sorted(list(valid))}")

        # normalize rougeL naming
        normalized = []
        for m in metrics:
            if m == "rougel":
                normalized.append("rougeL")
            else:
                normalized.append(m)
        metrics = normalized

        # 如果 pycocoevalcap 不可用，则只能计算 accuracy
        if not COCOEVAL_AVAILABLE:
            if any(m != "accuracy" for m in metrics):
                print_log(
                    "Warning: pycocoevalcap not available; non-accuracy metrics will be skipped.",
                    logger="current",
                )
            metrics = ["accuracy"]

        # 去重保持顺序
        seen = set()
        out = []
        for m in metrics:
            if m not in seen:
                out.append(m)
                seen.add(m)
        return out

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
    # COCO scorers builder
    # =========================
    def _build_coco_scorers(self, metrics: List[str]):
        """
        返回 list[(scorer, method_names)]
          - Bleu(4) -> ["bleu1","bleu2","bleu3","bleu4"]
          - Rouge() -> "rougeL"
          - Meteor() -> "meteor"
          - Cider() -> "cider"
          - Spice() -> "spice"
        """
        scorers = []

        need_bleu = any(m.startswith("bleu") for m in metrics)
        if need_bleu:
            scorers.append((Bleu(4), ["bleu1", "bleu2", "bleu3", "bleu4"]))

        if "rougeL" in metrics:
            scorers.append((Rouge(), "rougeL"))

        if "meteor" in metrics:
            scorers.append((Meteor(), "meteor"))

        if "cider" in metrics:
            scorers.append((Cider(), "cider"))

        if "spice" in metrics:
            scorers.append((Spice(), "spice"))

        return scorers

    def _to_float(self, x) -> float:
        try:
            return float(x)
        except Exception:
            return 0.0

    def _spice_item_to_float(self, item) -> float:
        """
        pycocoevalcap 的 SPICE per-image scores 在不同 fork 可能是：
        - float
        - dict（含 All->f 或 score->All->f 等）
        这里做一个稳健解析。
        """
        if item is None:
            return 0.0
        if isinstance(item, (int, float)):
            return float(item)
        if isinstance(item, dict):
            # 常见：{"All": {"f": 0.123, ...}, ...}
            if "All" in item:
                allv = item.get("All")
                if isinstance(allv, dict) and "f" in allv:
                    return self._to_float(allv.get("f"))
                if isinstance(allv, (int, float)):
                    return float(allv)
            # 可能：{"scores": {"All": {"f": ...}}}
            if "scores" in item and isinstance(item["scores"], dict):
                sc = item["scores"]
                if "All" in sc and isinstance(sc["All"], dict) and "f" in sc["All"]:
                    return self._to_float(sc["All"].get("f"))
        return 0.0

    # =========================
    # Evaluate
    # =========================
    def evaluate(self):
        metrics = getattr(self, "_metrics", ["accuracy", "bleu4", "rougeL", "meteor", "cider"])

        need_acc = "accuracy" in metrics
        need_bleu_ns = sorted({int(m[-1]) for m in metrics if m.startswith("bleu")})
        need_any_bleu = len(need_bleu_ns) > 0
        need_rougeL = "rougeL" in metrics
        need_meteor = "meteor" in metrics
        need_cider = "cider" in metrics
        need_spice = "spice" in metrics

        need_any_coco = any([
            need_any_bleu,
            need_rougeL,
            need_meteor,
            need_cider,
            need_spice,
        ]) and COCOEVAL_AVAILABLE

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
            entry = {
                "sample_id": sample_id,
                "image_file": image_file,
                "question": question,
                "prediction": pred,
                "reference": ref,
            }
            if task_category != "":
                entry["task_category"] = task_category
            predictions_json.append(entry)

        # --------- 4) 过滤：只对有 GT 的样本打分（避免 COCO scorer 报错）---------
        valid_pairs = []  # (orig_sample_id, item)
        for sample_id, item in enumerate(merged_items):
            ref = (item.get("reference") or "").strip()
            if ref:
                valid_pairs.append((sample_id, item))

        num_with_gt = len(valid_pairs)

        results = {
            "task": "imgconv",
            "data_name": self._data_name,
            "metrics": metrics,
            "num_samples": len(merged_items),
            "num_with_gt": num_with_gt,
        }

        detailed_scores = []

        if num_with_gt == 0:
            print_log("Warning: No ground truth answers found, cannot calculate metrics", logger="current")
            # 仍可落盘 predictions + 空 summary
            if self._output_dir is not None:
                os.makedirs(self._output_dir, exist_ok=True)
                predictions_file = osp.join(self._output_dir, "predictions.json")
                with open(predictions_file, "w", encoding="utf-8") as f:
                    json.dump(predictions_json, f, indent=2, ensure_ascii=False)

                results_file = osp.join(self._output_dir, "imgconv_evaluation_results.json")
                with open(results_file, "w", encoding="utf-8") as f:
                    json.dump({"summary": results, "detailed_scores": detailed_scores}, f, indent=2, ensure_ascii=False)
            return results

        # --------- 5) accuracy：逐样本计算 ---------
        correct = 0
        base_detail = []
        correct_group = defaultdict(int)
        grout_cnt = defaultdict(int)
        for local_id, (orig_sample_id, item) in enumerate(valid_pairs):
            pred = (item.get("prediction") or "").strip()
            ref = (item.get("reference") or "").strip()
            question = (item.get("question") or "").strip()
            image_file = item.get("image_file") or ""

            d = {
                "sample_id": orig_sample_id,  # 保持原 sample_id
                "image_file": image_file,
                "question": question,
                "prediction": pred,
                "reference": ref,
            }
            task_category = item.get("task_category") or ""
            if task_category != "":
                d["task_category"] = task_category
                
            if need_acc:
                em = 1 if self._normalize_answer(pred) == self._normalize_answer(ref) else 0
                correct += em
                d["exact_match"] = em
                if task_category != "":
                    correct_group[task_category] += em
                    grout_cnt[task_category] += 1

            base_detail.append(d)

        if need_acc:
            results["accuracy"] = correct / num_with_gt
            if len(correct_group) > 0:
                for category, c in correct_group.items():
                    results[f"accuracy_{category}"] = c / grout_cnt[category]

        # --------- 6) COCO metrics：一次性调用 pycocoevalcap ---------
        per_sample_buf: Dict[str, List[float]] = {}  # metric_name -> per-sample list aligned with valid_pairs

        if need_any_coco:
            # COCO expects:
            # gts[id] = [{'caption': ref}, ...]
            # res[id] = [{'caption': pred}]
            gts = {}
            res = {}
            for local_id, (orig_sample_id, item) in enumerate(valid_pairs):
                pred = (item.get("prediction") or "").strip()
                ref = (item.get("reference") or "").strip()
                gts[local_id] = [{"caption": ref}]
                res[local_id] = [{"caption": pred}]

            # tokenize
            try:
                gts_tok = self._coco_tokenizer.tokenize(gts)
                res_tok = self._coco_tokenizer.tokenize(res)
            except Exception as e:
                print_log(f"Warning: COCO tokenizer failed: {e}. Non-accuracy metrics will be set to 0.", logger="current")
                gts_tok, res_tok = None, None

            if gts_tok is not None and res_tok is not None:
                coco_scorers = self._build_coco_scorers(metrics)

                for scorer, method in coco_scorers:
                    try:
                        score, scores = scorer.compute_score(gts_tok, res_tok)
                    except Exception as e:
                        print_log(f"Warning: scorer {scorer.__class__.__name__} failed: {e}", logger="current")
                        # 失败则置 0
                        if isinstance(method, list):
                            for m in method:
                                per_sample_buf[m] = [0.0] * num_with_gt
                        else:
                            per_sample_buf[method] = [0.0] * num_with_gt
                        continue

                    # BLEU：method 为 list
                    if isinstance(method, list):
                        # score: [bleu1..bleu4] (corpus-level)
                        # scores: [[per-sample bleu1], [per-sample bleu2], ...]
                        for m_name, m_score, m_scores in zip(method, score, scores):
                            # 只写入用户请求的 bleuN
                            if m_name in metrics:
                                results[m_name] = self._to_float(m_score)
                            # per-sample 保存全部，后面按需写入 detailed
                            per_sample_buf[m_name] = [self._to_float(x) for x in m_scores]
                    else:
                        # Rouge/Meteor/Cider/Spice：method 为 str
                        m_name = method
                        if m_name == "spice":
                            # corpus score
                            if "spice" in metrics:
                                results["spice"] = self._to_float(score)
                            # per-sample might be dicts or floats
                            per_sample_buf["spice"] = [self._spice_item_to_float(x) for x in scores]
                        else:
                            if m_name in metrics:
                                results[m_name] = self._to_float(score)
                            per_sample_buf[m_name] = [self._to_float(x) for x in scores]

                # BLEU：用户只要 bleu4/bleu3 等时，上面已按需写入 results
                # ROUGE-L：pycocoevalcap Rouge 返回 key "ROUGE_L" 通常对应 method "rougeL"（这里我们用 "rougeL" 作为内部 key）
                # 为了对齐你原打印与 keys，这里将 rougeL 的结果统一映射到 "rougeL"
                # 上面 method 已是 "rougeL"，直接用即可
            else:
                # tokenizer 失败：置 0
                if need_any_bleu:
                    for n_bleu in need_bleu_ns:
                        results[f"bleu{n_bleu}"] = 0.0
                        per_sample_buf[f"bleu{n_bleu}"] = [0.0] * num_with_gt
                if need_rougeL:
                    results["rougeL"] = 0.0
                    per_sample_buf["rougeL"] = [0.0] * num_with_gt
                if need_meteor:
                    results["meteor"] = 0.0
                    per_sample_buf["meteor"] = [0.0] * num_with_gt
                if need_cider:
                    results["cider"] = 0.0
                    per_sample_buf["cider"] = [0.0] * num_with_gt
                if need_spice:
                    results["spice"] = 0.0
                    per_sample_buf["spice"] = [0.0] * num_with_gt

        # --------- 7) 组装 detailed_scores（与 valid_pairs 对齐）---------
        for local_id, d in enumerate(base_detail):
            # BLEU：只填 requested 的 bleuN
            if need_any_bleu:
                for n_bleu in need_bleu_ns:
                    k = f"bleu{n_bleu}"
                    if k in per_sample_buf:
                        d[k] = per_sample_buf[k][local_id]

            if need_rougeL:
                # 注意：我们的 metrics key 统一用 "rougeL"
                if "rougeL" in per_sample_buf:
                    d["rougeL"] = per_sample_buf["rougeL"][local_id]

            if need_meteor:
                if "meteor" in per_sample_buf:
                    d["meteor"] = per_sample_buf["meteor"][local_id]

            if need_cider:
                if "cider" in per_sample_buf:
                    d["cider"] = per_sample_buf["cider"][local_id]

            if need_spice:
                if "spice" in per_sample_buf:
                    d["spice"] = per_sample_buf["spice"][local_id]

            detailed_scores.append(d)

        # --------- 8) 打印 ---------
        print_log(f"\n{'='*80}", logger="current")
        print_log(f"ImgConv Evaluation Results for {self._data_name}", logger="current")
        print_log(f"Metrics: {metrics}", logger="current")
        print_log(f"{'='*80}", logger="current")
        print_log(f"Number of samples: {results['num_samples']}", logger="current")
        print_log(f"Number with ground truth: {results['num_with_gt']}", logger="current")

        if need_acc:
            print_log(f"Accuracy: {results.get('accuracy', 0.0):.4f}", logger="current")
        if need_any_bleu:
            for n_bleu in need_bleu_ns:
                key = f"bleu{n_bleu}"
                if key in results:
                    print_log(f"BLEU-{n_bleu}: {results[key]:.4f}", logger="current")
        if need_rougeL and "rougeL" in results:
            print_log(f"ROUGE-L: {results['rougeL']:.4f}", logger="current")
        if need_meteor and "meteor" in results:
            print_log(f"METEOR: {results['meteor']:.4f}", logger="current")
        if need_cider and "cider" in results:
            print_log(f"CIDEr: {results['cider']:.4f}", logger="current")
        if need_spice and "spice" in results:
            print_log(f"SPICE: {results['spice']:.4f}", logger="current")

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
                json.dump({"summary": results, "detailed_scores": detailed_scores}, f, indent=2, ensure_ascii=False)
            print_log(f"Detailed evaluation results saved to: {results_file}", logger="current")

        return results
