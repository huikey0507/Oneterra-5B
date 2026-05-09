import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from PIL import Image
import math

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



def evaluation_metrics(data_path):

    base = [json.loads(q) for q in open(data_path, "r")]
    correct_single=0
    incorrect_single=0
    count = 0
    tp_total = 0
    fp_total = 0
    fn_total = 0
    for answers in tqdm(base):
        question_text = answers['question']
        if question_text.endswith("Answer in one word or a short phrase."):
            mode = "single"
        elif question_text.endswith("Answer with all applicable classes separated by commas."):
            mode = "multi"
        
        gt=answers['ground_truth'].lower()
        if mode == "single":
            if gt==answers['answer'].lower():
                correct_single=correct_single+1
            else:
                incorrect_single=incorrect_single+1

        elif mode == "multi":
            gt_obj = [label.strip() for label in gt.split(",")]
            answer_obj = [an.strip() for an in answers['answer'].lower().split(",")]
            tp, fp, fn = calculate_tpfpfn(gt_obj, answer_obj)
            tp_total+=tp
            fp_total+=fp
            fn_total+=fn
            count += 1
            
    print('correct_scene:',correct_single)
    print('incorrect_scene:',incorrect_single)
    print('Total:',correct_single+incorrect_single)
    if (correct_single+incorrect_single)>0:
        print('Scene Classify Accuracy:',(correct_single/(correct_single+incorrect_single)))

    precision_total, recall_total, f1_total = calculate_PRF1(tp_total, fp_total, fn_total)
    print(f'New Average Precision: {precision_total:.4f}')
    print(f'New Average Recall: {recall_total:.4f}')
    print(f'New F1 score: {f1_total:.4f}')