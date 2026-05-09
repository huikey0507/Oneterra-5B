#!/usr/bin/env python

import argparse
import json
import os
import os.path as osp
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from xsam.dataset.collate_fns import xsam_collate_fn
from xsam.evaluation.evaluators import GenericSegEvaluator
from xsam.utils.logging import print_log, set_default_logging_format
from xsam.utils.utils import register_function

set_default_logging_format()
warnings.filterwarnings("ignore")


class SegmentorValidator:
    """X-SAM分割器验证器"""
    
    def __init__(self, config_path: str, checkpoint_path: str, output_dir: str):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 加载配置和模型
        self.config = self._load_config()
        self.model = self._build_model()
        self.model.to(self.device)
        
        # 加载检查点
        self._load_checkpoint()
        
        # 构建数据加载器
        self.val_dataloader = self._build_dataloader()
        
        # 构建评估器
        self.evaluator = self._build_evaluator()
        
        # 初始化结果存储
        self.validation_results = {
            "config": self.config_path,
            "checkpoint": self.checkpoint_path,
            "timestamp": datetime.now().isoformat(),
            "device": str(self.device),
            "metrics": {},
            "predictions": [],
            "visualizations": []
        }
        
    def _load_config(self):
        """加载配置文件"""
        from mmengine.config import Config
        return Config.fromfile(self.config_path)
    
    def _build_model(self):
        """构建模型"""
        from xsam.registry import MODELS
        model = MODELS.build(self.config.model)
        return model
    
    def _load_checkpoint(self):
        """加载检查点"""
        if self.checkpoint_path and osp.exists(self.checkpoint_path):
            print_log(f"Loading checkpoint from {self.checkpoint_path}", logger="current")
            
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            
            # 只加载分割器相关的权重
            segmentor_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("segmentor."):
                    segmentor_state_dict[key] = value
            
            missing_keys, unexpected_keys = self.model.load_state_dict(
                segmentor_state_dict, strict=False
            )
            
            print_log(f"Loaded {len(segmentor_state_dict)} segmentor weights", logger="current")
            if missing_keys:
                print_log(f"Missing keys: {len(missing_keys)}", logger="current")
            if unexpected_keys:
                print_log(f"Unexpected keys: {len(unexpected_keys)}", logger="current")
        else:
            print_log("No checkpoint provided, using random weights", logger="current")
    
    def _build_dataloader(self):
        """构建验证数据加载器"""
        from xsam.registry import DATASETS, DATALOADERS
        from mmengine.dataset import DefaultSampler
        
        # 构建数据集
        dataset = DATASETS.build(self.config.val_dataloader.dataset)
        
        # 构建数据加载器
        dataloader = DATALOADERS.build(
            self.config.val_dataloader,
            dataset=dataset,
            sampler=DefaultSampler(dataset, shuffle=False),
            collate_fn=xsam_collate_fn,
        )
        
        print_log(f"Built dataloader with {len(dataset)} samples", logger="current")
        return dataloader
    
    def _build_evaluator(self):
        """构建评估器"""
        from xsam.registry import EVALUATORS
        
        evaluator = EVALUATORS.build(self.config.val_evaluator)
        return evaluator
    
    def validate(self) -> Dict:
        """执行验证"""
        self.model.eval()
        
        print_log("Starting validation...", logger="current")
        
        results = []
        predictions = []
        
        with torch.no_grad():
            for batch_idx, data_batch in enumerate(tqdm(self.val_dataloader, desc="Validating")):
                # 将数据移到设备上
                data_batch = self._move_to_device(data_batch)
                
                # 前向传播
                outputs = self.model(
                    data_batch,
                    mode="predict",
                    do_postprocess=True
                )
                
                # 处理输出
                batch_results = self._process_outputs(outputs, data_batch)
                results.extend(batch_results)
                
                # 保存预测结果
                batch_predictions = self._save_predictions(outputs, data_batch, batch_idx)
                predictions.extend(batch_predictions)
                
                # 保存可视化结果
                self._save_visualizations(outputs, data_batch, batch_idx)
        
        # 计算评估指标
        print_log("Computing evaluation metrics...", logger="current")
        metrics = self.evaluator.process(results)
        final_metrics = self.evaluator.evaluate(len(results))
        
        # 保存所有结果
        self._save_all_results(final_metrics, predictions, results)
        
        print_log(f"Validation completed. Results saved to {self.output_dir}", logger="current")
        
        return {
            "metrics": final_metrics,
            "results": results,
            "predictions": predictions
        }
    
    def _move_to_device(self, data_batch):
        """将数据批次移到设备上"""
        if isinstance(data_batch, dict):
            return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                   for k, v in data_batch.items()}
        elif isinstance(data_batch, list):
            return [self._move_to_device(item) for item in data_batch]
        else:
            return data_batch.to(self.device) if isinstance(data_batch, torch.Tensor) else data_batch
    
    def _process_outputs(self, outputs, data_batch):
        """处理模型输出"""
        results = []
        
        # 处理每个样本的输出
        for i, output in enumerate(outputs):
            result = {
                "pred_panoptic_seg": output.pred_panoptic_seg.cpu().numpy(),
                "pred_sem_seg": output.pred_sem_seg.cpu().numpy() if hasattr(output, 'pred_sem_seg') else None,
                "pred_instances": output.pred_instances.cpu().numpy() if hasattr(output, 'pred_instances') else None,
            }
            
            # 添加真实标签（如果存在）
            if "gt_panoptic_seg" in data_batch:
                result["gt_panoptic_seg"] = data_batch["gt_panoptic_seg"][i].cpu().numpy()
            if "gt_sem_seg" in data_batch:
                result["gt_sem_seg"] = data_batch["gt_sem_seg"][i].cpu().numpy()
            if "gt_instances" in data_batch:
                result["gt_instances"] = data_batch["gt_instances"][i].cpu().numpy()
                
            results.append(result)
        
        return results
    
    def _save_predictions(self, outputs, data_batch, batch_idx):
        """保存预测结果"""
        predictions = []
        
        for i, output in enumerate(outputs):
            pred_info = {
                "batch_idx": batch_idx,
                "sample_idx": i,
                "pred_panoptic_seg": output.pred_panoptic_seg.cpu().numpy().tolist(),
                "pred_sem_seg": output.pred_sem_seg.cpu().numpy().tolist() if hasattr(output, 'pred_sem_seg') else None,
                "pred_instances": output.pred_instances.cpu().numpy().tolist() if hasattr(output, 'pred_instances') else None,
            }
            predictions.append(pred_info)
        
        return predictions
    
    def _save_visualizations(self, outputs, data_batch, batch_idx):
        """保存可视化结果"""
        vis_dir = osp.join(self.output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        for i, output in enumerate(outputs):
            # 保存全景分割结果
            if hasattr(output, 'pred_panoptic_seg'):
                panoptic_mask = output.pred_panoptic_seg.cpu().numpy()
                self._save_mask_as_image(panoptic_mask, vis_dir, f"batch_{batch_idx}_sample_{i}_panoptic.png")
            
            # 保存语义分割结果
            if hasattr(output, 'pred_sem_seg'):
                sem_mask = output.pred_sem_seg.cpu().numpy()
                self._save_mask_as_image(sem_mask, vis_dir, f"batch_{batch_idx}_sample_{i}_semantic.png")
            
            # 保存实例分割结果
            if hasattr(output, 'pred_instances'):
                inst_mask = output.pred_instances.cpu().numpy()
                self._save_mask_as_image(inst_mask, vis_dir, f"batch_{batch_idx}_sample_{i}_instances.png")
    
    def _save_mask_as_image(self, mask, save_dir, filename):
        """将掩码保存为图像"""
        try:
            # 归一化到0-255
            if mask.max() > 0:
                mask_normalized = ((mask - mask.min()) / (mask.max() - mask.min()) * 255).astype(np.uint8)
            else:
                mask_normalized = mask.astype(np.uint8)
            
            # 转换为PIL图像并保存
            mask_image = Image.fromarray(mask_normalized)
            save_path = osp.join(save_dir, filename)
            mask_image.save(save_path)
            
            # 记录保存的文件
            self.validation_results["visualizations"].append({
                "file": filename,
                "path": save_path,
                "shape": mask.shape,
                "dtype": str(mask.dtype)
            })
            
        except Exception as e:
            print_log(f"Failed to save mask {filename}: {e}", logger="current")
    
    def _save_all_results(self, metrics, predictions, results):
        """保存所有验证结果"""
        
        # 1. 保存评估指标
        metrics_file = osp.join(self.output_dir, "evaluation_metrics.json")
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        # 2. 保存预测结果
        predictions_file = osp.join(self.output_dir, "predictions.json")
        with open(predictions_file, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        
        # 3. 保存完整验证结果
        self.validation_results["metrics"] = metrics
        self.validation_results["predictions"] = predictions
        
        complete_results_file = osp.join(self.output_dir, "complete_validation_results.json")
        with open(complete_results_file, "w", encoding="utf-8") as f:
            json.dump(self.validation_results, f, indent=2, ensure_ascii=False)
        
        # 4. 保存评估报告
        self._save_evaluation_report(metrics)
        
        # 5. 保存配置信息
        config_file = osp.join(self.output_dir, "validation_config.json")
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump({
                "config_path": self.config_path,
                "checkpoint_path": self.checkpoint_path,
                "output_dir": self.output_dir,
                "timestamp": datetime.now().isoformat(),
                "device": str(self.device)
            }, f, indent=2, ensure_ascii=False)
        
        print_log(f"All results saved to {self.output_dir}", logger="current")
    
    def _save_evaluation_report(self, metrics):
        """保存评估报告"""
        report_file = osp.join(self.output_dir, "evaluation_report.txt")
        
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("X-SAM Segmentor Validation Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Checkpoint: {self.checkpoint_path}\n")
            f.write(f"Device: {self.device}\n\n")
            
            f.write("Evaluation Metrics:\n")
            f.write("-" * 20 + "\n")
            
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    f.write(f"{metric_name}: {metric_value:.4f}\n")
                else:
                    f.write(f"{metric_name}: {metric_value}\n")
            
            f.write("\n" + "=" * 50 + "\n")
            f.write("Report generated by X-SAM Segmentor Validator\n")


def main():
    parser = argparse.ArgumentParser(description="Validate X-SAM Segmentor and Save Results")
    parser.add_argument("config", help="config file path")
    parser.add_argument("--checkpoint", required=True, help="path to checkpoint for validation")
    parser.add_argument("--output-dir", required=True, help="directory to save validation results")
    parser.add_argument("--save-predictions", action="store_true", help="save prediction masks")
    parser.add_argument("--save-visualizations", action="store_true", help="save visualization images")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    
    args = parser.parse_args()
    
    # 设置随机种子
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print_log(f"Set random seed to {args.seed}", logger="current")
    
    # 检查文件存在性
    if not osp.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    if not osp.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
    
    # 创建验证器
    validator = SegmentorValidator(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir
    )
    
    # 执行验证
    try:
        validation_results = validator.validate()
        
        # 打印主要指标
        print_log("\n" + "="*50, logger="current")
        print_log("VALIDATION COMPLETED SUCCESSFULLY!", logger="current")
        print_log("="*50, logger="current")
        
        print_log("Main Metrics:", logger="current")
        for metric_name, metric_value in validation_results["metrics"].items():
            if isinstance(metric_value, (int, float)):
                print_log(f"  {metric_name}: {metric_value:.4f}", logger="current")
            else:
                print_log(f"  {metric_name}: {metric_value}", logger="current")
        
        print_log(f"\nAll results saved to: {args.output_dir}", logger="current")
        
    except Exception as e:
        print_log(f"Validation failed with error: {e}", logger="current")
        raise


if __name__ == "__main__":
    main()