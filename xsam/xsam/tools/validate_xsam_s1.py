#!/usr/bin/env python

import argparse
import json
import os
import os.path as osp
import sys
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# 添加项目根目录到Python路径
current_dir = osp.dirname(osp.abspath(__file__))
project_root = osp.dirname(osp.dirname(osp.dirname(current_dir)))
sys.path.insert(0, project_root)

# 打印调试信息
print(f"Current directory: {current_dir}")
print(f"Project root: {project_root}")
print(f"Expected inits path: {osp.join(project_root, 'inits')}")
print(f"Expected inits/sam-vit-large path: {osp.join(project_root, 'inits', 'sam-vit-large')}")

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from mmengine.config import Config
from mmengine.registry import MODELS, DATASETS
from mmengine.dataset import DefaultSampler
from mmengine.utils import ProgressBar
from mmengine.logging import MMLogger, print_log

# 现在应该可以正确导入了
try:
    from xsam.dataset.collate_fns import xsam_collate_fn
    from xsam.evaluation.evaluators import GenericSegEvaluator
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    print(f"Project root: {project_root}")
    raise

warnings.filterwarnings("ignore")


class XSAMStage1Validator:
    """X-SAM第一阶段验证器 - 基于mmengine实现"""
    
    def __init__(self, config_path: str, checkpoint_path: str, output_dir: str):
        """
        初始化验证器
        
        Args:
            config_path: 配置文件路径
            checkpoint_path: 检查点文件路径
            output_dir: 输出目录路径
        """
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.output_dir = output_dir
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化日志记录器
        self.logger = MMLogger.get_current_instance()
        
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
        
        self.logger.info(f"X-SAM Stage1 Validator initialized successfully")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Output directory: {self.output_dir}")
        
    def _load_config(self) -> Config:
        """加载配置文件"""
        if not osp.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        config = Config.fromfile(self.config_path)
        self.logger.info(f"Loaded config from: {self.config_path}")
        return config
    
    def _build_model(self):
        """构建模型"""
        if 'model' not in self.config:
            raise KeyError("Model configuration not found in config")
        
        # 直接导入和构建模型，不使用mmengine注册表
        try:
            from xsam.model import XSamModel
            from xsam.model.segmentors import XSegmentor
            from xsam.model.segmentors.sam import SamModel
            from xsam.model.segmentors.mask2former import Mask2FormerModel, Mask2FormerConfig
            from xsam.dataset.process_fns import generic_seg_postprocess_fn
            
            model_config = self.config.model
            
            # 构建分割器
            segmentor_config = model_config.segmentor
            
            # 构建编码器
            encoder = SamModel.from_pretrained(
                segmentor_config.encoder.pretrained_model_name_or_path,
                trust_remote_code=segmentor_config.encoder.trust_remote_code,
                torch_dtype=segmentor_config.encoder.torch_dtype
            )
            
            # 构建解码器
            decoder_config = segmentor_config.decoder.config
            decoder = Mask2FormerModel._from_config(
                Mask2FormerConfig.from_pretrained(
                    decoder_config.pretrained_model_name_or_path,
                    use_backbone=decoder_config.use_backbone,
                    feature_channels=decoder_config.feature_channels,
                    num_queries=decoder_config.num_queries,
                    num_transformer_enc_layers=decoder_config.num_transformer_enc_layers,
                    num_transformer_dec_layers=decoder_config.num_transformer_dec_layers,
                    num_feature_levels=decoder_config.num_feature_levels,
                    enforce_input_proj=decoder_config.enforce_input_proj,
                    mask_predictor_hidden_dim=decoder_config.mask_predictor_hidden_dim,
                    num_classes=decoder_config.num_classes
                )
            )
            
            # 构建XSegmentor
            segmentor = XSegmentor(
                encoder=encoder,
                decoder=decoder
            )
            
            # 构建XSamModel
            model = XSamModel(
                freeze_segmentor_encoder=model_config.freeze_segmentor_encoder,
                use_activation_checkpointing=model_config.use_activation_checkpointing,
                postprocess_fn=generic_seg_postprocess_fn,
                connector_type=model_config.connector_type,
                seg_select_layers=model_config.seg_select_layers,
                connector_hidden_dim=model_config.connector_hidden_dim,
                connector_scale_factor=model_config.connector_scale_factor,
                segmentor=segmentor
            )
            
            self.logger.info(f"Built model: {type(model).__name__}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to build model: {e}")
            raise RuntimeError(f"Model building failed: {e}")
    
    def _load_checkpoint(self):
        """加载检查点"""
        if not self.checkpoint_path or not osp.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_path}")
        
        self.logger.info(f"Loading checkpoint from: {self.checkpoint_path}")
        
        # 加载检查点
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        
        # 处理不同的检查点格式
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        
        # 只加载分割器相关的权重
        segmentor_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("segmentor."):
                segmentor_state_dict[key] = value
        
        if not segmentor_state_dict:
            # 如果没有找到segmentor前缀的权重，尝试直接加载
            self.logger.warning("No segmentor weights found, trying to load all weights")
            segmentor_state_dict = state_dict
        
        # 加载权重
        missing_keys, unexpected_keys = self.model.load_state_dict(
            segmentor_state_dict, strict=False
        )
        
        self.logger.info(f"Loaded {len(segmentor_state_dict)} weights")
        if missing_keys:
            self.logger.warning(f"Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            self.logger.warning(f"Unexpected keys: {len(unexpected_keys)}")
    
    def _build_dataloader(self):
        """构建验证数据加载器"""
        if 'val_dataloader' not in self.config:
            raise KeyError("Validation dataloader configuration not found in config")
        
        # 构建数据集
        dataset = DATASETS.build(self.config.val_dataloader.dataset)
        
        # 直接构建数据加载器，不使用注册表
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.config.val_dataloader.batch_size,
            num_workers=self.config.val_dataloader.num_workers,
            sampler=DefaultSampler(dataset, shuffle=False),
            collate_fn=xsam_collate_fn,
            persistent_workers=self.config.val_dataloader.get('persistent_workers', True),
            pin_memory=False
        )
        
        self.logger.info(f"Built dataloader with {len(dataset)} samples")
        return dataloader
    
    def _build_evaluator(self):
        """构建评估器"""
        if 'val_evaluator' not in self.config:
            # 如果没有配置评估器，使用默认的GenericSegEvaluator
            self.logger.warning("No evaluator configured, using default GenericSegEvaluator")
            evaluator = GenericSegEvaluator(
                data_name="panoptic_genseg",
                output_dir=osp.join(self.output_dir, "evaluation")
            )
        else:
            # 直接构建评估器，不使用注册表
            evaluator_config = self.config.val_evaluator
            if evaluator_config['type'] == 'GenericSegEvaluator':
                evaluator = GenericSegEvaluator(
                    data_name=evaluator_config.get('data_name', 'panoptic_genseg'),
                    output_dir=evaluator_config.get('output_dir', osp.join(self.output_dir, "evaluation")),
                    distributed=evaluator_config.get('distributed', False),
                    show_categories=evaluator_config.get('show_categories', True)
                )
            else:
                # 对于其他类型的评估器，尝试动态导入
                try:
                    evaluator_class = eval(evaluator_config['type'])
                    evaluator = evaluator_class(**{k: v for k, v in evaluator_config.items() if k != 'type'})
                except Exception as e:
                    self.logger.warning(f"Failed to build evaluator {evaluator_config['type']}, using default: {e}")
                    evaluator = GenericSegEvaluator(
                        data_name="panoptic_genseg",
                        output_dir=osp.join(self.output_dir, "evaluation")
                    )
        
        self.logger.info(f"Built evaluator: {type(evaluator).__name__}")
        return evaluator
    
    def validate(self) -> Dict:
        """执行验证"""
        self.model.eval()
        
        self.logger.info("Starting validation...")
        
        results = []
        predictions = []
        
        # 使用mmengine的ProgressBar
        progress_bar = ProgressBar(len(self.val_dataloader))
        
        with torch.no_grad():
            for batch_idx, data_batch in enumerate(self.val_dataloader):
                # 将数据移到设备上
                data_batch = self._move_to_device(data_batch)
                
                try:
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
                    
                except Exception as e:
                    self.logger.error(f"Error processing batch {batch_idx}: {e}")
                    continue
                
                progress_bar.update()
        
        # 计算评估指标
        self.logger.info("Computing evaluation metrics...")
        metrics = self.evaluator.process(results)
        final_metrics = self.evaluator.evaluate(len(results))
        
        # 保存所有结果
        self._save_all_results(final_metrics, predictions, results)
        
        self.logger.info(f"Validation completed. Results saved to {self.output_dir}")
        
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
            result = {}
            
            # 处理全景分割结果
            if hasattr(output, 'pred_panoptic_seg'):
                result["pred_panoptic_seg"] = output.pred_panoptic_seg.cpu().numpy()
            
            # 处理语义分割结果
            if hasattr(output, 'pred_sem_seg'):
                result["pred_sem_seg"] = output.pred_sem_seg.cpu().numpy()
            
            # 处理实例分割结果
            if hasattr(output, 'pred_instances'):
                result["pred_instances"] = output.pred_instances.cpu().numpy()
            
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
                "timestamp": datetime.now().isoformat()
            }
            
            # 保存全景分割预测
            if hasattr(output, 'pred_panoptic_seg'):
                pred_info["pred_panoptic_seg"] = output.pred_panoptic_seg.cpu().numpy().tolist()
            
            # 保存语义分割预测
            if hasattr(output, 'pred_sem_seg'):
                pred_info["pred_sem_seg"] = output.pred_sem_seg.cpu().numpy().tolist()
            
            # 保存实例分割预测
            if hasattr(output, 'pred_instances'):
                pred_info["pred_instances"] = output.pred_instances.cpu().numpy().tolist()
            
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
                self._save_mask_as_image(
                    panoptic_mask, 
                    vis_dir, 
                    f"batch_{batch_idx:04d}_sample_{i:04d}_panoptic.png"
                )
            
            # 保存语义分割结果
            if hasattr(output, 'pred_sem_seg'):
                sem_mask = output.pred_sem_seg.cpu().numpy()
                self._save_mask_as_image(
                    sem_mask, 
                    vis_dir, 
                    f"batch_{batch_idx:04d}_sample_{i:04d}_semantic.png"
                )
            
            # 保存实例分割结果
            if hasattr(output, 'pred_instances'):
                inst_mask = output.pred_instances.cpu().numpy()
                self._save_mask_as_image(
                    inst_mask, 
                    vis_dir, 
                    f"batch_{batch_idx:04d}_sample_{i:04d}_instances.png"
                )
    
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
            self.logger.error(f"Failed to save mask {filename}: {e}")
    
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
        
        self.logger.info(f"All results saved to {self.output_dir}")
    
    def _save_evaluation_report(self, metrics):
        """保存评估报告"""
        report_file = osp.join(self.output_dir, "evaluation_report.txt")
        
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("X-SAM Stage1 Validation Report\n")
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
            f.write("Report generated by X-SAM Stage1 Validator\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Validate X-SAM Stage1 Model")
    parser.add_argument("config", help="配置文件路径")
    parser.add_argument("--checkpoint", required=True, help="检查点文件路径")
    parser.add_argument("--output-dir", required=True, help="输出目录路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    # 设置随机种子
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print_log(f"设置随机种子为: {args.seed}")
    
    # 检查文件存在性
    if not osp.exists(args.config):
        raise FileNotFoundError(f"配置文件未找到: {args.config}")
    
    if not osp.exists(args.checkpoint):
        raise FileNotFoundError(f"检查点文件未找到: {args.checkpoint}")
    
    # 创建验证器
    validator = XSAMStage1Validator(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir
    )
    
    # 执行验证
    try:
        validation_results = validator.validate()
        
        # 打印主要指标
        print_log("\n" + "="*50)
        print_log("验证完成！")
        print_log("="*50)
        
        print_log("主要指标:")
        for metric_name, metric_value in validation_results["metrics"].items():
            if isinstance(metric_value, (int, float)):
                print_log(f"  {metric_name}: {metric_value:.4f}")
            else:
                print_log(f"  {metric_name}: {metric_value}")
        
        print_log(f"\n所有结果已保存到: {args.output_dir}")
        
    except Exception as e:
        print_log(f"验证失败，错误信息: {e}")
        raise


if __name__ == "__main__":
    main() 