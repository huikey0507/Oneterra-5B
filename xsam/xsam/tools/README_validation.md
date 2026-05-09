# X-SAM 第一阶段验证脚本使用说明

## 概述

本验证脚本用于验证X-SAM第一阶段的训练结果，基于mmengine实现，支持全景分割、语义分割和实例分割的评估。

## 文件结构

```
xsam/xsam/tools/
├── validate_xsam_s1.py          # 主验证脚本
└── README_validation.md         # 使用说明

xsam/xsam/configs/xsam/phi3_mini_4k_instruct_siglip2_so400m_p14_384/s1_seg_finetune/
└── validate_s1.py               # 验证配置文件
```

## 使用方法

### 1. 基本命令

```bash
python xsam/xsam/tools/validate_xsam_s1.py \
    configs/xsam/phi3_mini_4k_instruct_siglip2_so400m_p14_384/s1_seg_finetune/validate_s1.py \
    --checkpoint /path/to/your/checkpoint.pth \
    --output-dir /path/to/validation/results
```

### 2. 参数说明

- `config`: 验证配置文件路径（必需）
- `--checkpoint`: 训练好的检查点文件路径（必需）
- `--output-dir`: 验证结果输出目录（必需）
- `--seed`: 随机种子，默认为42

### 3. 环境变量设置

在运行前，请确保设置以下环境变量：

```bash
export CODE_DIR="./xsam/"
export DATA_DIR="./xsam_data/"
export INIT_DIR="./inits/"
```

## 验证流程

### 1. 初始化阶段
- 加载配置文件
- 构建模型架构
- 加载训练好的权重
- 构建验证数据加载器
- 初始化评估器

### 2. 验证阶段
- 在验证集上运行推理
- 处理模型输出（全景分割、语义分割、实例分割）
- 保存预测结果
- 生成可视化图像

### 3. 评估阶段
- 计算评估指标
- 生成评估报告
- 保存所有结果

## 输出结果

验证完成后，会在指定的输出目录中生成以下文件：

```
output_dir/
├── evaluation_metrics.json          # 评估指标（JSON格式）
├── predictions.json                 # 预测结果（JSON格式）
├── complete_validation_results.json # 完整验证结果
├── evaluation_report.txt            # 评估报告（文本格式）
├── validation_config.json           # 验证配置信息
└── visualizations/                  # 可视化结果目录
    ├── batch_0000_sample_0000_panoptic.png
    ├── batch_0000_sample_0000_semantic.png
    └── batch_0000_sample_0000_instances.png
```

## 评估指标

脚本会计算以下评估指标：

### 全景分割指标
- **PQ (Panoptic Quality)**: 全景质量
- **SQ (Segmentation Quality)**: 分割质量
- **RQ (Recognition Quality)**: 识别质量

### 语义分割指标
- **mIoU**: 平均交并比
- **Pixel Accuracy**: 像素准确率

### 实例分割指标
- **AP**: 平均精度
- **AP50**: 50% IoU阈值下的平均精度
- **AP75**: 75% IoU阈值下的平均精度

## 注意事项

### 1. 数据准备
- 确保验证集数据路径正确
- 验证集标注文件格式需要与训练时一致
- 图像和标签文件需要一一对应

### 2. 模型权重
- 检查点文件需要包含训练好的分割器权重
- 权重键名通常以"segmentor."开头
- 如果权重键名不匹配，脚本会尝试直接加载

### 3. 内存管理
- 验证时使用较小的批次大小（batch_size=1）
- 如果GPU内存不足，可以进一步减少批次大小
- 考虑使用CPU进行验证（虽然速度较慢）

### 4. 错误处理
- 脚本包含完整的错误处理机制
- 如果某个批次处理失败，会记录错误并继续处理其他批次
- 所有错误都会记录在日志中

## 故障排除

### 1. 配置文件错误
```
KeyError: Model configuration not found in config
```
**解决方案**: 检查配置文件中的model字段是否正确配置

### 2. 检查点加载失败
```
FileNotFoundError: Checkpoint file not found
```
**解决方案**: 检查检查点文件路径是否正确

### 3. 数据集加载失败
```
KeyError: Validation dataloader configuration not found in config
```
**解决方案**: 检查配置文件中的val_dataloader字段

### 4. 内存不足
```
CUDA out of memory
```
**解决方案**: 减少batch_size或使用CPU进行验证

## 示例输出

验证成功后的示例输出：

```
==================================================
验证完成！
==================================================
主要指标:
  PQ: 0.6543
  SQ: 0.7892
  RQ: 0.8234
  mIoU: 0.7123
  Pixel Accuracy: 0.8567

所有结果已保存到: /path/to/validation/results
```

## 扩展功能

### 1. 自定义评估指标
可以在配置文件中添加自定义的评估器配置

### 2. 批量验证
可以修改脚本支持多个检查点的批量验证

### 3. 结果分析
可以添加结果分析和可视化功能

## 技术支持

如果遇到问题，请检查：
1. 环境变量设置
2. 文件路径配置
3. 模型权重格式
4. 数据集格式

更多信息请参考X-SAM官方文档。 