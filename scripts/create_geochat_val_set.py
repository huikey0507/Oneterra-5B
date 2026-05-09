#!/usr/bin/env python
"""从geochat数据集中抽取200个样本作为验证集"""
import json
import os
import random
import argparse

def create_validation_set(input_json, output_json, num_samples=200, seed=42):
    """
    从geochat数据集中随机抽取指定数量的样本作为验证集
    
    Args:
        input_json: 输入的geochat数据集JSON文件路径
        output_json: 输出的验证集JSON文件路径
        num_samples: 要抽取的样本数量
        seed: 随机种子，确保可重复
    """
    # 设置随机种子
    random.seed(seed)
    
    # 读取原始数据集
    print(f"正在读取数据集: {input_json}")
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_samples = len(data)
    print(f"总样本数: {total_samples}")
    
    if num_samples > total_samples:
        print(f"警告: 请求的样本数 ({num_samples}) 大于总样本数 ({total_samples})，将使用全部样本")
        num_samples = total_samples
    
    # 随机抽取样本
    print(f"正在随机抽取 {num_samples} 个样本...")
    val_samples = random.sample(data, num_samples)
    
    # 保存验证集
    output_dir = os.path.dirname(output_json)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"正在保存验证集到: {output_json}")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(val_samples, f, indent=2, ensure_ascii=False)
    
    print(f"完成! 已创建包含 {len(val_samples)} 个样本的验证集")
    return val_samples


def main():
    parser = argparse.ArgumentParser(description="从geochat数据集中抽取验证集")
    parser.add_argument(
        "--input",
        type=str,
        default="./datas/img_conv_data/geochat/geochat_llava.json",
        help="输入的geochat数据集JSON文件路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./datas/img_conv_data/geochat/geochat_llava_val.json",
        help="输出的验证集JSON文件路径"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=200,
        help="要抽取的样本数量（默认: 200）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认: 42）"
    )
    
    args = parser.parse_args()
    
    create_validation_set(
        args.input,
        args.output,
        args.num_samples,
        args.seed
    )


if __name__ == "__main__":
    main()

