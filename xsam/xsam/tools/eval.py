#!/usr/bin/env python

import argparse
import copy
import json
import os
import os.path as osp
import re
import sys
import traceback
import warnings
from typing import Dict, Optional, Tuple

# 添加项目根目录到Python路径
# 获取当前文件的目录，然后向上找到项目根目录（包含xsam目录的目录）
current_dir = osp.dirname(osp.abspath(__file__))
# eval.py 在 xsam/xsam/tools/ 下，需要向上3级到项目根目录
project_root = osp.dirname(osp.dirname(osp.dirname(current_dir)))
# 添加项目根目录和xsam目录到Python路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# 同时添加xsam目录（因为xsam模块在xsam/xsam/下）
xsam_dir = osp.join(project_root, "xsam")
if xsam_dir not in sys.path:
    sys.path.insert(0, xsam_dir)

import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.runner.utils import set_random_seed
from PIL import Image
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import GenerationConfig, StoppingCriteriaList
from xtuner.configs import cfgs_name_path
from xtuner.registry import BUILDER
from xtuner.tools.utils import set_model_resource
from xtuner.utils.device import get_device

from xsam.dataset.collate_fns import xsam_collate_fn
from xsam.utils.checkpoint import load_checkpoint
from xsam.utils.config import setup_model_config
from xsam.utils.constants import DEFAULT_SEG_TOKEN
from xsam.utils.dist import setup_distributed
from xsam.utils.logging import print_log, set_default_logging_format
from xsam.utils.misc import data_dict_to_device
from xsam.utils.utils import register_function

# Global setup
set_default_logging_format()
warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument("config", help="config file name or path")
    parser.add_argument("--work-dir", help="directory to save logs and models")
    parser.add_argument(
        "--pth_model",
        type=str,
        default=None,
        help="path to model checkpoint for evaluation",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override config options, format: xxx=yyy",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher type",
    )
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate per task (None means evaluate all samples)",
    )
    return parser.parse_args()


def get_gcg_phrases(input_ids, tokenizer, pstart_token_idx, pend_token_idx):
    pstart_idx = [i for i, x in enumerate(input_ids) if x == pstart_token_idx]
    pend_idx = [i + 1 for i, x in enumerate(input_ids) if x == pend_token_idx]
    phrases = []
    for ps, pe in zip(pstart_idx, pend_idx):
        phrase_ids = input_ids[ps + 1 : pe - 1]
        if (phrase_ids < 0).any():
            phrase = ""
        else:
            phrase = tokenizer.decode(phrase_ids).strip()
        phrases.append(phrase)
    return phrases


def get_gcg_caption(llm_generation_output):
    if DEFAULT_SEG_TOKEN not in llm_generation_output:
        return ""

    parts = llm_generation_output.split(".")
    sents = [part.strip() for part in parts if DEFAULT_SEG_TOKEN not in part]
    caption = ". ".join(sents)
    caption = re.sub(r"<.*?>", "", caption)
    caption = " ".join(caption.split()).strip("'").strip()
    return caption


def extract_answer_from_llm_output(llm_output: str) -> str:
    """从LLM输出中提取答案
    
    提取<|assistant|>之后、<|end|>之前的内容作为答案。
    如果找不到<|assistant|>标签，则尝试移除<|end|>标记并清理格式。
    只移除格式标记（<|end|>、<|endoftext|>、<|user|>等），保留模型生成的所有内容。
    如果模型生成了[question]:或Question:格式的内容，也会保留（不进行过滤）。
    
    Args:
        llm_output: LLM的完整生成输出
        
    Returns:
        提取的答案字符串（保留模型生成的所有内容，只移除格式标记）
    """
    if not llm_output:
        return ""
    
    # 首先尝试提取<|assistant|>之后的内容
    assistant_tag = "<|assistant|>"
    if assistant_tag in llm_output:
        # 找到最后一个<|assistant|>标签（可能有多轮对话）
        parts = llm_output.split(assistant_tag)
        if len(parts) > 1:
            # 取最后一部分（最后一个assistant的回答）
            answer_part = parts[-1]
            # 移除<|end|>标记
            answer = answer_part.split("<|end|>")[0].strip()
            # 移除<|endoftext|>标记
            answer = answer.split("<|endoftext|>")[0].strip()
            # 清理可能的格式标记（如###、##等）
            answer = re.sub(r'^#+\s*', '', answer)
            answer = answer.strip()
            
            # 移除任何残留的<|user|>标签（这是格式标记，不是模型生成的内容）
            # 如果答案中包含<|user|>标签，说明模型重复了输入，需要移除
            user_tag = "<|user|>"
            if user_tag in answer:
                # 找到最后一个<|user|>标签，只保留之前的内容
                user_parts = answer.split(user_tag)
                if len(user_parts) > 1:
                    # 取<|user|>之前的部分（这应该是真正的答案）
                    answer = user_parts[0].strip()
                else:
                    # 如果<|user|>在开头，移除它及其后面的内容
                    answer = answer.split(user_tag)[0].strip()
            
            # 保留模型生成的所有内容，包括[question]:或Question:格式（如果模型真的生成了这些，应该保留）
            return answer.strip()
    
    # 如果没有找到<|assistant|>标签，尝试直接移除<|end|>标记
    # 这种情况可能发生在输出格式不标准时
    answer = llm_output.split("<|end|>")[0].strip()
    # 移除<|endoftext|>标记
    answer = answer.split("<|endoftext|>")[0].strip()
    
    # 移除任何<|user|>标签（这是格式标记，不是模型生成的内容）
    user_tag = "<|user|>"
    if user_tag in answer:
        # 找到最后一个<|user|>标签，只保留之前的内容
        user_parts = answer.split(user_tag)
        if len(user_parts) > 1:
            # 取<|user|>之前的部分（这应该是真正的答案）
            answer = user_parts[0].strip()
        else:
            # 如果<|user|>在开头，移除它及其后面的内容
            answer = answer.split(user_tag)[0].strip()
    
    # 保留模型生成的所有内容，包括[question]:或Question:格式（如果模型真的生成了这些，应该保留）
    # 只清理格式标记（如开头的###等markdown标记）
    answer = re.sub(r'^#+\s*', '', answer)
    return answer.strip()


def extract_question_from_input(input_text: str) -> str:
    """从输入文本中提取最后一个user部分的问题
    
    对于imgconv任务，输入可能包含多轮对话历史，但评估时只需要最后一个问题。
    提取最后一个<|user|>之后、<|end|>之前的内容作为问题。
    
    Args:
        input_text: 完整的输入文本（可能包含多轮对话）
        
    Returns:
        提取的问题字符串（只包含最后一个user的问题）
    """
    print_log(f"text input: ")
    print_log(input_text)
    if not input_text:
        return ""
    
    # 查找最后一个<|user|>标签
    user_tag = "<|user|>"
    if user_tag in input_text:
        # 找到最后一个<|user|>标签
        parts = input_text.split(user_tag)
        if len(parts) > 1:
            # 取最后一部分（最后一个user的问题）
            question_part = parts[-1]
            # 移除<|end|>标记及其之后的内容
            question = question_part.split("<|end|>")[0].strip()
            # 移除<|endoftext|>标记
            question = question.split("<|endoftext|>")[0].strip()
            # 移除<|assistant|>标记及其之后的内容（如果模型在输入中包含了assistant标签）
            if "<|assistant|>" in question:
                question = question.split("<|assistant|>")[0].strip()
            # 清理首尾空白字符
            return question.strip()
    
    # 如果没有找到<|user|>标签，返回原始文本（可能是单轮对话）
    # 尝试移除<|end|>标记
    question = input_text.split("<|end|>")[0].strip()
    # 移除<|endoftext|>标记
    question = question.split("<|endoftext|>")[0].strip()
    # 移除<|assistant|>标记及其之后的内容
    if "<|assistant|>" in question:
        question = question.split("<|assistant|>")[0].strip()
    # 移除开头的<s>标记（如果存在）
    question = re.sub(r'^<s>\s*', '', question)
    return question.strip()


def process_batch(
    model,
    data: Dict,
    data_name: str,
    metadata: Dict,
    generation_config: Optional[GenerationConfig] = None,
    stop_criteria: Optional[StoppingCriteriaList] = None,
    mode: str = "tensor",
    save_llm_output: bool = False,
    llm_outputs_list: Optional[list] = None,
) -> Tuple[bool, Optional[torch.Tensor], Optional[str], Optional[list]]:
    """Process a single batch of data.

    Args:
        model: The model to evaluate
        data: Input data dictionary
        data_name: Name of the dataset
        generation_config: Generation configuration for LLM
        stop_criteria: Stopping criteria for LLM
        mode: Mode of the model
        save_llm_output: Whether to save LLM output
        llm_outputs_list: List to store LLM outputs

    Returns:
        Tuple of (success status, segmentation outputs, llm_generation_output, llm_generation_output_list)
    """
    data_samples = data["data_samples"]
    image_files = data_samples.image_files

    data_dict = {
        "input_ids": data["data_dict"].get("input_ids", None),
        "pixel_values": data["data_dict"].get("pixel_values", None),
        "seg_pixel_values": data["data_dict"].get("seg_pixel_values", None),
        "cond_ids": data["data_dict"].get("cond_ids", None),
        "seg_ids": data["data_dict"].get("seg_ids", None),
        "vprompt_masks": data["data_dict"].get("vprompt_masks", None),
    }

    llm_question_input = ""
    if data_dict["input_ids"] is not None:
        _input_ids = data_dict["input_ids"]
        llm_question_input = model.tokenizer.decode(_input_ids[_input_ids > 0])
    
    data_dict = data_dict_to_device(data_dict, device=model.device, dtype=model.dtype)

    llm_generation_output = ""
    with torch.no_grad():
        llm_outputs, seg_outputs = model(
            data_dict,
            data_samples,
            mode=mode,
            generation_config=generation_config,
            stopping_criteria=stop_criteria,
            metadata=metadata,
            do_postprocess=True,
            do_loss=False,
        )

    # Extract LLM generation output
    # 只解码新生成的部分（排除输入部分），避免输出中包含<|user|>标签和问题
    llm_generation_output_list = []
    if llm_outputs is not None and hasattr(llm_outputs, "sequences"):
        output_sequences = llm_outputs.sequences
        input_ids = data_dict.get("input_ids")
        
        # 只解码新生成的部分（排除输入部分）
        if input_ids is not None:
            # 确定batch大小和处理输入/输出序列
            if isinstance(input_ids, torch.Tensor):
                if input_ids.dim() > 1:
                    batch_size = input_ids.shape[0]
                    input_ids_processed = input_ids
                else:
                    batch_size = 1
                    input_ids_processed = input_ids.unsqueeze(0)
                    # 确保output_sequences也是2D
                    if isinstance(output_sequences, torch.Tensor) and output_sequences.dim() == 1:
                        output_sequences = output_sequences.unsqueeze(0)
            elif isinstance(input_ids, list):
                batch_size = len(input_ids)
                input_ids_processed = input_ids
            else:
                batch_size = 1
                input_ids_processed = [input_ids] if input_ids is not None else []
            
            # 处理每个样本
            for i in range(batch_size):
                # 获取输入长度
                if isinstance(input_ids_processed, torch.Tensor):
                    if input_ids_processed.dim() > 1:
                        input_seq = input_ids_processed[i]
                    else:
                        input_seq = input_ids_processed
                    # 计算有效输入长度（排除padding）
                    input_length = (input_seq > 0).sum().item()
                elif isinstance(input_ids_processed, list):
                    input_seq = input_ids_processed[i] if i < len(input_ids_processed) else input_ids_processed[0]
                    if isinstance(input_seq, torch.Tensor):
                        input_length = (input_seq > 0).sum().item()
                    else:
                        input_length = len([x for x in input_seq if x > 0])
                else:
                    input_length = 0
                
                # 获取输出序列
                if isinstance(output_sequences, torch.Tensor):
                    if output_sequences.dim() > 1:
                        output_seq = output_sequences[i]
                    else:
                        output_seq = output_sequences
                    output_length = output_seq.shape[0]
                elif isinstance(output_sequences, list):
                    output_seq = output_sequences[i] if i < len(output_sequences) else output_sequences[0]
                    if isinstance(output_seq, torch.Tensor):
                        output_length = output_seq.shape[0]
                    else:
                        output_length = len(output_seq)
                else:
                    output_length = 0
                
                # 只解码新生成的部分
                if output_length > input_length:
                    if isinstance(output_seq, torch.Tensor):
                        generated_ids = output_seq[input_length:]
                    else:
                        generated_ids = output_seq[input_length:]
                    generated_text = model.tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
                    
                    # 检查生成的文本是否仍然包含<|user|>标签（说明提取可能有问题）
                    # 如果包含，尝试找到<|assistant|>标签之后的内容
                    if "<|user|>" in generated_text and "<|assistant|>" in generated_text:
                        # 找到最后一个<|assistant|>标签之后的内容
                        assistant_parts = generated_text.split("<|assistant|>")
                        if len(assistant_parts) > 1:
                            generated_text = assistant_parts[-1].strip()
                            # 移除<|end|>标记
                            generated_text = generated_text.split("<|end|>")[0].strip()
                            # 移除<|endoftext|>标记
                            generated_text = generated_text.split("<|endoftext|>")[0].strip()
                    elif generated_text.startswith("<|user|>"):
                        # 如果生成的文本以<|user|>开头，说明提取有问题，返回空字符串
                        # 或者尝试找到<|assistant|>之后的内容
                        if "<|assistant|>" in generated_text:
                            assistant_parts = generated_text.split("<|assistant|>")
                            if len(assistant_parts) > 1:
                                generated_text = assistant_parts[-1].strip()
                                generated_text = generated_text.split("<|end|>")[0].strip()
                                generated_text = generated_text.split("<|endoftext|>")[0].strip()
                            else:
                                generated_text = ""
                        else:
                            generated_text = ""
                    else:
                        # 移除<|endoftext|>标记（即使没有<|user|>标签）
                        generated_text = generated_text.split("<|endoftext|>")[0].strip()
                    
                    llm_generation_output_list.append(generated_text)
                else:
                    # 如果输出长度等于或小于输入长度，尝试从完整输出中移除输入部分
                    full_output = model.tokenizer.decode(output_seq, skip_special_tokens=False).strip()
                    if isinstance(input_seq, torch.Tensor):
                        input_text = model.tokenizer.decode(input_seq[input_seq > 0], skip_special_tokens=False).strip()
                    else:
                        input_text = model.tokenizer.decode([x for x in input_seq if x > 0], skip_special_tokens=False).strip()
                    
                    if full_output.startswith(input_text):
                        generated_text = full_output[len(input_text):].strip()
                    else:
                        # 如果无法匹配，尝试找到<|assistant|>之后的内容
                        if "<|assistant|>" in full_output:
                            assistant_parts = full_output.split("<|assistant|>")
                            if len(assistant_parts) > 1:
                                generated_text = assistant_parts[-1].strip()
                                generated_text = generated_text.split("<|end|>")[0].strip()
                                generated_text = generated_text.split("<|endoftext|>")[0].strip()
                            else:
                                generated_text = full_output.split("<|endoftext|>")[0].strip()
                        else:
                            # 如果无法匹配，返回完整输出（可能输入已经被处理过）
                            generated_text = full_output.split("<|endoftext|>")[0].strip()
                    
                    # 再次检查是否包含<|user|>标签
                    if generated_text.startswith("<|user|>"):
                        if "<|assistant|>" in generated_text:
                            assistant_parts = generated_text.split("<|assistant|>")
                            if len(assistant_parts) > 1:
                                generated_text = assistant_parts[-1].strip()
                                generated_text = generated_text.split("<|end|>")[0].strip()
                                generated_text = generated_text.split("<|endoftext|>")[0].strip()
                            else:
                                generated_text = ""
                        else:
                            generated_text = ""
                    else:
                        # 移除<|endoftext|>标记（即使没有<|user|>标签）
                        generated_text = generated_text.split("<|endoftext|>")[0].strip()
                    
                    llm_generation_output_list.append(generated_text)
        else:
            # 如果没有input_ids，使用原来的方法（解码整个序列）
            llm_generation_output_list = model.tokenizer.batch_decode(output_sequences)
        
        llm_generation_output = llm_generation_output_list[0] if llm_generation_output_list else ""
        
        # Save LLM output if requested
        # 为batch中的每个样本都保存输出（支持batch_size > 1的情况）
        if save_llm_output and llm_outputs_list is not None:
            num_outputs = len(llm_generation_output_list)
            num_images = len(image_files) if isinstance(image_files, list) else 1
            # 确保每个样本都有对应的输出
            for i in range(max(num_outputs, num_images)):
                output_idx = min(i, len(llm_generation_output_list) - 1)
                image_file = image_files[i] if isinstance(image_files, list) and i < len(image_files) else (image_files[0] if image_files else "")
                llm_outputs_list.append({
                    "image_file": image_file,
                    "question": llm_question_input,  # 同一个batch的问题可能相同
                    "answer": llm_generation_output_list[output_idx] if llm_generation_output_list else "",
                })
    else:
        # 如果llm_outputs为None或没有sequences属性，记录警告信息
        if "imgconv" in data_name:
            # 对于imgconv任务，这是严重问题，因为需要生成文本
            print_log(
                f"Warning: imgconv task - llm_outputs is None or has no 'sequences' attribute. "
                f"mode={mode}, llm_outputs type={type(llm_outputs)}, "
                f"hasattr(sequences)={hasattr(llm_outputs, 'sequences') if llm_outputs is not None else False}",
                logger="current"
            )
        llm_generation_output = ""

    # 对于imgconv任务，seg_outputs为None是正常的（这是对话任务，没有分割输出）
    if seg_outputs is None:
        # 检查是否是imgconv任务
        if "imgconv" in data_name:
            # imgconv任务没有分割输出是正常的，返回成功
            # 确保llm_generation_output_list不为None（即使为空列表）
            if llm_generation_output_list is None:
                llm_generation_output_list = []
            # 如果llm_generation_output_list为空但llm_generation_output不为空，将其添加到列表中
            if len(llm_generation_output_list) == 0 and llm_generation_output:
                llm_generation_output_list = [llm_generation_output]
            return True, None, llm_generation_output, llm_generation_output_list
        else:
            # 其他任务如果seg_outputs为None则失败
            print_log(
                rf"Failed to get segmentation outputs: {image_files}, "
                rf"llm question_input: {repr(llm_question_input)}, "
                rf"llm generation_output: {repr(llm_generation_output)}",
                logger="current",
            )
            return False, None, llm_generation_output, llm_generation_output_list

    if "gcg" in data_name and llm_outputs is not None and hasattr(llm_outputs, "sequences"):
        gcg_phrases = [
            get_gcg_phrases(output_ids, model.tokenizer, model.pstart_token_idx, model.pend_token_idx)
            for output_ids in llm_outputs.sequences
        ]
        gcg_captions = [get_gcg_caption(output) for output in llm_generation_output_list]
        for i, segmentation_output in enumerate(seg_outputs):
            segmentation_output.update({"gcg_phrases": gcg_phrases[i], "gcg_caption": gcg_captions[i]})

    return True, seg_outputs, llm_generation_output, llm_generation_output_list


def evaluate_dataset(
    model,
    dataset,
    evaluator,
    rank: int,
    world_size: int,
    generation_config: Optional[GenerationConfig] = None,
    stop_criteria: Optional[StoppingCriteriaList] = None,
    output_dir: Optional[str] = None,
    visualizer: Optional[object] = None,
    save_visualizations: bool = True,
    max_vis_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
) -> None:
    """Evaluate model on a single dataset."""
    data_name = evaluator.data_name
    metadata = dataset.metadata
    output_ids_with_output = dataset.output_ids_with_output

    if visualizer is not None:
        visualizer.metadata = metadata
        for _cache_attr in (
            "_category_id_to_name_cache",
            "_debug_logged_categories",
            "_debug_logged_thing_categories",
        ):
            if hasattr(visualizer, _cache_attr):
                delattr(visualizer, _cache_attr)

    # 对于imgconv任务，必须使用predict模式来生成文本
    # 对于其他任务，根据output_ids_with_output决定使用tensor还是predict模式
    if "imgconv" in data_name:
        mode = "predict"
    else:
        mode = "tensor" if output_ids_with_output else "predict"

    # Setup dataloader
    sampler = DistributedSampler(dataset=dataset, rank=rank, num_replicas=world_size, shuffle=False)
    dataloader = DataLoader(
        dataset, batch_size=1, num_workers=4, sampler=sampler, shuffle=False, collate_fn=xsam_collate_fn
    )

    # Create directories for saving results
    if output_dir is not None and rank == 0:
        vis_dir = osp.join(output_dir, "visualizations", data_name)
        llm_output_dir = osp.join(output_dir, "llm_outputs")
        os.makedirs(vis_dir, exist_ok=True)
        os.makedirs(llm_output_dir, exist_ok=True)

    # Evaluation loop
    failed_cnt = 0
    evaluator.reset()
    llm_outputs_list = []
    vis_count = 0
    processed_count = 0  # Track total processed samples
    
    print_log(f"Evaluating {data_name}...", logger="current")
    if max_eval_samples is not None:
        print_log(f"Will evaluate at most {max_eval_samples} samples for this task", logger="current")
    else:
        print_log(f"Will evaluate ALL samples in the test set", logger="current")
    if max_vis_samples is not None:
        print_log(f"Will visualize at most {max_vis_samples} samples (to save time)", logger="current")
    print_log(f"Will save visualizations: {save_visualizations}", logger="current")

    for batch_idx, data in enumerate(tqdm(dataloader, desc=f"Evaluating {data_name}", disable=rank != 0)):
        # Check before processing to avoid unnecessary work (only if max_eval_samples is set)
        if max_eval_samples is not None and processed_count >= max_eval_samples:
            print_log(f"Reached max_eval_samples ({max_eval_samples}), stopping evaluation", logger="current")
            break
            
        success, seg_outputs, llm_output, llm_generation_output_list = process_batch(
            model, data, data_name, metadata, generation_config, stop_criteria, mode,
            save_llm_output=True, llm_outputs_list=llm_outputs_list
        )
        if not success:
            failed_cnt += 1
            continue
        
        # Debug: 对于imgconv任务，检查llm_output
        if "imgconv" in data_name and batch_idx < 3:
            print_log(f"DEBUG imgconv batch {batch_idx}: llm_generation_output_list length={len(llm_generation_output_list) if llm_generation_output_list else 0}, "
                     f"llm_output={repr(llm_output[:50]) if llm_output else 'None'}, "
                     f"llm_outputs_list length={len(llm_outputs_list) if llm_outputs_list else 0}", logger="current")
            if llm_outputs_list and len(llm_outputs_list) > 0:
                last_item = llm_outputs_list[-1]
                if isinstance(last_item, dict):
                    print_log(f"DEBUG imgconv batch {batch_idx}: last llm_outputs_list item answer={repr(last_item.get('answer', '')[:50]) if last_item.get('answer') else 'None'}", logger="current")

        image_infos = data["data_samples"].metainfo["image_infos"]
        conversations = data["data_samples"].metainfo["conversations"]
        val_inputs = copy.deepcopy(image_infos)
        if conversations is not None:
            assert len(conversations) == len(val_inputs), \
                f"Length mismatch: conversations({len(conversations)}) vs val_inputs({len(val_inputs)})"
            for i in range(len(val_inputs)):
                val_inputs[i]["conversation"] = conversations[i]
        print_log(f"val_inputs: {val_inputs}", logger="current")
        # Count actual number of samples in this batch
        num_samples_in_batch = len(val_inputs) if isinstance(val_inputs, list) else 1
        
        # Check if adding this batch would exceed the evaluation limit (only if max_eval_samples is set)
        # if max_eval_samples is not None and processed_count + num_samples_in_batch > max_eval_samples:
        #     # Only process the samples that fit within the limit
        #     remaining = max_eval_samples - processed_count
        #     if remaining > 0:
        #         val_inputs = val_inputs[:remaining]
        #         if isinstance(seg_outputs, list):
        #             seg_outputs = seg_outputs[:remaining]
        #         if llm_generation_output_list:
        #             llm_generation_output_list = llm_generation_output_list[:remaining]
        #         # Process the remaining samples
        #         if "imgconv" in data_name:
        #             # imgconv任务特殊处理
        #             for i, val_input in enumerate(val_inputs):
        #                 if isinstance(val_input, dict):
        #                     # 直接使用llm_generation_output_list（字符串列表）或llm_output（字符串），就像可视化部分那样
        #                     if llm_generation_output_list and i < len(llm_generation_output_list):
        #                         raw_answer = llm_generation_output_list[i]
        #                     elif llm_output:
        #                         raw_answer = llm_output
        #                     else:
        #                         raw_answer = ""
                            
        #                     # 从LLM输出中提取答案（移除<|assistant|>和<|end|>标签）
        #                     pred_answer = extract_answer_from_llm_output(raw_answer)
                            
        #                     if data["data_dict"].get("input_ids") is not None:
        #                         _input_ids = data["data_dict"]["input_ids"]
        #                         if isinstance(_input_ids, list) and i < len(_input_ids):
        #                             raw_question = model.tokenizer.decode(_input_ids[i][_input_ids[i] > 0])
        #                         elif isinstance(_input_ids, torch.Tensor):
        #                             if _input_ids.dim() > 1 and i < _input_ids.shape[0]:
        #                                 raw_question = model.tokenizer.decode(_input_ids[i][_input_ids[i] > 0])
        #                             else:
        #                                 raw_question = model.tokenizer.decode(_input_ids[_input_ids > 0])
        #                         else:
        #                             raw_question = model.tokenizer.decode(_input_ids[_input_ids > 0])
        #                     else:
        #                         raw_question = ""
                            
        #                     # 对于imgconv任务，只提取最后一个user部分的问题
        #                     if "imgconv" in data_name:
        #                         question = extract_question_from_input(raw_question)
        #                     else:
        #                         question = raw_question
                            
        #                     # 获取图像文件名，优先从val_input获取，其次从data_samples获取
        #                     image_file = val_input.get("image_file", "")
        #                     print_log(f"val_input image_file: {image_file}", logger="current")
        #                     # if not image_file and hasattr(data["data_samples"], "image_files") and data["data_samples"].image_files:
        #                     #     if isinstance(data["data_samples"].image_files, list) and i < len(data["data_samples"].image_files):
        #                     #         image_file = data["data_samples"].image_files[i]
        #                     #     elif data["data_samples"].image_files:
        #                     #         image_file = data["data_samples"].image_files[0] if not isinstance(data["data_samples"].image_files, list) else ""
                            
        #                     if hasattr(evaluator, 'add_prediction'):
        #                         evaluator.add_prediction(pred_answer, question, image_file)
                            
        #                     evaluator.process([val_input], None)
        #         else:
        #             evaluator.process(val_inputs, seg_outputs)
        #         processed_count += remaining
        #     break
        
        # 对于imgconv任务，即使seg_outputs为None也要处理
        if "imgconv" in data_name:
            # imgconv任务：将问题和答案传递给evaluator
            # 优先使用llm_generation_output_list（字符串列表）或llm_output（字符串）
            # 如果都没有，尝试从llm_outputs_list（字典列表）中获取当前batch的答案
            # 因为llm_outputs_list在process_batch中被填充，可能包含答案
            # 注意：可视化部分也使用llm_output，所以这里恢复的值会被可视化部分使用
            if not llm_generation_output_list and not llm_output:
                # 尝试从llm_outputs_list中获取当前batch的答案
                # llm_outputs_list是累积的，需要找到当前batch对应的条目
                # 由于每个batch都会追加，当前batch的答案应该在列表的末尾
                if llm_outputs_list and len(llm_outputs_list) > 0:
                    # 获取最后一个（或最后几个）条目，对应当前batch
                    num_samples = len(val_inputs) if isinstance(val_inputs, list) else 1
                    start_idx = max(0, len(llm_outputs_list) - num_samples)
                    batch_outputs = llm_outputs_list[start_idx:]
                    if batch_outputs and len(batch_outputs) > 0:
                        # 从字典中提取answer字段
                        if isinstance(batch_outputs[0], dict):
                            llm_generation_output_list = [item.get("answer", "") for item in batch_outputs]
                        else:
                            llm_generation_output_list = batch_outputs
                        if llm_generation_output_list and len(llm_generation_output_list) > 0:
                            llm_output = llm_generation_output_list[0] if llm_generation_output_list else ""
                            if batch_idx < 3:
                                print_log(f"DEBUG imgconv batch {batch_idx}: Retrieved answer from llm_outputs_list: {repr(llm_output[:50])}", logger="current")
            
            # 在恢复后再次检查，如果还是没有答案才打印警告
            if not llm_generation_output_list and not llm_output:
                print_log(f"Warning: No LLM output available for imgconv batch {batch_idx}, skipping predictions", logger="current")
            
            for i, val_input in enumerate(val_inputs):
                if isinstance(val_input, dict):
                    # 直接使用llm_generation_output_list（字符串列表）或llm_output（字符串），就像可视化部分那样
                    if llm_generation_output_list and i < len(llm_generation_output_list):
                        raw_answer = llm_generation_output_list[i]
                    elif llm_output:
                        # 如果索引超出范围，使用llm_output作为fallback
                        raw_answer = llm_output
                    else:
                        # 如果都没有，使用空字符串
                        raw_answer = ""
                        print_log(f"Warning: No prediction available for imgconv sample {i} in batch {batch_idx}", logger="current")
                    
                    # 从LLM输出中提取答案（移除<|assistant|>和<|end|>标签）
                    pred_answer = extract_answer_from_llm_output(raw_answer)
                    
                    # 从input_ids中提取问题
                    if data["data_dict"].get("input_ids") is not None:
                        _input_ids = data["data_dict"]["input_ids"]
                        if isinstance(_input_ids, list) and i < len(_input_ids):
                            raw_question = model.tokenizer.decode(_input_ids[i][_input_ids[i] > 0])
                        elif isinstance(_input_ids, torch.Tensor):
                            if _input_ids.dim() > 1 and i < _input_ids.shape[0]:
                                raw_question = model.tokenizer.decode(_input_ids[i][_input_ids[i] > 0])
                            else:
                                raw_question = model.tokenizer.decode(_input_ids[_input_ids > 0])
                        else:
                            raw_question = model.tokenizer.decode(_input_ids[_input_ids > 0])
                    else:
                        raw_question = ""
                    
                    # 对于imgconv任务，只提取最后一个user部分的问题
                    if "imgconv" in data_name:
                        question = extract_question_from_input(raw_question)
                    else:
                        question = raw_question
                    
                    # 获取图像文件名，优先从val_input获取，其次从data_samples获取
                    image_file = val_input.get("image_file", "")
                    # if not image_file and hasattr(data["data_samples"], "image_files") and data["data_samples"].image_files:
                    #     if isinstance(data["data_samples"].image_files, list) and i < len(data["data_samples"].image_files):
                    #         image_file = data["data_samples"].image_files[i]
                    #     elif data["data_samples"].image_files:
                    #         image_file = data["data_samples"].image_files[0] if not isinstance(data["data_samples"].image_files, list) else ""
                    
                    # 添加预测答案到evaluator
                    if hasattr(evaluator, 'add_prediction'):
                        print_log(f"Adding prediction for image_file: {image_file}, question: {question}, answer: {pred_answer}", logger="current")
                        evaluator.add_prediction(pred_answer, question, image_file)
                    else:
                        print_log(f"Warning: evaluator does not have add_prediction method for imgconv", logger="current")

                    evaluator.process([val_input], None)
        else:
            # 其他任务正常处理
            evaluator.process(val_inputs, seg_outputs)
        
        processed_count += num_samples_in_batch

        # Save visualizations and print LLM outputs
        # 对于imgconv任务，即使seg_outputs为None也要可视化
        # 对于refseg任务，即使seg_outputs为None也要尝试可视化（可能在某些情况下为None）
        should_visualize = (
            rank == 0 and 
            save_visualizations and 
            (max_vis_samples is None or vis_count < max_vis_samples) and
            (seg_outputs is not None or "imgconv" in data_name or "refseg" in data_name)
        )
        if should_visualize:
            try:
                # Print LLM output
                if llm_output:
                    print_log(f"\n{'='*80}", logger="current")
                    print_log(f"Sample {batch_idx + 1} - {data_name}", logger="current")
                    
                    # 获取图像文件名，优先从val_inputs获取，其次从data_samples获取
                    image_file = "unknown"
                    if val_inputs and len(val_inputs) > 0:
                        if isinstance(val_inputs[0], dict):
                            image_file = val_inputs[0].get('image_file', 'unknown')
                        elif hasattr(val_inputs[0], 'image_file'):
                            image_file = val_inputs[0].image_file
                    
                    if image_file == "unknown" and hasattr(data["data_samples"], "image_files") and data["data_samples"].image_files:
                        if isinstance(data["data_samples"].image_files, list) and len(data["data_samples"].image_files) > 0:
                            image_file = data["data_samples"].image_files[0]
                        elif data["data_samples"].image_files:
                            image_file = data["data_samples"].image_files[0] if not isinstance(data["data_samples"].image_files, list) else "unknown"
                    
                    print_log(f"Image: {image_file}", logger="current")
                    
                    # 从data中获取问题，而不是从llm_outputs_list
                    if data["data_dict"].get("input_ids") is not None:
                        _input_ids = data["data_dict"]["input_ids"]
                        if isinstance(_input_ids, (list, torch.Tensor)) and len(_input_ids) > 0:
                            if isinstance(_input_ids, torch.Tensor):
                                raw_question = model.tokenizer.decode(_input_ids[_input_ids > 0])
                            else:
                                raw_question = model.tokenizer.decode(_input_ids[0][_input_ids[0] > 0])
                        else:
                            raw_question = model.tokenizer.decode(_input_ids[_input_ids > 0])
                    else:
                        raw_question = "N/A"
                    
                    # 对于imgconv任务，只提取最后一个user部分的问题
                    if "imgconv" in data_name:
                        question = extract_question_from_input(raw_question)
                    else:
                        question = raw_question
                    
                    print_log(f"LLM Question: {question}", logger="current")
                    
                    # 提取并显示答案（移除<|assistant|>和<|end|>标签）
                    extracted_answer = extract_answer_from_llm_output(llm_output)
                    print_log(f"LLM Answer: {extracted_answer}", logger="current")
                    print_log(f"{'='*80}\n", logger="current")

                # Debug: Check conditions for visualization
                if batch_idx < 3:  # Only log first 3 samples to avoid too much output
                    print_log(f"DEBUG: batch_idx={batch_idx}, visualizer={visualizer is not None}, seg_outputs_len={len(seg_outputs) if seg_outputs else 0}", logger="current")
                    if hasattr(dataset, "image_folder"):
                        print_log(f"DEBUG: dataset.image_folder={dataset.image_folder}", logger="current")
                    else:
                        print_log(f"DEBUG: dataset does not have image_folder attribute", logger="current")

                # Save visualization if visualizer is available
                # 对于imgconv任务，即使seg_outputs为None也要可视化
                if visualizer is not None and (len(seg_outputs) > 0 if seg_outputs is not None else "imgconv" in data_name):
                    try:
                        # Get original image
                        # First try to get full path from val_inputs
                        image_file = val_inputs[0].get("image_file", "")
                        
                        # If image_file is just a filename (not a full path), combine with dataset.image_folder
                        if image_file and not osp.isabs(image_file) and hasattr(dataset, "image_folder") and dataset.image_folder:
                            image_file = osp.join(dataset.image_folder, image_file)
                        
                        if batch_idx < 3:
                            print_log(f"DEBUG: image_file from val_inputs: {image_file}, exists: {osp.exists(image_file) if image_file else False}", logger="current")
                        
                        if image_file and osp.exists(image_file):
                            sample_image = np.array(Image.open(image_file).convert("RGB"))
                        else:
                            # Try to get from data_samples
                            if hasattr(data["data_samples"], "image_files") and data["data_samples"].image_files:
                                image_file = data["data_samples"].image_files[0]
                                # If it's a relative path, combine with dataset.image_folder
                                if image_file and not osp.isabs(image_file) and hasattr(dataset, "image_folder") and dataset.image_folder:
                                    image_file = osp.join(dataset.image_folder, image_file)
                                if batch_idx < 3:
                                    print_log(f"DEBUG: image_file from data_samples: {image_file}, exists: {osp.exists(image_file) if image_file else False}", logger="current")
                                if osp.exists(image_file):
                                    sample_image = np.array(Image.open(image_file).convert("RGB"))
                                else:
                                    sample_image = None
                            else:
                                sample_image = None
                        
                        if batch_idx < 3:
                            print_log(f"DEBUG: sample_image is None: {sample_image is None}", logger="current")

                        # 对于imgconv和refseg任务，即使seg_outputs为None也要可视化
                        if sample_image is not None and (len(seg_outputs) > 0 if seg_outputs is not None else ("imgconv" in data_name or "refseg" in data_name)):
                            # 获取原始图片文件名用于命名可视化文件
                            original_image_file = ""
                            if val_inputs and len(val_inputs) > 0:
                                val_input = val_inputs[0]
                                if isinstance(val_input, dict):
                                    original_image_file = val_input.get("image_file", "")
                            
                            # 生成可视化文件名：使用原始图片文件名，如果没有则使用sample_编号
                            if original_image_file:
                                # 提取文件名（不含路径）并替换扩展名为.png
                                base_name = osp.splitext(osp.basename(original_image_file))[0]
                                vis_filename = f"{base_name}.png"
                            else:
                                vis_filename = f"sample_{batch_idx:05d}.png"
                            
                            vis_output_file = osp.join(vis_dir, vis_filename)
                            
                            # Get question text for refseg and imgconv tasks
                            question = None
                            answer = None
                            
                            if "imgconv" in data_name:
                                # 对于imgconv任务，提取答案（移除<|assistant|>和<|end|>标签）
                                answer = extract_answer_from_llm_output(llm_output) if llm_output else ""
                                # 从input_ids中提取问题
                                if data["data_dict"].get("input_ids") is not None:
                                    _input_ids = data["data_dict"]["input_ids"]
                                    if isinstance(_input_ids, (list, torch.Tensor)) and len(_input_ids) > 0:
                                        if isinstance(_input_ids, torch.Tensor):
                                            raw_question = model.tokenizer.decode(_input_ids[_input_ids > 0])
                                        else:
                                            raw_question = model.tokenizer.decode(_input_ids[0][_input_ids[0] > 0])
                                    else:
                                        raw_question = model.tokenizer.decode(_input_ids[_input_ids > 0])
                                else:
                                    raw_question = ""
                                
                                # 对于imgconv任务，只提取最后一个user部分的问题
                                question = extract_question_from_input(raw_question)
                                
                                # Draw predictions for imgconv task
                                try:
                                    visualizer.draw_predictions(
                                        sample_image,
                                        data_name=data_name,
                                        output_file=vis_output_file,
                                        question=question,
                                        answer=answer,
                                    )
                                    vis_count += 1
                                    if batch_idx < 3:
                                        print_log(f"Successfully saved imgconv visualization to {vis_output_file}", logger="current")
                                except Exception as e:
                                    print_log(f"Error saving imgconv visualization for sample {batch_idx}: {e}", logger="current")
                            elif "refseg" in data_name or "reaseg" in data_name:
                                # refseg/reaseg任务：处理seg_outputs
                                # 对于refseg任务，seg_outputs应该不为None，但为了安全起见，检查一下
                                if seg_outputs is None or len(seg_outputs) == 0:
                                    print_log(f"Warning: seg_outputs is None or empty for {data_name}, skipping visualization", logger="current")
                                    continue
                                # Get the first segmentation output
                                seg_output = seg_outputs[0]
                                
                                # Convert to dict if it's an object
                                if isinstance(seg_output, dict):
                                    vis_kwargs = seg_output.copy()
                                elif hasattr(seg_output, "__dict__"):
                                    vis_kwargs = seg_output.__dict__.copy()
                                elif hasattr(seg_output, "to_dict"):
                                    vis_kwargs = seg_output.to_dict()
                                else:
                                    vis_kwargs = {}
                                
                                if batch_idx < 3:
                                    print_log(f"DEBUG: seg_output type: {type(seg_output)}, vis_kwargs keys: {list(vis_kwargs.keys())}", logger="current")
                                
                                # Get phrases if available, default to empty list if None
                                phrases = vis_kwargs.get("gcg_phrases", None)
                                if phrases is None:
                                    phrases = []
                                
                                # Ensure segments_info has correct format
                                # 对于refseg任务，segments_info可能是一个字典而不是列表
                                if "segments_info" in vis_kwargs:
                                    segments_info = vis_kwargs["segments_info"]
                                    # 如果是字典，转换为列表格式（refseg通常使用字典格式）
                                    if isinstance(segments_info, dict):
                                        # refseg的segments_info通常是单个字典，包含score等信息
                                        # 保持字典格式，draw_ref_seg会处理
                                        pass
                                    elif isinstance(segments_info, list):
                                        # 对于panoptic任务，segments_info是列表
                                        # _PanopticPrediction.semantic_masks() and instance_masks() require 'isthing' key
                                        # Check if metadata has thing_dataset_id_to_contiguous_id
                                        thing_contiguous_ids = set()
                                        if metadata is not None and hasattr(metadata, "thing_dataset_id_to_contiguous_id"):
                                            thing_contiguous_ids = set(metadata.thing_dataset_id_to_contiguous_id.values())
                                        
                                        # Ensure each segment info dict has 'isthing' key
                                        for seg_info in segments_info:
                                            if isinstance(seg_info, dict) and "isthing" not in seg_info:
                                                # Try to determine isthing from category_id
                                                category_id = seg_info.get("category_id", None)
                                                if category_id is not None and thing_contiguous_ids:
                                                    seg_info["isthing"] = category_id in thing_contiguous_ids
                                                else:
                                                    # Default to False (stuff) if cannot determine
                                                    seg_info["isthing"] = False
                                
                                # Ensure segmentation is torch.Tensor (visualizer expects tensor, not numpy array)
                                # _PanopticPrediction.__init__ calls torch.unique(segmentation) which requires a Tensor
                                if "segmentation" in vis_kwargs:
                                    seg = vis_kwargs["segmentation"]
                                    # Convert to tensor if needed
                                    if isinstance(seg, np.ndarray):
                                        # Convert numpy array to tensor, ensuring correct dtype for segmentation masks
                                        seg_tensor = torch.from_numpy(seg.copy()).cpu()  # Use .copy() to avoid memory sharing issues
                                        # Ensure integer dtype for segmentation masks
                                        if seg_tensor.dtype not in (torch.int32, torch.int64, torch.long):
                                            seg_tensor = seg_tensor.long()
                                        vis_kwargs["segmentation"] = seg_tensor
                                    elif torch.is_tensor(seg):
                                        # Ensure it's on CPU and has correct dtype
                                        seg_tensor = seg.cpu()
                                        if seg_tensor.dtype not in (torch.int32, torch.int64, torch.long):
                                            seg_tensor = seg_tensor.long()
                                        vis_kwargs["segmentation"] = seg_tensor
                                    else:
                                        # Convert other types to tensor
                                        seg_tensor = torch.tensor(seg, dtype=torch.long).cpu()
                                        vis_kwargs["segmentation"] = seg_tensor
                                    
                                    if batch_idx < 3:
                                        print_log(f"DEBUG: segmentation type: {type(vis_kwargs['segmentation'])}, shape: {vis_kwargs['segmentation'].shape}, dtype: {vis_kwargs['segmentation'].dtype}, is_tensor: {torch.is_tensor(vis_kwargs['segmentation'])}", logger="current")
                                
                                # Remove non-visualization keys
                                vis_kwargs.pop("gcg_phrases", None)
                                vis_kwargs.pop("gcg_caption", None)
                                
                                if batch_idx < 3:
                                    print_log(f"DEBUG: About to call draw_predictions, output_file: {vis_output_file}", logger="current")
                                
                                # Get question text for refseg and reaseg tasks
                                if "refseg" in data_name or "reaseg" in data_name:
                                    # Get question from val_inputs (phrases or sampled_sents)
                                    if val_inputs and len(val_inputs) > 0:
                                        val_input = val_inputs[0]
                                        # Try to get phrases or sampled_sents from image_info
                                        if "phrases" in val_input and val_input["phrases"]:
                                            # phrases might be a list, get the first one
                                            phrases_list = val_input["phrases"]
                                            if isinstance(phrases_list, list) and len(phrases_list) > 0:
                                                question = phrases_list[0]
                                            elif isinstance(phrases_list, str):
                                                question = phrases_list
                                        elif "sampled_sents" in val_input and val_input["sampled_sents"]:
                                            # sampled_sents might be a list, get the first one
                                            sents_list = val_input["sampled_sents"]
                                            if isinstance(sents_list, list) and len(sents_list) > 0:
                                                question = sents_list[0]
                                            elif isinstance(sents_list, str):
                                                question = sents_list
                                    if batch_idx < 3:
                                        print_log(f"DEBUG: refseg question from val_inputs: {question}", logger="current")
                                
                                # Draw predictions for other tasks
                                try:
                                    visualizer.draw_predictions(
                                        sample_image,
                                        data_name=data_name,
                                        output_file=vis_output_file,
                                        phrases=phrases,
                                        question=question,
                                        **vis_kwargs,
                                    )
                                    vis_count += 1
                                    if batch_idx < 3 or vis_count % 10 == 0:
                                        print_log(f"Successfully saved visualization {vis_count} to {vis_output_file}", logger="current")
                                except Exception as vis_e:
                                    print_log(f"Error in draw_predictions for sample {batch_idx}: {vis_e}", logger="current")
                                    print_log(f"  seg_output keys: {list(vis_kwargs.keys())}", logger="current")
                                    import traceback
                                    print_log(f"  Traceback: {traceback.format_exc()}", logger="current")
                        else:
                            if batch_idx < 3:
                                if sample_image is None:
                                    print_log(f"DEBUG: Skipping visualization - sample_image is None", logger="current")
                                if len(seg_outputs) == 0:
                                    print_log(f"DEBUG: Skipping visualization - seg_outputs is empty", logger="current")
                    except Exception as e:
                        print_log(f"Error saving visualization for sample {batch_idx}: {e}", logger="current")
            except Exception as e:
                print_log(f"Error processing visualization for sample {batch_idx}: {e}", logger="current")

    # Save LLM outputs to JSON file
    if rank == 0 and output_dir is not None and llm_outputs_list:
        llm_output_file = osp.join(llm_output_dir, f"{data_name}_llm_outputs.json")
        with open(llm_output_file, "w", encoding="utf-8") as f:
            json.dump(llm_outputs_list, f, indent=2, ensure_ascii=False)
        print_log(f"Saved {len(llm_outputs_list)} LLM outputs to {llm_output_file}", logger="current")

    print_log(f"Processed {processed_count} samples for {data_name}", logger="current")
    print_log(f"Failed number of {data_name}: {failed_cnt}", logger="current")
    print_log(f"Evaluating {data_name} done!", logger="current")
    if rank == 0 and save_visualizations:
        print_log(f"Saved {vis_count} visualizations to {vis_dir}", logger="current")
    
    # 执行评估并获取结果
    eval_results = evaluator.evaluate()
    
    # 保存评估结果到txt文件
    if rank == 0 and output_dir is not None and eval_results is not None:
        results_txt_file = osp.join(output_dir, "evaluation_results", f"{data_name}_results.txt")
        os.makedirs(osp.dirname(results_txt_file), exist_ok=True)
        
        with open(results_txt_file, 'w', encoding='utf-8') as f:
            f.write(f"Evaluation Results for {data_name}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Number of samples: {processed_count}\n")
            f.write(f"Failed samples: {failed_cnt}\n\n")
            
            # 根据评估器类型格式化结果
            if isinstance(eval_results, str):
                # 评估器返回的是表格字符串（通过tabulate生成）
                f.write(eval_results)
            elif isinstance(eval_results, dict):
                # 评估器返回的是字典格式
                if len(eval_results) > 0:
                    for key, value in eval_results.items():
                        if isinstance(value, (int, float)):
                            f.write(f"{key}: {value:.4f}\n")
                        elif isinstance(value, (list, tuple)):
                            f.write(f"{key}: {value}\n")
                        else:
                            f.write(f"{key}: {value}\n")
                else:
                    f.write("No evaluation results (empty dict).\n")
            else:
                # 其他格式，尝试转换为字符串
                f.write(str(eval_results))
        
        print_log(f"Evaluation results saved to: {results_txt_file}", logger="current")


def main():
    """Main evaluation function."""
    args = parse_args()
    rank, local_rank, world_size = setup_distributed(args)

    # Load and process config
    if not osp.isfile(args.config):
        try:
            args.config = cfgs_name_path[args.config]
        except KeyError:
            raise FileNotFoundError(f"Cannot find {args.config}")

    cfg = Config.fromfile(args.config)
    set_model_resource(cfg)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    register_function(cfg._cfg_dict)
    if args.seed is not None:
        # Use args.seed
        set_random_seed(args.seed)
        print_log(
            f"Set the random seed to {args.seed}.",
            logger="current",
        )

    # Handle latest checkpoint
    if args.pth_model == "latest":
        from mmengine.runner import find_latest_checkpoint

        if osp.exists(osp.join(args.work_dir, "pytorch_model.bin")):
            args.pth_model = osp.join(args.work_dir, "pytorch_model.bin")
        else:
            args.pth_model = find_latest_checkpoint(args.work_dir)
        print_log(f"Found latest checkpoint: {args.pth_model}", logger="current")
    
    # Handle DeepSpeed checkpoint directory (iter_*.pth)
    if args.pth_model and osp.isdir(args.pth_model):
        # Check if it's a DeepSpeed checkpoint directory
        model_states_file = osp.join(args.pth_model, "mp_rank_00_model_states.pt")
        pytorch_model_file = osp.join(args.pth_model, "pytorch_model.bin")
        
        if osp.exists(model_states_file):
            # Use the model states file for DeepSpeed checkpoints
            args.pth_model = model_states_file
            print_log(f"Detected DeepSpeed checkpoint, using: {args.pth_model}", logger="current")
        elif osp.exists(pytorch_model_file):
            # Use pytorch_model.bin if it exists in the directory
            args.pth_model = pytorch_model_file
            print_log(f"Using pytorch_model.bin from checkpoint directory: {args.pth_model}", logger="current")
        else:
            # Try to use the directory itself (guess_load_checkpoint might handle it)
            print_log(f"Using checkpoint directory: {args.pth_model}", logger="current")
            print_log("Note: If loading fails, try specifying the full path to mp_rank_00_model_states.pt", logger="current")

    # Build and setup model
    model = BUILDER.build(cfg.model)
    if "llm" in cfg.model:
        model.llm.to(cfg.model.llm.torch_dtype)
    model.eval()
    model = model.to(get_device())
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank]).module

    load_checkpoint(model, args.pth_model)
    stop_criteria, generation_config = setup_model_config(model, cfg)

    # Setup visualizer if available in config
    visualizer = None
    if hasattr(cfg, "visualizer") and cfg.visualizer is not None:
        try:
            from xsam.utils.visualize import Visualizer
            visualizer = BUILDER.build(cfg.visualizer)
            print_log("Visualizer initialized successfully", logger="current")
        except Exception as e:
            print_log(f"Warning: Could not initialize visualizer: {e}", logger="current")
            visualizer = None

    # # Evaluate on all datasets
    # # 支持通过环境变量过滤任务
    # eval_only_task = os.environ.get("EVAL_ONLY_TASK", None)
    # if eval_only_task:
    #     print_log(f"只评估任务: {eval_only_task}", logger="current")
    #     cfg.val_datasets = [d for d in cfg.val_datasets if d.get('task_name') == eval_only_task]
    #     cfg.val_evaluators = [e for e in cfg.val_evaluators if eval_only_task in e.get('data_name', '')]
    #     print_log(f"过滤后: {len(cfg.val_datasets)} 个数据集, {len(cfg.val_evaluators)} 个评估器", logger="current")
    
    # # 临时过滤：只保留 refseg 和 imgconv 任务进行测试
    # # 屏蔽 genseg 和 ovseg 的评估
    # allowed_keywords = [ 'imgconv']
    # excluded_keywords = ['genseg', 'ovseg','refseg']
    # print_log(f"测试模式：只评估包含以下关键词的任务: {allowed_keywords}", logger="current")
    # print_log(f"将排除包含以下关键词的任务: {excluded_keywords}", logger="current")
    # original_dataset_count = len(cfg.val_datasets)
    # original_evaluator_count = len(cfg.val_evaluators)
    
    # # 过滤数据集：只保留包含 allowed_keywords 且不包含 excluded_keywords 的任务
    # def should_keep_task(data_name):
    #     if not data_name:
    #         return False
    #     data_name_lower = data_name.lower()
    #     # 必须包含至少一个允许的关键词
    #     has_allowed = any(keyword in data_name_lower for keyword in allowed_keywords)
    #     # 不能包含任何排除的关键词
    #     has_excluded = any(keyword in data_name_lower for keyword in excluded_keywords)
    #     return has_allowed and not has_excluded
    
    # cfg.val_datasets = [
    #     d for d in cfg.val_datasets 
    #     if should_keep_task(d.get('data_name', ''))
    # ]
    
    # # 过滤评估器：只保留包含 allowed_keywords 且不包含 excluded_keywords 的任务
    # cfg.val_evaluators = [
    #     e for e in cfg.val_evaluators 
    #     if should_keep_task(e.get('data_name', ''))
    # ]
    
    # filtered_out_datasets = original_dataset_count - len(cfg.val_datasets)
    # filtered_out_evaluators = original_evaluator_count - len(cfg.val_evaluators)
    # print_log(f"已过滤掉 {filtered_out_datasets} 个数据集, {filtered_out_evaluators} 个评估器", logger="current")
    # print_log(f"剩余: {len(cfg.val_datasets)} 个数据集, {len(cfg.val_evaluators)} 个评估器", logger="current")
    # if len(cfg.val_datasets) > 0:
    #     kept_datasets = [d.get('data_name', '') for d in cfg.val_datasets]
    #     print_log(f"保留的数据集: {kept_datasets}", logger="current")
    # if len(cfg.val_evaluators) > 0:
    #     kept_evaluators = [e.get('data_name', '') for e in cfg.val_evaluators]
    #     print_log(f"保留的评估器: {kept_evaluators}", logger="current")
    
    # 根据 data_name 匹配数据集和评估器，而不是简单按顺序 zip
    # 创建评估器字典，以 data_name 为键
    evaluator_dict = {e.get('data_name', ''): e for e in cfg.val_evaluators}
    
    # 匹配数据集和评估器
    matched_pairs = []
    unmatched_datasets = []
    for dataset_cfg in cfg.val_datasets:
        data_name = dataset_cfg.get('data_name', '')
        if data_name in evaluator_dict:
            matched_pairs.append((dataset_cfg, evaluator_dict[data_name]))
        else:
            unmatched_datasets.append(data_name)
            print_log(f"警告: 数据集 '{data_name}' 没有对应的评估器，将跳过", logger="current")
    
    if not matched_pairs:
        raise ValueError("没有找到匹配的数据集-评估器对，无法进行评估")
    
    if unmatched_datasets:
        print_log(f"警告: {len(unmatched_datasets)} 个数据集没有对应的评估器: {unmatched_datasets}", logger="current")
    
    print_log(f"Evaluating {len(matched_pairs)} datasets...", logger="current")
    print_log(f"Results will be saved to: {args.work_dir}", logger="current")
    
    for dataset_cfg, evaluator_cfg in matched_pairs:
        dataset = BUILDER.build(dataset_cfg)
        model.postprocess_fn = dataset.postprocess_fn

        evaluator = BUILDER.build(evaluator_cfg)
        evaluator.metadata = dataset.metadata
        evaluator.output_dir = osp.join(args.work_dir, "pred_data", evaluator.data_name)

        # 获取该任务的最大评估样本数：优先使用配置文件中的设置，否则使用命令行参数
        task_max_eval_samples = None
        if hasattr(dataset_cfg, 'max_eval_samples') and dataset_cfg.max_eval_samples is not None:
            task_max_eval_samples = dataset_cfg.max_eval_samples
        elif isinstance(dataset_cfg, dict) and 'max_eval_samples' in dataset_cfg:
            task_max_eval_samples = dataset_cfg.get('max_eval_samples')
        elif args.max_eval_samples is not None:
            task_max_eval_samples = args.max_eval_samples
        
        if task_max_eval_samples is not None:
            print_log(f"Task '{evaluator.data_name}' will evaluate at most {task_max_eval_samples} samples", logger="current")

        try:
            evaluate_dataset(
                model, dataset, evaluator, rank, world_size, generation_config, stop_criteria,
                output_dir=args.work_dir,
                visualizer=visualizer,
                save_visualizations=True,
                max_vis_samples=None,  # Process all samples for visualization
                max_eval_samples=task_max_eval_samples,  # Use task-specific or global limit
            )
        except Exception as e:
            print_log(f"Error evaluating {evaluator.data_name}: {e}\n{traceback.format_exc()}", logger="current")
            continue
    
    print_log(f"\n{'='*80}", logger="current")
    print_log("Evaluation completed!", logger="current")
    print_log(f"Results saved to: {args.work_dir}", logger="current")
    print_log(f"  - Predictions: {args.work_dir}/pred_data/", logger="current")
    print_log(f"  - Visualizations: {args.work_dir}/visualizations/", logger="current")
    print_log(f"  - LLM Outputs: {args.work_dir}/llm_outputs/", logger="current")
    print_log(f"  - Evaluation Results (TXT): {args.work_dir}/evaluation_results/", logger="current")
    print_log(f"{'='*80}\n", logger="current")


if __name__ == "__main__":
    main()
