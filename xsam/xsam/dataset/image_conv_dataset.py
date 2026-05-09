import copy
import json
import logging
import os

import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk
from mmengine import print_log
from PIL import Image
from xtuner.dataset.huggingface import process_hf_dataset
from xtuner.dataset.utils import expand2square

from .base_dataset import BaseDataset
from .utils.load import load_jsonl


class ImageConvDataset(BaseDataset):
    def __init__(
        self,
        *args,
        task_name="imgconv",
        offline_processed_text_folder=None,
        max_dataset_length=None,
        preprocess_text_data=False,
        is_multimodal=False,
        **kwargs,
    ):
        super().__init__(
            *args,
            task_name=task_name,
            offline_processed_text_folder=offline_processed_text_folder,
            max_dataset_length=max_dataset_length,
            preprocess_text_data=preprocess_text_data,
            is_multimodal=is_multimodal,
            **kwargs,
        )

    def custom_init(self, **kwargs):
        self.offline_processed_text_folder = kwargs.get("offline_processed_text_folder", None)
        self.max_dataset_length = kwargs.get("max_dataset_length", None)
        self.preprocess_text_data = kwargs.get("preprocess_text_data", False)
        self.is_multimodal = kwargs.get("is_multimodal", False)
    
    def _set_metadata(self, **kwargs):
        """为imgconv任务设置metadata（imgconv不需要分割相关的metadata）"""
        from .utils.catalog import MetadataCatalog
        metadata = MetadataCatalog.get(f"{self.data_name}")
        # imgconv任务不需要分割相关的metadata，只需要一个基本的metadata对象
        metadata.set(
            ignore_label=255,
            label_divisor=1000,
        )
        self._metadata = metadata

    @property
    def modality_length(self):
        # 缓存 modality_length 结果，避免重复计算
        if not hasattr(self, '_cached_modality_length'):
            length_list = []
            for data_dict in self.data:
                cur_len = (
                    sum(len(conv["value"].split()) for conv in data_dict["conversations"])
                    if not self.preprocess_text_data
                    else len(data_dict["input_ids"])
                )
                if data_dict.get("image", None) is None:
                    cur_len = -cur_len
                length_list.append(cur_len)
            self._cached_modality_length = length_list
        return self._cached_modality_length

    def _load_ann_data(self):
        assert self.offline_processed_text_folder or (self.data_path and self.tokenizer)
        if self.offline_processed_text_folder and self.data_path:
            print_log(
                "Both `offline_processed_text_folder` and "
                "`data_path` are set, and we load dataset from"
                "`offline_processed_text_folder` "
                f"({self.offline_processed_text_folder})",
                logger="current",
                level=logging.WARNING,
            )

        if self.offline_processed_text_folder is not None:
            self.data = load_from_disk(self.offline_processed_text_folder)
        else:
            # 使用更高效的文件读取方式
            if self.data_path.endswith(".json"):
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
            elif self.data_path.endswith(".jsonl"):
                json_data = load_jsonl(self.data_path)
            else:
                raise NotImplementedError

            data = json_data
            if self.preprocess_text_data:
                for idx in range(len(json_data)):
                    if isinstance(json_data[idx]["id"], int):
                        json_data[idx]["id"] = str(json_data[idx]["id"])
                json_data = DatasetDict({"train": HFDataset.from_list(json_data)})
                text_data = process_hf_dataset(
                    dataset=json_data,
                    tokenizer=self.tokenizer,
                    max_length=self.max_length,
                    dataset_map_fn=self.dataset_map_fn,
                    template_map_fn=self.template_map_fn,
                    split="train",
                    max_dataset_length=self.max_dataset_length,
                    remove_unused_columns=False,
                    pack_to_max_length=False,
                    with_image_token=True,
                )
                data = text_data

        # 优化数据预处理：使用列表推导式而不是循环
        processed_data = []
        for d in data:
            if self.preprocess_fn is not None:
                d = self.preprocess_fn(d)
            if "image" not in d:
                continue
            d["image_file"] = d.pop("image")
            processed_data.append(d)
        
        # 初始化metadata（imgconv任务需要）
        self._set_metadata()

        return processed_data

    def __getitem__(self, index):
        index = index % self.data_length
        data_dict = copy.deepcopy(self.data[index])
        if data_dict.get("image_file", None) is not None:
            image_file = data_dict["image_file"]
            # 使用更高效的路径组合和错误处理
            image_path = os.path.join(self.image_folder, image_file)
            try:
                pil_image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print_log(
                    f"Failed to load image: {image_path}, error: {e}",
                    logger="current",
                    level=logging.ERROR,
                )
                raise
            if self.image_processor is not None:
                image = pil_image
                # 如果 image_processor 是 SiglipProcessor，需要使用内部的 image_processor 属性
                actual_image_processor = self.image_processor
                if hasattr(self.image_processor, 'image_processor'):
                    actual_image_processor = self.image_processor.image_processor
                
                if self.pad_image_to_square:
                    # 获取 image_mean，优先使用 actual_image_processor 的，否则使用外层的
                    if hasattr(actual_image_processor, 'image_mean'):
                        image_mean = actual_image_processor.image_mean
                    elif hasattr(self.image_processor, 'image_mean'):
                        image_mean = self.image_processor.image_mean
                    else:
                        # 默认值（SigLIP 的标准均值）
                        image_mean = [0.5, 0.5, 0.5]
                    image = expand2square(pil_image, tuple(int(x * 255) for x in image_mean))
                
                # 使用实际的图像处理器处理图像
                if hasattr(actual_image_processor, 'preprocess'):
                    image = actual_image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
                else:
                    image = actual_image_processor(image, return_tensors="pt")["pixel_values"][0]
                data_dict["pixel_values"] = image
            if self.extra_image_processor is not None:
                seg_output = self.extra_image_processor.preprocess(pil_image, return_tensors="pt")
                data_dict["image_info"] = {"image_file": image_file}
                data_dict["seg_pixel_values"] = seg_output["pixel_values"][0]
                data_dict["image_size"] = seg_output["original_sizes"][0]
                data_dict["scaled_size"] = tuple(seg_output["scaled_sizes"][0].tolist())
                data_dict["task_name"] = self.task_name
            task_category = data_dict.get("task_category", None)
            data_dict.update(self._get_input_ids(data_dict, with_image_token=True))
            if task_category is not None:
                data_dict["task_category"] = task_category
        elif self.is_multimodal:
            if hasattr(self.image_processor, "crop_size"):
                crop_size = self.image_processor.crop_size
            else:
                crop_size = self.image_processor.size
            data_dict["pixel_values"] = torch.zeros(3, crop_size["height"], crop_size["width"])
            if self.extra_image_processor is not None:
                if hasattr(self.extra_image_processor, "crop_size"):
                    crop_size = self.extra_image_processor.crop_size
                elif hasattr(self.extra_image_processor, "pad_size"):
                    crop_size = self.extra_image_processor.pad_size
                else:
                    crop_size = self.extra_image_processor.size
                data_dict["seg_pixel_values"] = torch.zeros(3, crop_size["height"], crop_size["width"])
                data_dict["image_file"] = None
                data_dict["image_size"] = {"height": crop_size["height"], "width": crop_size["width"]}
                data_dict["image_info"] = {"image_file": None}
                data_dict["scaled_size"] = (crop_size["height"], crop_size["width"])
                data_dict["task_name"] = self.task_name
            data_dict.update(self._get_input_ids(data_dict, with_image_token=False))
        else:
            data_dict.update(self._get_input_ids(data_dict, with_image_token=True))

        return data_dict
