# FIT-RS imgconv训练数据集（使用cleaned.json文件）
# 1. Complex Comprehension (复杂理解)
fitrs_complexcompre_imgconv_dataset = dict(
    type=ImgConvDataset,
    data_path=fitrs_imgconv_data_path + "train_instruction_complexcompre_708k_cleaned.json",
    tokenizer=tokenizer,
    cond_type=cond_type,
    special_tokens=special_tokens,
    image_folder=fitrs_imgconv_image_folder,
    image_processor=image_processor,
    extra_image_processor=train_extra_image_processor,
    task_name="imgconv",
    data_name="fitrs_complexcompre_imgconv",
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=image_conv_map_fn,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pixel_values_ndim=2,
    is_multimodal=True,
    exclude_pure_text=True,
    pad_image_to_square=False,
    preprocess_text_data=True,
    repeats_scale=1,  # 708K样本，降低权重避免数据过多
)

# 2. Image Caption (图像描述)
fitrs_imagecaption_imgconv_dataset = dict(
    type=ImgConvDataset,
    data_path=fitrs_imgconv_data_path + "train_instruction_imagecaption_65k_cleaned.json",
    tokenizer=tokenizer,
    cond_type=cond_type,
    special_tokens=special_tokens,
    image_folder=fitrs_imgconv_image_folder,
    image_processor=image_processor,
    extra_image_processor=train_extra_image_processor,
    task_name="imgconv",
    data_name="fitrs_imagecaption_imgconv",
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=image_conv_map_fn,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pixel_values_ndim=2,
    is_multimodal=True,
    exclude_pure_text=True,
    pad_image_to_square=False,
    preprocess_text_data=True,
    repeats_scale=1,  # 65K样本，提高权重
)

# 3. Image Classification (图像分类)
fitrs_imageclassification_imgconv_dataset = dict(
    type=ImgConvDataset,
    data_path=fitrs_imgconv_data_path + "train_instruction_imageclassification_130k_cleaned.json",
    tokenizer=tokenizer,
    cond_type=cond_type,
    special_tokens=special_tokens,
    image_folder=fitrs_imgconv_image_folder,
    image_processor=image_processor,
    extra_image_processor=train_extra_image_processor,
    task_name="imgconv",
    data_name="fitrs_imageclassification_imgconv",
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=image_conv_map_fn,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pixel_values_ndim=2,
    is_multimodal=True,
    exclude_pure_text=True,
    pad_image_to_square=False,
    preprocess_text_data=True,
    repeats_scale=1,  # 130K样本，适中权重
)

# 4. Multi-turn Conversation (多轮对话)
fitrs_multiturn_imgconv_dataset = dict(
    type=ImgConvDataset,
    data_path=fitrs_imgconv_data_path + "train_instruction_multiturn_50k_cleaned.json",
    tokenizer=tokenizer,
    cond_type=cond_type,
    special_tokens=special_tokens,
    image_folder=fitrs_imgconv_image_folder,
    image_processor=image_processor,
    extra_image_processor=train_extra_image_processor,
    task_name="imgconv",
    data_name="fitrs_multiturn_imgconv",
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=image_conv_map_fn,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pixel_values_ndim=2,
    is_multimodal=True,
    exclude_pure_text=True,
    pad_image_to_square=False,
    preprocess_text_data=True,
    repeats_scale=1,  # 50K样本，提高权重
)

# 5. Region Caption (区域描述)
fitrs_regioncaption_imgconv_dataset = dict(
    type=ImgConvDataset,
    data_path=fitrs_imgconv_data_path + "train_instruction_regioncaption_72k_cleaned.json",
    tokenizer=tokenizer,
    cond_type=cond_type,
    special_tokens=special_tokens,
    image_folder=fitrs_imgconv_image_folder,
    image_processor=image_processor,
    extra_image_processor=train_extra_image_processor,
    task_name="imgconv",
    data_name="fitrs_regioncaption_imgconv",
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=image_conv_map_fn,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pixel_values_ndim=2,
    is_multimodal=True,
    exclude_pure_text=True,
    pad_image_to_square=False,
    preprocess_text_data=True,
    repeats_scale=1,  # 72K样本，提高权重
)

# 6. VQA (视觉问答)
fitrs_vqa_imgconv_dataset = dict(
    type=ImgConvDataset,
    data_path=fitrs_imgconv_data_path + "train_instruction_vqa_400k_cleaned.json",
    tokenizer=tokenizer,
    cond_type=cond_type,
    special_tokens=special_tokens,
    image_folder=fitrs_imgconv_image_folder,
    image_processor=image_processor,
    extra_image_processor=train_extra_image_processor,
    task_name="imgconv",
    data_name="fitrs_vqa_imgconv",
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=image_conv_map_fn,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pixel_values_ndim=2,
    is_multimodal=True,
    exclude_pure_text=True,
    pad_image_to_square=False,
    preprocess_text_data=True,
    repeats_scale=1,  # 400K样本，降低权重避免数据过多
)

        geochat_imgconv_dataset,
        fitrs_complexcompre_imgconv_dataset,
        fitrs_imagecaption_imgconv_dataset,
        fitrs_imageclassification_imgconv_dataset,
        fitrs_multiturn_imgconv_dataset,
        fitrs_regioncaption_imgconv_dataset,
        fitrs_vqa_imgconv_dataset,
        # SAR imgconv训练数据集
        sarlang_caption_imgconv_dataset,
        sarlang_vqa_imgconv_dataset,
        fusar_clip_caption_imgconv_dataset,
        fusar_clip_gf_vqa_atr_imgconv_dataset,
        sar_text_vqa_conv_imgconv_dataset,
        sar_total_imgconv_dataset,
        remotesam_refseg_dataset,