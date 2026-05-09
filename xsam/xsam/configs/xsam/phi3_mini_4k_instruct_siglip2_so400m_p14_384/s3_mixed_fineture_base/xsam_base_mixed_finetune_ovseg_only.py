from copy import deepcopy

_base_ = "./xsam_base_mixed_finetune_all.py"

# 仅保留 ovseg 验证集，避免 genseg/reasonseg/imgconv 干扰定位。
val_datasets = [
    ds
    for ds in _base_.val_datasets
    if ds.get("task_name", "") == "ovseg" or "ovseg" in ds.get("data_name", "")
]

val_evaluators = [ev for ev in _base_.val_evaluators if "ovseg" in ev.get("data_name", "")]

# 可视化数据也同步只保留 ovseg，阈值沿用 base 逻辑。
vis_datasets = deepcopy(val_datasets)
for dataset in vis_datasets:
    if dataset.get("task_name") in ["genseg", "ovseg"]:
        dataset["postprocess_fn"]["threshold"] = 0.5  # type: ignore[index]

assert len(val_datasets) > 0, "No ovseg dataset found in base config."
assert len(val_datasets) == len(val_evaluators), (
    f"len(val_datasets)={len(val_datasets)} != len(val_evaluators)={len(val_evaluators)}"
)

