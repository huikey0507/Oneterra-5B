"""与 xsam/tools/predict_genseg_pano.py 共用的 pano genseg 推理辅助函数。"""

import json
import os.path as osp
from typing import Dict, List, Tuple

from xsam.dataset.utils.catalog import MetadataCatalog

GENSEG_PANO_METADATA_NAME = "panoptic_genseg_pano_predict"
GENSEG_PANO_VIS_DATA_NAME = "panoptic_genseg_pano_val"


def resolve_pano_categories_json(cfg=None, explicit_path: str = None) -> str:
    """解析 pano 类别 JSON，默认与 xsam_predict_genseg_pano_021.sh 一致。"""
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    if cfg is not None:
        if hasattr(cfg, "pano_data_root") and cfg.pano_data_root:
            root = cfg.pano_data_root
            candidates.extend(
                [
                    osp.join(root, "annotations_val.json"),
                    osp.join(root, "annotations_train.json"),
                ]
            )
    candidates.extend(
        [
            "./assets/annotations_val.json",
            "./assets/pano/annotations_val.json",
            "./datas/pano/annotations_val.json",
        ]
    )
    for p in candidates:
        if p and osp.isfile(p):
            return osp.abspath(p)
    raise FileNotFoundError(
        "未找到 pano 类别 JSON（可设置 --pano-categories-json 或配置 pano_data_root）。"
        f" 已尝试: {candidates}"
    )


def load_pano_categories(pano_json_path: str) -> Tuple[List[str], List[str], List[str], Dict]:
    with open(pano_json_path, "r", encoding="utf-8") as f:
        anno = json.load(f)
    cats = anno.get("categories", [])
    if not cats:
        raise ValueError(f"categories is empty in {pano_json_path}")
    cats_by_id = sorted(cats, key=lambda x: int(x["id"]))
    all_classes = [c["name"] for c in cats_by_id]
    thing = [c["name"] for c in cats_by_id if int(c.get("isthing", 0)) == 1]
    stuff = [c["name"] for c in cats_by_id if int(c.get("isthing", 0)) == 0]
    return all_classes, thing, stuff, {"categories": cats}


def build_metadata_from_categories(data_name: str, cats: List[Dict]):
    dataset_id_to_contiguous_id = {int(c["id"]): i for i, c in enumerate(cats)}
    thing_dataset_id_to_contiguous_id = {
        int(c["id"]): dataset_id_to_contiguous_id[int(c["id"])] for c in cats if int(c.get("isthing", 0)) == 1
    }
    stuff_dataset_id_to_contiguous_id = {
        int(c["id"]): dataset_id_to_contiguous_id[int(c["id"])] for c in cats if int(c.get("isthing", 0)) == 0
    }
    thing_classes = {
        dataset_id_to_contiguous_id[int(c["id"])]: c["name"] for c in cats if int(c.get("isthing", 0)) == 1
    }
    stuff_classes = {
        dataset_id_to_contiguous_id[int(c["id"])]: c["name"] for c in cats if int(c.get("isthing", 0)) == 0
    }
    dataset_classes = {int(c["id"]): c["name"] for c in cats}

    metadata = MetadataCatalog.get(data_name)
    metadata.set(
        data_name=data_name,
        label_divisor=1000,
        ignore_label=255,
        dataset_id_to_contiguous_id=dataset_id_to_contiguous_id,
        thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id,
        stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id,
        thing_classes=thing_classes,
        stuff_classes=stuff_classes,
        dataset_classes=dataset_classes,
    )
    return metadata


def convert_segments_info_to_dataset_id(segments_info, metadata):
    if segments_info is None:
        return []
    cont_to_ds = {int(v): int(k) for k, v in metadata.dataset_id_to_contiguous_id.items()}
    thing_ds_ids = set(metadata.thing_dataset_id_to_contiguous_id.keys())
    out = []
    for s in segments_info:
        t = dict(s)
        cid = int(t.get("category_id", -1))
        ds_cid = cont_to_ds.get(cid, cid)
        t["category_id"] = ds_cid
        t["isthing"] = ds_cid in thing_ds_ids
        out.append(t)
    return out
