# 读取pkl
import pickle
import json

def load_pkl(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

def load_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

if __name__ == "__main__":
    json_path = "/mnt_llm_A100_V1/shui/LAE/X-SAM/X-SAM/datas/ref_seg_data/remotesam/instances.json"
    pkl_path = "/mnt_llm_A100_V1/shui/LAE/X-SAM/X-SAM/datas/ref_seg_data/remotesam/refs(unc).p"

    json_data = load_json(json_path)
    pkl_data = load_pkl(pkl_path)
    
    annotations = json_data["annotations"]
    images = json_data["images"]
    id2imagename = {img["id"]: img["file_name"] for img in images}

    print(f"Total annotations in JSON: {len(annotations)}")
    print(f"Total references in PKL: {len(pkl_data)}")

    for ann in annotations:
        segmentation = ann["segmentation"]
        image_id = ann["image_id"]
        ann_id = ann["id"]
        image_name = id2imagename[image_id]
        # print(f"Image ID: {image_id}, Image Name: {image_name}, Annotation ID: {ann_id}, Number of segmentations: {len(segmentation)}")
        if "200999.png" in image_name:
            print(ann)
        if len(segmentation) > 1:
            print(f"Image ID: {image_id}, Annotation ID: {ann_id}, Number of segmentations: {len(segmentation)}")
    
    for ref in pkl_data:
        ref_id = ref["ref_id"]
        ann_id = ref["ann_id"]
        if isinstance(ann_id, list):
            print(f"Ref ID: {ref_id}, Annotation IDs: {ann_id}")