import os
import json
import glob

# =======================================================================
# 1. 完美复刻你配置文件中的目录变量体系
# =======================================================================
base_root = "/mnt_llm_A100_V1/"
data_dir = "./datas/"  # 请确保在 xsam 根目录下执行此脚本
oneterra_data_root = base_root + "shui/oneterra_data/"
yangsen_data_root = base_root + "yangsen/datasets/"

fitrs_data_root = oneterra_data_root + "imgconv/FIT-RS/raw_data/"
fitrs_imgconv_data_path = fitrs_data_root + "train_data_of_each_individual_task/"
optical_caption_data_root = oneterra_data_root + "imgconv/image_caption/"
imgconv_data_root = data_dir + "img_conv_data/"
imgconv_cc_data_root = oneterra_data_root + "imgconv/FIT-RS/"
rsvqa_hr_data_root = oneterra_data_root + "imgconv/VQA/RSVQA_HR/"
rsvqa_lr_data_root = oneterra_data_root + "imgconv/VQA/RSVQA-LR/"
pano_data_root = data_dir + "pano/"
refseg_data_root = data_dir + "ref_seg_data/"
reasonseg_data_root_oneterra = oneterra_data_root + "reasonseg/"

# =======================================================================
# 2. 你的所有训练集分类与精确路径
# =======================================================================
DATASETS_TO_COUNT = {
    "🔴 算力黑洞：FIT-RS 海量泛化数据区 (需重点关注数据量)": [
        fitrs_imgconv_data_path + "train_instruction_complexcompre_708k_cleaned.json",
        fitrs_imgconv_data_path + "train_instruction_imagecaption_65k_cleaned.json",
        fitrs_imgconv_data_path + "train_instruction_imageclassification_130k_cleaned.json",
        fitrs_imgconv_data_path + "train_instruction_multiturn_50k_cleaned.json",
        fitrs_imgconv_data_path + "train_instruction_regioncaption_72k_cleaned.json",
        fitrs_imgconv_data_path + "train_instruction_vqa_400k_cleaned.json",
    ],
    
    "🟡 核心底座区：特征与模态维持": [
        imgconv_data_root + "geochat/geochat_llava.json",  # geochat
        yangsen_data_root + "sar_total/sft/train.json",    # sar_total
        pano_data_root + "annotations_train.json",         # pano_genseg / pano_ovseg (共用)
    ],
    
    "🟢 终极打榜突击队：Caption与VQA (评测集对应 train)": [
        optical_caption_data_root + "UCM-Captions/dataset_qwenvl_train_only.json",
        optical_caption_data_root + "NWPU-Captions/dataset_nwpu_qwenvl_train_only.json",
        imgconv_cc_data_root + "FIT-RSRC/FIT-RSRC_Questions_2k_qwenvl_train_only.json",
        imgconv_cc_data_root + "FIT-RSFG/FIT-RSFG-Bench/hrben_qwenvl_train_only.json",
        rsvqa_lr_data_root + "train_cleaned.json",
    ],
    
    "🟣 终极打榜突击队：ReaSeg (推理分割)": [
        # 从配置的 explain_path 中提取
        oneterra_data_root + "reasonseg/EarthReason_convert/explanatory/train.json",
        # diy1 没有写 explain_path，通常在根目录或 explanatory 下
        reasonseg_data_root_oneterra + "diy1/train.json", 
        reasonseg_data_root_oneterra + "diy1/explanatory/train.json", 
    ],

    "🔵 终极打榜突击队：RefSeg (指代分割目录扫描)": [
        # 指代分割没有直接给 json，给定的是 data_root
        refseg_data_root,                               # remotesam
        oneterra_data_root + "refseg/FAST/fast",        # fast
        oneterra_data_root + "refseg/RefSegRS",         # refsegrs
        oneterra_data_root + "refseg/RRSIS-D",          # rrsisd
        oneterra_data_root + "refseg/RISBench",         # risbench
    ]
}

def count_json_file(file_path):
    """自适应读取 JSON/JSONL 并返回条目数"""
    if not os.path.exists(file_path):
        return -1
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return len(data)
            elif isinstance(data, dict):
                for key in ['annotations', 'data', 'images']:
                    if key in data:
                        return len(data[key])
                return len(data.keys())
            return 0
    except json.JSONDecodeError:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return sum(1 for line in f if line.strip())
        except Exception:
            return -2
    except Exception:
        return -2

def scan_refseg_dir(dir_path):
    """专门扫描 RefSeg 等只提供目录的数据集，寻找可能的 json 标注"""
    if not os.path.exists(dir_path):
        return -1
    json_files = glob.glob(os.path.join(dir_path, "**", "*.json"), recursive=True)
    target_files = [f for f in json_files if "train" in f.lower() or "instances" in f.lower()]
    
    total_refs = 0
    found_files = []
    for f in target_files:
        c = count_json_file(f)
        if c > 0:
            total_refs += c
            found_files.append((os.path.basename(f), c))
    return total_refs, found_files

def main():
    total_grand = 0
    print("\n" + "="*80)
    print("🚀 X-SAM 全量数据集精准扫描与统计 (基于用户最新 Config)")
    print("="*80)

    for category, paths in DATASETS_TO_COUNT.items():
        print(f"\n{category}")
        print("-" * 80)
        category_total = 0
        
        for p in paths:
            # 如果是明确的 .json 文件
            if p.endswith('.json'):
                count = count_json_file(p)
                if count == -1:
                    print(f"  [缺失] ❌ {p.split('/')[-1]:<45} : 找不到文件 (请检查路径或执行目录)")
                elif count == -2:
                    print(f"  [错误] ⚠️ {p.split('/')[-1]:<45} : 解析失败")
                else:
                    print(f"  [正常] ✅ {p.split('/')[-1]:<45} : {count:>8} 条")
                    category_total += count
            
            # 如果是提供给 RefSeg 的目录
            else:
                count, files = scan_refseg_dir(p)
                dir_name = os.path.basename(p.strip('/'))
                if count == -1:
                    print(f"  [缺失] ❌ {dir_name:<45} : 找不到该目录")
                elif count == 0:
                    print(f"  [为空] ⚠️ {dir_name:<45} : 目录内未找到 train/instances 的 json 文件")
                else:
                    details = ", ".join([f"{n}({c})" for n, c in files])
                    print(f"  [目录] 📂 {dir_name:<45} : {count:>8} 条 (来源: {details})")
                    category_total += count

        print("-" * 80)
        print(f"  👉 本区总计估算: {category_total} 条")
        total_grand += category_total

    print("\n" + "="*80)
    print(f"🔥 所有训练集物理文件合并总计: {total_grand} 条")
    print("="*80)

    # 给用户的算力建议
    print("\n💡 【4卡 A40 算力微调建议】")
    if total_grand > 200000:
        print("🚨 警告：数据量极其庞大！如果在配置中硬跑，单 Epoch 耗时会按『天』计算。")
        print("   建议：立刻将【FIT-RS 海量泛化数据区】的 6 个 json 用脚本做物理抽样（各抽 2000 条），其余打榜数据全留。")
    elif total_grand > 80000:
        print("⚠️ 提醒：数据量偏大。如果通过 `repeats_scale` 将打榜数据提权 3~5 倍，Dataloader 里的数据会被撑到几十万条。请斟酌抽样。")
    else:
        print("✅ 完美：物理数据量合理。配合之前给你的大权重、冻结主干策略，可以直接开训！")

if __name__ == "__main__":
    main()
    
    
    
    
#diy_train: 233910/2条




# ================================================================================
# 🚀 X-SAM 全量数据集精准扫描与统计 (基于用户最新 Config)
# ================================================================================

# 🔴 算力黑洞：FIT-RS 海量泛化数据区 (需重点关注数据量)
# --------------------------------------------------------------------------------
#   [正常] ✅ train_instruction_complexcompre_708k_cleaned.json :   707552 条
#   [正常] ✅ train_instruction_imagecaption_65k_cleaned.json :    65197 条
#   [正常] ✅ train_instruction_imageclassification_130k_cleaned.json :   130400 条
#   [正常] ✅ train_instruction_multiturn_50k_cleaned.json  :    50624 条
#   [正常] ✅ train_instruction_regioncaption_72k_cleaned.json :    72026 条
#   [正常] ✅ train_instruction_vqa_400k_cleaned.json       :   389675 条
# --------------------------------------------------------------------------------
#   👉 本区总计估算: 1415474 条

# 🟡 核心底座区：特征与模态维持
# --------------------------------------------------------------------------------
#   [正常] ✅ geochat_llava.json                            :    99740 条
#   [正常] ✅ train.json                                    :  1006886 条
#   [正常] ✅ annotations_train.json                        :   111066 条
# --------------------------------------------------------------------------------
#   👉 本区总计估算: 1217692 条

# 🟢 终极打榜突击队：Caption与VQA (评测集对应 train)
# --------------------------------------------------------------------------------
#   [正常] ✅ dataset_qwenvl_train_only.json                :     1680 条
#   [正常] ✅ dataset_nwpu_qwenvl_train_only.json           :    25200 条
#   [正常] ✅ FIT-RSRC_Questions_2k_qwenvl_train_only.json  :     8230 条
#   [正常] ✅ hrben_qwenvl_train_only.json                  :    62534 条
#   [正常] ✅ train_cleaned.json                            :    57223 条
# --------------------------------------------------------------------------------
#   👉 本区总计估算: 154867 条

# 🟣 终极打榜突击队：ReaSeg (推理分割)
# --------------------------------------------------------------------------------
#   [正常] EearthReason_train.json                                    :     2371 条
#   [缺失] diy_train.json                                   : 116955 条
# --------------------------------------------------------------------------------
#   👉 本区总计估算: 2371 条

# 🔵 终极打榜突击队：RefSeg (指代分割目录扫描)
# --------------------------------------------------------------------------------
#   [目录] 📂 ref_seg_data                                  :   213662 条 (来源: instances.json(213662))
#   [目录] 📂 fast                                          :   124612 条 (来源: instances.json(124612))
#   [目录] 📂 RefSegRS                                      :     4420 条 (来源: instances.json(4420))
#   [目录] 📂 RRSIS-D                                       :    17402 条 (来源: instances.json(17402))
#   [目录] 📂 RISBench                                      :    52472 条 (来源: instances.json(52472))
# --------------------------------------------------------------------------------
#   👉 本区总计估算: 412568 条

# ================================================================================
# 🔥 所有训练集物理文件合并总计: 3202972 条
# ================================================================================