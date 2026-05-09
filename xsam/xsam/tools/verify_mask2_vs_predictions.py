#!/usr/bin/env python3
"""校验 mask2 与 mask_npy + predictions.json 中 segments_info 是否一致（与 vis 使用的类别 id 同源）。

推理未全部结束时可能没有 predictions.json，可仅用 --check-internal 做 npy/png 与 segment 一致性检查。

用法:
  python xsam/xsam/tools/verify_mask2_vs_predictions.py --pred-root ./output_align_eval/pred_data/panoptic_genseg_pano_predict
  python xsam/xsam/tools/verify_mask2_vs_predictions.py --pred-root ... --check-internal
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys

import numpy as np
from PIL import Image


def load_mask2_png(path: str) -> np.ndarray:
    im = Image.open(path)
    if im.mode == "L":
        return np.array(im, dtype=np.int32)
    if im.mode == "I":
        return np.array(im, dtype=np.int32)
    raise ValueError(f"unsupported PNG mode: {im.mode}")


def rebuild_category_map(seg_map: np.ndarray, segments_info: list, background: int = 0) -> np.ndarray:
    out = np.full(seg_map.shape, background, dtype=np.int32)
    for s in segments_info:
        sid = int(s["id"])
        cid = int(s["category_id"])
        out[seg_map == sid] = cid
    return out


def seg_cat_consistent(seg: np.ndarray, cat: np.ndarray) -> tuple[bool, int | None]:
    fs, fc = seg.ravel(), cat.ravel()
    m = fs > 0
    if not m.any():
        return True, None
    fs, fc = fs[m], fc[m]
    order = np.lexsort((fc, fs))
    fs, fc = fs[order], fc[order]
    ch = np.concatenate([[True], fs[1:] != fs[:-1]])
    idx = np.where(ch)[0]
    idx_end = np.concatenate([idx[1:], [len(fs)]])
    for a, b in zip(idx, idx_end):
        block = fc[a:b]
        if block.min() != block.max():
            return False, int(fs[a])
    return True, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", type=str, required=True, help="含 mask_npy、mask2、predictions.json 的目录")
    ap.add_argument(
        "--check-internal",
        action="store_true",
        help="仅检查 mask2 npy/png 一致及每 segment 对应单一类别（无需 predictions.json）",
    )
    args = ap.parse_args()
    root = args.pred_root
    pj = osp.join(root, "predictions.json")
    m2 = osp.join(root, "mask2")
    mn = osp.join(root, "mask_npy")

    npys = sorted(f for f in os.listdir(m2) if f.endswith(".npy"))
    if not npys:
        print(f"未找到 {m2}/*.npy", file=sys.stderr)
        return 1

    png_ok = seg_ok = 0
    png_bad: list[str] = []
    seg_bad: list[tuple] = []
    for fn in npys:
        stem = fn[:-4]
        a = np.load(osp.join(m2, fn))
        pp = osp.join(m2, stem + ".png")
        if osp.isfile(pp):
            b = load_mask2_png(pp)
            if a.shape == b.shape and np.array_equal(a, b):
                png_ok += 1
            else:
                png_bad.append(stem)
        seg = np.load(osp.join(mn, f"{stem}.npy"))
        if seg.shape != a.shape:
            seg_bad.append((stem, "shape"))
            continue
        ok, _ = seg_cat_consistent(seg, a)
        if ok:
            seg_ok += 1
        else:
            seg_bad.append((stem, "multi_cat"))

    print(f"[内部一致性] mask2 样本数: {len(npys)}")
    print(f"  mask2 .npy 与 .png 一致: {png_ok}/{len(npys)}")
    print(f"  mask_npy 每 segment 在 mask2 上为单一类别: {seg_ok}/{len(npys)}")
    if png_bad:
        print(f"  npy/png 不一致: {png_bad[:5]}...")
    if seg_bad:
        print(f"  segment 异常: {seg_bad[:5]}...")

    if args.check_internal:
        return 0 if not png_bad and not seg_bad and png_ok == len(npys) and seg_ok == len(npys) else 1

    if not osp.isfile(pj):
        print(f"\n未找到 {pj}（推理全部结束后才会写出）。已跳过与 segments_info 的逐项对齐。")
        print("与 vis 的关系: vis 与 mask2 均来自同一次推理的 seg_map + segments_info(category_id 为 dataset id)。")
        print("内部一致性通过后，类别语义与 vis 上色所用 category_id 一致。")
        return 0

    with open(pj, "r", encoding="utf-8") as f:
        entries = json.load(f)

    match = 0
    mismatch: list[str] = []
    for e in entries:
        stem = osp.splitext(e["image_file"])[0]
        p2 = osp.join(m2, f"{stem}.npy")
        if not osp.isfile(p2):
            continue
        seg = np.load(osp.join(root, e["mask_npy"]))
        exp = rebuild_category_map(seg, e["segments_info"])
        got = np.load(p2)
        if exp.shape == got.shape and np.array_equal(exp, got):
            match += 1
        else:
            mismatch.append(stem)

    print(f"\n[predictions.json] 可对齐条目中 mask2 与重建完全一致: {match}/{len(entries)}")
    if mismatch:
        print(f"  不一致样例: {mismatch[:8]}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
