import numpy as np
from tabulate import tabulate


class IouStat:
    def __init__(self, cat_names=["ignore", "refer"], thresholds=(0.5, 0.6, 0.7, 0.8, 0.9)):
        self.cat_names = cat_names
        self.num_cats = len(cat_names)

        # --- NEW: Pr@X thresholds ---
        self.thresholds = np.asarray(thresholds, dtype=np.float64)
        self.num_thr = len(self.thresholds)

        # stats (make them arrays so .fill works)
        self.intersection = np.zeros(self.num_cats, dtype=np.float64)
        self.union = np.zeros(self.num_cats, dtype=np.float64)
        self.count = 0.0
        self.acc_iou = np.zeros(self.num_cats, dtype=np.float64)

        self.ciou = np.zeros(self.num_cats, dtype=np.float64)
        self.giou = np.zeros(self.num_cats, dtype=np.float64)

        # --- NEW: Pr hits & Pr values (per class) ---
        self.pr_hits = np.zeros((self.num_cats, self.num_thr), dtype=np.float64)
        self.pr = np.zeros((self.num_cats, self.num_thr), dtype=np.float64)

    def update(self, intersection, union, n=1):
        """
        Args:
            intersection: array-like, shape (num_cats,)
            union: array-like, shape (num_cats,)
            n: number of samples (usually 1)
        """
        intersection = np.asarray(intersection, dtype=np.float64)
        union = np.asarray(union, dtype=np.float64)

        self.intersection += intersection
        self.union += union
        self.count += float(n)

        iou_per_sample = np.where(union > 0, intersection / union, 1.0)  # (C,)
        self.acc_iou += iou_per_sample

        # --- NEW: Pr@X (hit rate): 1[IoU >= X] ---
        # shape: (C, T)
        hits = (iou_per_sample[:, None] >= self.thresholds[None, :]).astype(np.float64)
        self.pr_hits += hits * float(n)

    def average(self):
        # cIoU (cumulative IoU)
        self.ciou = np.where(self.union > 0, self.intersection / self.union * 100, 100.0)

        # gIoU (mean over samples)
        self.giou = np.where(self.count > 0, self.acc_iou / self.count * 100, 0.0)

        # --- NEW: Pr@X (%), mean over samples ---
        self.pr = np.where(self.count > 0, self.pr_hits / self.count * 100, 0.0)

    def reset(self):
        self.intersection.fill(0.0)
        self.union.fill(0.0)
        self.count = 0.0
        self.acc_iou.fill(0.0)
        self.ciou.fill(0.0)
        self.giou.fill(0.0)
        self.pr_hits.fill(0.0)
        self.pr.fill(0.0)

    def __repr__(self) -> str:
        pr_cols = [f"Pr@{t:.1f}" for t in self.thresholds]
        headers = ["", "cIoU", "gIoU", *pr_cols]

        data = []
        for i, cat_name in enumerate(self.cat_names):
            data.append([cat_name, self.ciou[i], self.giou[i], *list(self.pr[i])])

        table = tabulate(
            data,
            headers=headers,
            tablefmt="outline",
            floatfmt=".2f",
            stralign="center",
            numalign="center",
        )
        return str(table)