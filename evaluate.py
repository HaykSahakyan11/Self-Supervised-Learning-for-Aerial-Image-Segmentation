# eval_uavid_patch_stitch_miou.py
import os
import torch
from tqdm import tqdm
import numpy as np

from data_process.uavid_dataset import UAVIDPatchStitch, CLASSES
from config import CONFIG
from utils.metric_tool import print_class_metrics_table

config = CONFIG()

patch_meta = config.UAVID_patched['9']['no_overlap']['val'] + '/patches_metadata.json'
logits_root = config.UAVID_patch_inf['dino_mc']['9']['no_overlap']['val']
label_root = config.UAVID['val'] + '/labels'
orig_h = 2160
orig_w = 3840

ds = UAVIDPatchStitch(
    split="val",
    patch_meta=patch_meta,
    logits_root=logits_root,
    label_root=label_root,
    expected_n=9,  # 4-patch grid in this example
    rep="logits",  # returns (C,H,W) logits
    resize=(orig_h, orig_w)  # keep native 2160×3840
)

NCLS = len(CLASSES)
inter = torch.zeros(NCLS, dtype=torch.float64)
union = torch.zeros(NCLS, dtype=torch.float64)
pix_tot = torch.zeros(NCLS, dtype=torch.float64)  # for overall accuracy
correct = torch.zeros(NCLS, dtype=torch.float64)

with torch.no_grad():
    for logits, gt in tqdm(ds, total=len(ds), desc="Evaluating"):
        pred = logits.argmax(0)  # (H,W)  int64

        for c in range(NCLS):
            p = (pred == c)
            g = (gt == c)

            inter[c] += torch.logical_and(p, g).sum()
            union[c] += torch.logical_or(p, g).sum()
            pix_tot[c] += g.sum()
            correct[c] += (p & g).sum()

eps = 1e-6
iou = (inter / (union + eps)).cpu()
acc_c = (correct / (pix_tot + eps)).cpu()
f1 = (2 * inter / (union + inter + eps)).cpu()
miou = iou[union > 0].mean().item()
mf1 = f1[union > 0].mean().item()
overall_acc = correct.sum().item() / pix_tot.sum().item()

metrics = dict(
    mean_accuracy=overall_acc,
    mean_iou=miou,
    mean_f1=mf1,
    per_class_iou=iou.tolist(),
    per_class_f1=f1.tolist(),
    per_class_accuracy=acc_c.tolist(),
    class_names=CLASSES
)

print_class_metrics_table(metrics)
print(f"\nOverall acc: {overall_acc * 100:.2f}% | "
      f"mIoU: {miou * 100:.2f}% | mean-F1: {mf1 * 100:.2f}%")
