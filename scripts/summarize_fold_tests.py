#!/usr/bin/env python3
"""Evaluate every fold's best.pt on its held-out test.txt and summarize mean +/- std."""
import csv
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO

EXP = Path("/data/awod_data/exp_fog5fold_v2")
FOLDS = Path("/data/awod_data/processed/fog_2cls_clean_v1/folds5_grouped_v1")
NAMES = {0: "vehicle", 1: "person"}

rows = []
for k in range(1, 6):
    best = EXP / f"runs/fold{k}/weights/best.pt"
    yaml = FOLDS / f"fold{k}/data.yaml"
    model = YOLO(str(best))
    m = model.val(data=str(yaml), split="test", project=str(EXP / "runs"),
                  name=f"fold{k}_test_api", exist_ok=True, verbose=False)
    d = m.results_dict
    n_images = sum(1 for line in (FOLDS / f"fold{k}/test.txt").read_text().splitlines() if line.strip())
    row = {
        "fold": k,
        "images": n_images,
        "precision": round(float(d["metrics/precision(B)"]), 4),
        "recall": round(float(d["metrics/recall(B)"]), 4),
        "map50": round(float(d["metrics/mAP50(B)"]), 4),
        "map50_95": round(float(d["metrics/mAP50-95(B)"]), 4),
        "vehicle_map50": None,
        "vehicle_map50_95": None,
        "person_map50": None,
        "person_map50_95": None,
    }
    idx = list(m.box.ap_class_index)
    for i, c in enumerate(idx):
        nm = NAMES.get(int(c), str(c))
        if nm in ("vehicle", "person"):
            row[f"{nm}_map50_95"] = round(float(m.box.maps[i]), 4)
    # per-class AP50 from m.box.ap50 if available
    if getattr(m.box, "ap50", None) is not None:
        for i, c in enumerate(idx):
            nm = NAMES.get(int(c), str(c))
            if nm in ("vehicle", "person"):
                row[f"{nm}_map50"] = round(float(m.box.ap50[i]), 4)
    rows.append(row)

metrics = ["precision", "recall", "map50", "map50_95",
           "vehicle_map50", "vehicle_map50_95", "person_map50", "person_map50_95"]
summary = {}
for key in metrics:
    vals = [r[key] for r in rows if r[key] is not None]
    if vals:
        summary[key] = {"mean": round(float(np.mean(vals)), 4),
                        "std": round(float(np.std(vals, ddof=1)), 4),
                        "min": round(float(min(vals)), 4),
                        "max": round(float(max(vals)), 4)}

with (EXP / "test_summary.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

(EXP / "test_summary.json").write_text(
    json.dumps({"per_fold": rows, "summary": summary}, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8")

print("per-fold:")
for r in rows:
    print(r)
print("\nmean +/- std (test set):")
for k, v in summary.items():
    print(f"  {k:20s} {v['mean']:.4f} +/- {v['std']:.4f}  [{v['min']}, {v['max']}]")
