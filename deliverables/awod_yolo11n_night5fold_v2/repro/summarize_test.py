#!/usr/bin/env python3
"""夜间五折：逐折在 test 上评测 best.pt，采集总体 + per-class 指标，汇总 mean ± std (ddof=1)。

用法: python summarize_test.py <EXP_DIR>
      EXP_DIR 下需有 runs/fold{k}/weights/best.pt，folds 由 --folds 指定。
"""
import json
import os
import statistics as st
import sys

from ultralytics import YOLO

EXP = sys.argv[1] if len(sys.argv) > 1 else "/data/awod_data/exp_night5fold_v2"
FOLDS = "/data/awod_data/processed/night_2cls_clean_v2/folds5_grouped_v1"
OUT = os.path.join(EXP, "test")
os.makedirs(OUT, exist_ok=True)

rows = {}
for k in range(1, 6):
    best = os.path.join(EXP, "runs", f"fold{k}", "weights", "best.pt")
    yaml = os.path.join(FOLDS, f"fold{k}", "data.yaml")
    if not (os.path.exists(best) and os.path.exists(yaml)):
        print(f"[skip] fold{k} 缺少权重或 yaml")
        continue
    model = YOLO(best)
    r = model.val(data=yaml, split="test", imgsz=640, batch=16, device=0,
                  project=os.path.join(EXP, "runs"), name=f"fold{k}_test",
                  exist_ok=True, plots=True, verbose=False)
    b = r.box
    names = {int(i): n for i, n in (r.names or model.names).items()}
    d = {
        "precision": float(b.mp),
        "recall": float(b.mr),
        "map50": float(b.map50),
        "map50_95": float(b.map),
        "per_class_map50": {},
        "per_class_map50_95": {},
        "n_images": sum(1 for _ in open(os.path.join(FOLDS, f"fold{k}", "test.txt"))),
    }
    def _l(x):
        if x is None:
            return []
        try:
            return list(x)
        except TypeError:
            return []

    maps = _l(getattr(b, "maps", None))
    ap50 = _l(getattr(b, "ap50", None))
    idx = _l(getattr(b, "ap_class_index", None))
    for j, ci in enumerate(idx):
        name = names.get(int(ci), str(ci))
        if j < len(maps):
            d["per_class_map50_95"][name] = float(maps[j])
        if j < len(ap50):
            d["per_class_map50"][name] = float(ap50[j])
    rows[k] = d
    with open(os.path.join(OUT, f"fold{k}.json"), "w") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)
    print(f"fold{k}: mAP50={d['map50']:.4f} mAP50-95={d['map50_95']:.4f} "
          f"per-class={ {n: round(v, 4) for n, v in d['per_class_map50_95'].items()} }")

AGGS = [
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("map50", "mAP50"),
    ("map50_95", "mAP50-95"),
]
agg = {}
for key, label in AGGS:
    xs = [rows[k][key] for k in sorted(rows)]
    agg[label] = {"mean": st.mean(xs), "std": st.stdev(xs) if len(xs) > 1 else 0.0,
                  "min": min(xs), "max": max(xs), "n": len(xs)}
classes = sorted({c for d in rows.values() for c in d["per_class_map50_95"]})
for c in classes:
    for key, label in (("per_class_map50", "mAP50"), ("per_class_map50_95", "mAP50-95")):
        xs = [rows[k][key][c] for k in sorted(rows) if c in rows[k][key]]
        if xs:
            agg[f"{c} {label}"] = {"mean": st.mean(xs),
                                   "std": st.stdev(xs) if len(xs) > 1 else 0.0,
                                   "min": min(xs), "max": max(xs), "n": len(xs)}
with open(os.path.join(OUT, "test_summary.json"), "w") as f:
    json.dump({"folds": {f"fold{k}": v for k, v in rows.items()}, "aggregate": agg}, f,
              indent=2, ensure_ascii=False)

import csv
with open(os.path.join(OUT, "test_summary.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["fold", "precision", "recall", "map50", "map50_95"] + [f"{c}_map50" for c in classes] + [f"{c}_map50_95" for c in classes])
    for k in sorted(rows):
        d = rows[k]
        w.writerow([k, f"{d['precision']:.6f}", f"{d['recall']:.6f}", f"{d['map50']:.6f}", f"{d['map50_95']:.6f}"]
                   + [f"{d['per_class_map50'].get(c, float('nan')):.6f}" for c in classes]
                   + [f"{d['per_class_map50_95'].get(c, float('nan')):.6f}" for c in classes])
    for label in ("mean", "std"):
        vals = []
        for k in ("precision", "recall", "map50", "map50_95"):
            vals.append(f"{agg[{'precision':'Precision','recall':'Recall','map50':'mAP50','map50_95':'mAP50-95'}[k]][label]:.6f}")
        for c in classes:
            for key in ("per_class_map50", "per_class_map50_95"):
                a = agg.get(f"{c} {'mAP50' if key.endswith('50') else 'mAP50-95'}", {})
                vals.append(f"{a.get(label, float('nan')):.6f}")
        w.writerow([label] + vals)

print(json.dumps({k: {"mean": round(v["mean"], 4), "std": round(v["std"], 4)} for k, v in agg.items()},
                 indent=2, ensure_ascii=False))
print("SUMMARY_DONE")
