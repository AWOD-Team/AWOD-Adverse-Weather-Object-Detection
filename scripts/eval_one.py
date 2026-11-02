#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对单个权重在 test split 上评测并保存 JSON（总体 + per-class）。

用法: python eval_one.py <weights> <data.yaml> <out.json> <project_dir> <name>
"""
import json
import sys

from ultralytics import YOLO

w, yaml, out, project, name = sys.argv[1:6]
m = YOLO(w)
r = m.val(data=yaml, split="test", imgsz=640, batch=16, device=0,
          project=project + "/runs", name=name, exist_ok=True, plots=True, verbose=False)
b = r.box
names = r.names or m.names
idx = list(getattr(b, "ap_class_index", []) or [])
maps = list(getattr(b, "maps", []) or [])
ap50 = list(getattr(b, "ap50", []) or [])
d = {
    "precision": float(b.mp),
    "recall": float(b.mr),
    "map50": float(b.map50),
    "map50_95": float(b.map),
    "per_class_map50": {str(names[int(c)]): float(ap50[i]) for i, c in enumerate(idx) if i < len(ap50)},
    "per_class_map50_95": {str(names[int(c)]): float(maps[i]) for i, c in enumerate(idx) if i < len(maps)},
    "weights": w,
    "data": yaml,
}
json.dump(d, open(out, "w"), indent=2, ensure_ascii=False)
print(json.dumps(d, ensure_ascii=False))
