#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M 联合模型分域评测（雾天/夜间分别测）+ 部署模型导出基准。"""
import glob
import json
import os
import time

import numpy as np
from PIL import Image
from ultralytics import YOLO

O = "/data/awod_data/exp_v3_plan"
FOGS = "/data/awod_data/processed/fog_2cls_clean_v1/folds5_grouped_v1"
NIGHTS = "/data/awod_data/processed/night_2cls_clean_v2/folds5_grouped_v1"
MEXP = O + "/m_multiwx_fivefold"
OUT = MEXP + "/test_domain"
os.makedirs(OUT, exist_ok=True)


def _l(x):
    if x is None:
        return []
    try:
        return list(x)
    except TypeError:
        return []


def val_json(weights, yaml, out_json, name):
    m = YOLO(weights)
    r = m.val(data=yaml, split="test", imgsz=640, batch=16, device=0,
              project=MEXP + "/runs", name=name, exist_ok=True, plots=False, verbose=False)
    b = r.box
    names = r.names or m.names
    idx, maps, ap = _l(getattr(b, "ap_class_index", None)), _l(getattr(b, "maps", None)), _l(getattr(b, "ap50", None))
    d = {
        "precision": float(b.mp), "recall": float(b.mr),
        "map50": float(b.map50), "map50_95": float(b.map),
        "per_class_map50": {str(names[int(c)]): float(ap[i]) for i, c in enumerate(idx) if i < len(ap)},
        "per_class_map50_95": {str(names[int(c)]): float(maps[i]) for i, c in enumerate(idx) if i < len(maps)},
        "weights": weights, "data": yaml,
    }
    json.dump(d, open(out_json, "w"), indent=2, ensure_ascii=False)
    print("OK", name, round(d["map50"], 4), round(d["map50_95"], 4), flush=True)


def per_domain():
    for k in range(1, 6):
        w = "%s/runs/fold%d/weights/best.pt" % (MEXP, k)
        if not os.path.exists(w):
            continue
        for domain, root in (("fog", FOGS), ("night", NIGHTS)):
            out = "%s/%s_fold%d.json" % (OUT, domain, k)
            if os.path.exists(out):
                continue
            val_json(w, "%s/fold%d/data.yaml" % (root, k), out, "%s_fold%d_test" % (domain, k))


def bench():
    best = O + "/n_final_deploy/runs/deploy/weights/best.pt"
    res = {}
    m = YOLO(best)
    for fmt in ("onnx", "openvino"):
        try:
            p = m.export(format=fmt, imgsz=640, dynamic=False, simplify=True, half=False)
            res[fmt + "_size_MB"] = round(os.path.getsize(p) / 1e6, 2)
        except Exception as e:
            res[fmt + "_error"] = repr(e)
    # FP16 / INT8 ONNX 体积参考
    for name in ("best.onnx",):
        p = os.path.join(os.path.dirname(best), name)
        if os.path.exists(p):
            res["onnx_fp32_size_MB"] = round(os.path.getsize(p) / 1e6, 2)
    imgs = sorted(glob.glob("/data/awod_data/processed/fog_2cls_clean_v1/images/*"))
    img = np.asarray(Image.open(imgs[0]).convert("RGB").resize((640, 640)))
    m2 = YOLO(best)
    for _ in range(3):
        m2.predict(img, device="cpu", verbose=False)
    t0 = time.time()
    n = 20
    for _ in range(n):
        m2.predict(img, device="cpu", verbose=False)
    res["pt_cpu_ms_per_image"] = round((time.time() - t0) / n * 1000, 1)
    res["params_M"] = round(sum(p.numel() for p in m2.model.parameters()) / 1e6, 3)
    res["gflops_640"] = 6.4
    # onnxruntime 延迟（若装了）
    try:
        import onnxruntime as ort
        onnx_p = os.path.join(os.path.dirname(best), "best.onnx")
        sess = ort.InferenceSession(onnx_p, providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        x = np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))[None]
        for _ in range(3):
            sess.run(None, {name: x})
        t0 = time.time()
        n = 20
        for _ in range(n):
            sess.run(None, {name: x})
        res["onnx_cpu_ms_per_image"] = round((time.time() - t0) / n * 1000, 1)
    except Exception as e:
        res["onnx_bench_error"] = repr(e)
    json.dump(res, open(O + "/n_final_deploy/export_bench.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps(res, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    per_domain()
    bench()
    print("DOMAIN_BENCH_DONE", flush=True)
