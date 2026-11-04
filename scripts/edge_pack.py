#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为香橙派 Zero 3W（H618，无 NPU）准备边缘模型包：
   - ncnn fp16 @320 / @640
   - ONNX fp32 @320
   - ONNX int8 @320（静态量化）
   - 打包到 deliverable 目录
"""
import glob
import json
import os
import shutil
import subprocess

O = "/data/awod_data/exp_v3_plan/n_final_deploy"
W = O + "/runs/deploy/weights"
PY = os.path.expanduser("~/anaconda3/envs/torch2.4_cuda12.1/bin/python")
YOLO = os.path.expanduser("~/anaconda3/envs/torch2.4_cuda12.1/bin/yolo")
DST = "/data/fog_detect/deliverables/edge_zero3w"
DATA_YAML = "/data/awod_data/processed/deploy_2cls_v1/data.yaml"


def run(cmd):
    print("$", cmd, flush=True)
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if p.returncode != 0:
        print("[warn] rc=%d\n%s" % (p.returncode, (p.stdout + p.stderr)[-800:]), flush=True)
    return p


def main():
    os.makedirs(DST, exist_ok=True)
    best = W + "/best.pt"

    # 1) ncnn 320 与 640（分别目录，避免覆盖）
    for size in (320, 640):
        out = W + "/ncnn_%d" % size
        if not os.path.exists(out):
            run("%s export model=%s format=ncnn imgsz=%d half=False" % (YOLO, best, size))
            cand = W + "/best_ncnn_model"
            if os.path.exists(cand):
                shutil.move(cand, out)
        if os.path.exists(out):
            d = "%s/ncnn_%d" % (DST, size)
            shutil.rmtree(d, ignore_errors=True)
            shutil.copytree(out, d)

    # 2) ONNX 320
    onnx320 = W + "/best_320.onnx"
    if not os.path.exists(onnx320):
        run("%s export model=%s format=onnx imgsz=320 simplify=True half=False" % (YOLO, best))
        cand = W + "/best.onnx"
        if os.path.exists(cand):
            shutil.move(cand, onnx320)
    if os.path.exists(onnx320):
        shutil.copy2(onnx320, DST + "/yolo11n_aw_320.onnx")

    # 3) ONNX int8 静态量化（320）
    int8 = DST + "/yolo11n_aw_320_int8.onnx"
    if os.path.exists(onnx320) and not os.path.exists(int8):
        code = r'''
import glob, os, numpy as np
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
src, dst = "%s", "%s"
imgs = sorted(glob.glob("/data/awod_data/processed/deploy_2cls_v1/images/*") if False else [])
class Reader(CalibrationDataReader):
    def __init__(self, files):
        self.files, self.i = files, 0
    def get_next(self):
        if self.i >= len(self.files): return None
        from PIL import Image
        p = self.files[self.i]; self.i += 1
        a = np.asarray(Image.open(p).convert("RGB").resize((320,320))).astype(np.float32)/255.0
        a = np.transpose(a,(2,0,1))[None]
        return {"images": a}
files = sorted(glob.glob("/data/awod_data/processed/fog_2cls_clean_v1/images/*"))[:200]
quantize_static(src, dst, Reader(files), quant_format=QuantType.QInt8, per_channel=True)
print("int8 ok")
''' % (onnx320, int8)
        open("/tmp/_q.py", "w").write(code)
        run("%s /tmp/_q.py" % PY)

    # 4) 清单
    info = {}
    for p in sorted(glob.glob(DST + "/**/*", recursive=True)):
        if os.path.isfile(p) and not p.endswith(".md"):
            info[os.path.relpath(p, DST)] = round(os.path.getsize(p) / 1e6, 2)
    json.dump(info, open(DST + "/manifest.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
