#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""夜间自动实验流水线（可断点续跑）。

阶段（按顺序，已完成自动跳过）:
  A_dehaze   用 DCP 生成去雾数据集（镜像雾天五折清单）
  B_d1       D1：去雾图训练 + 去雾图测 五折
  C_d2       D2：原图+去雾图混合训练 + 原图测 五折
  D_synth    用 DAWN 非雾子集合成 3 档人工雾
  E_b2       B2：真实雾 + 合成雾混合训练，真实雾测 五折
  F_b3       B3：仅合成雾训练，真实雾测（域偏差）
  G_subsets  从训练折内部构造 600/1200/2300 嵌套子集
  H_r0       R0：COCO 零样本直接评测（不训练）
  I_r1       R1：600 张全网络微调 五折
  J_r2       R2：600 张冻结骨干 五折
  K_r3       R3：600 张两阶段（冻结→解冻）五折
  L_summary  汇总所有 test 指标，写 CSV/MD 并提交 GitHub
"""
import json
import os
import shutil
import subprocess
import sys
import time

REPO = "/data/fog_detect"
BASE = "/data/awod_data"
PROC = BASE + "/processed"
FOG = PROC + "/fog_2cls_clean_v1"
FOGS = FOG + "/folds5_grouped_v1"
OUT = BASE + "/exp_v3_plan"
PY = os.path.expanduser("~/anaconda3/envs/torch2.4_cuda12.1/bin/python")
YOLO_BIN = os.path.expanduser("~/anaconda3/envs/torch2.4_cuda12.1/bin/yolo")
YOLO11N = "/data/fog_detect/yolo11n.pt"
LOG = OUT + "/log.txt"
CFG = ("epochs=100 imgsz=640 batch=16 seed=0 device=0 patience=30 lr0=0.001 "
       "amp=True cache=True workers=8 exist_ok=True")


def sh(cmd, log=LOG):
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write("\n$ " + cmd + "\n")
        f.flush()
        p = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
    return p.returncode


def note(msg):
    os.makedirs(OUT, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write("[%s] %s\n" % (time.strftime("%F %T"), msg))
    print("[%s] %s" % (time.strftime("%F %T"), msg), flush=True)


def done(stage):
    return os.path.exists(os.path.join(OUT, "done_" + stage))


def mark(stage):
    open(os.path.join(OUT, "done_" + stage), "w").write(time.strftime("%F %T"))


def train(yaml, project, name, extra=""):
    d = os.path.join(project, "runs", name)
    w = os.path.join(d, "weights", "best.pt")
    if os.path.exists(w):
        note("skip train %s (已有 best.pt)" % name)
        return w
    cmd = ("%s detect train model=%s data=%s project=%s/runs name=%s %s %s"
           % (YOLO_BIN, YOLO11N, yaml, project, name, CFG, extra))
    note("TRAIN %s" % name)
    rc = sh(cmd)
    if not os.path.exists(w):
        note("!! train failed rc=%d %s" % (rc, name))
    return w


def eval_test(weights, yaml, project, name):
    out_json = os.path.join(project, "test", name + ".json")
    if os.path.exists(out_json):
        note("skip eval %s" % name)
        return out_json
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    note("EVAL %s" % name)
    rc = sh("%s %s/scripts/eval_one.py %s %s %s %s %s_test"
            % (PY, REPO, weights, yaml, out_json, project, name))
    if not os.path.exists(out_json):
        note("!! eval failed rc=%d %s" % (rc, name))
    return out_json


def write_yaml(dirpath, name):
    y = (FOGS + "/fold%s/data.yaml")
    open(os.path.join(dirpath, "data.yaml"), "w").write(
        "path: %s\ntrain: %s/fold%s/train.txt\nval: %s/fold%s/val.txt\ntest: %s/fold%s/test.txt\n"
        "names:\n  0: vehicle\n  1: person\n"
        % ((dirpath, dirpath, name, dirpath, name, dirpath, name)))


def mirror_folds(src_folds, dst_dataset_root):
    """复制五折清单到新数据集，并把图片绝对路径前缀替换为新数据集根。"""
    folds = os.path.join(dst_dataset_root, "folds5_grouped_v1")
    for k in range(1, 6):
        src = os.path.join(src_folds, "fold%d" % k)
        dst = os.path.join(folds, "fold%d" % k)
        os.makedirs(dst, exist_ok=True)
        for split in ("train", "val", "test"):
            lines = [l.strip() for l in open(os.path.join(src, split + ".txt")) if l.strip()]
            lines = [l.replace(FOG + "/", dst_dataset_root.rstrip("/") + "/") for l in lines]
            open(os.path.join(dst, split + ".txt"), "w").write("\n".join(lines) + "\n")
        open(os.path.join(dst, "data.yaml"), "w").write(
            "path: %s\ntrain: %s/fold%d/train.txt\nval: %s/fold%d/val.txt\ntest: %s/fold%d/test.txt\n"
            "names:\n  0: vehicle\n  1: person\n"
            % (dst_dataset_root, folds, k, folds, k, folds, k))


# ---------------------------------------------------------------- stages
def s_A_dehaze():
    dst = PROC + "/fog_dehazed_2cls_v1"
    if done("A"):
        return
    note("== A: 生成去雾数据集 ==")
    code = r'''
import os,sys,multiprocessing as mp
import numpy as np
from PIL import Image
sys.path.insert(0,"/data/fog_detect")
from core.dehaze.dcp import DarkChannelPrior
SRC="/data/awod_data/processed/fog_2cls_clean_v1/images"
DST="/data/awod_data/processed/fog_dehazed_2cls_v1/images"
def work(n):
    p=os.path.join(SRC,n); o=os.path.join(DST,n)
    if os.path.exists(o): return 0
    img=np.array(Image.open(p).convert("RGB"))
    d=DarkChannelPrior(omega=0.95,win_size=15,t0=0.1,refinement="guided")
    out=d.dehaze(img)
    Image.fromarray(out).save(o,quality=95)
    return 1
if __name__=="__main__":
    os.makedirs(DST,exist_ok=True)
    names=sorted(os.listdir(SRC)); n=0
    with mp.Pool(8) as pool:
        for i,r in enumerate(pool.imap_unordered(work,names,chunksize=4)):
            n+=r
            if (i+1)%300==0: print("dehazed",i+1,flush=True)
    print("DONE dehazed",n,"of",len(names),flush=True)
'''
    open("/tmp/_dehaze.py", "w").write(code)
    sh("%s /tmp/_dehaze.py" % PY)
    dst = PROC + "/fog_dehazed_2cls_v1"
    os.makedirs(dst + "/labels", exist_ok=True)
    for n in os.listdir(FOG + "/labels"):
        s, d = FOG + "/labels/" + n, dst + "/labels/" + n
        if not os.path.exists(d):
            try:
                os.link(s, d)
            except OSError:
                shutil.copy2(s, d)
    shutil.copy2(FOG + "/classes.yaml", dst + "/classes.yaml")
    shutil.rmtree(dst + "/folds5_grouped_v1", ignore_errors=True)
    mirror_folds(FOGS, dst)
    note("A 完成: %d 张" % len(os.listdir(dst + "/images")))
    mark("A")


def s_B_d1():
    dst = PROC + "/fog_dehazed_2cls_v1"
    exp = OUT + "/d1_dehaze"
    if done("B"):
        return
    note("== B: D1 去雾图训练/测试 五折 ==")
    for k in range(1, 6):
        y = "%s/folds5_grouped_v1/fold%d/data.yaml" % (dst, k)
        w = train(y, exp, "fold%d" % k)
        if w and os.path.exists(w):
            eval_test(w, y, exp, "fold%d" % k)
    mark("B")


def s_C_d2():
    mixed = PROC + "/fog_plus_dehazed_2cls_v1"
    exp = OUT + "/d2_mixed"
    if done("C"):
        return
    note("== C: D2 混合训练/原图测试 五折 ==")
    if not os.path.exists(mixed + "/.built"):
        for sub in ("images", "labels"):
            os.makedirs(mixed + "/" + sub, exist_ok=True)
        dh = PROC + "/fog_dehazed_2cls_v1"
        for n in sorted(os.listdir(FOG + "/images")):
            for src, name in ((FOG + "/images/" + n, n), (dh + "/images/" + n, "dh_" + n)):
                d = mixed + "/images/" + name
                if not os.path.exists(d):
                    try:
                        os.link(src, d)
                    except OSError:
                        shutil.copy2(src, d)
        for n in sorted(os.listdir(FOG + "/labels")):
            for src, name in ((FOG + "/labels/" + n, n), (dh + "/labels/" + n, "dh_" + n)):
                d = mixed + "/labels/" + name
                if not os.path.exists(d):
                    try:
                        os.link(src, d)
                    except OSError:
                        shutil.copy2(src, d)
        # folds: train 翻倍，val/test 只用原图
        for k in range(1, 6):
            src = "%s/fold%d" % (FOGS, k)
            dst = "%s/folds5_grouped_v1/fold%d" % (mixed, k)
            os.makedirs(dst, exist_ok=True)
            base = mixed.rstrip("/")
            raw = [l.strip() for l in open(src + "/train.txt") if l.strip()]
            tr = [l.replace(FOG + "/", base + "/") for l in raw]
            tr_dh = [l.replace(FOG + "/images/", base + "/images/dh_") for l in raw]
            val = [l.strip().replace(FOG + "/", base + "/") for l in open(src + "/val.txt") if l.strip()]
            te = [l.strip().replace(FOG + "/", base + "/") for l in open(src + "/test.txt") if l.strip()]
            open(dst + "/train.txt", "w").write("\n".join(tr + tr_dh) + "\n")
            open(dst + "/val.txt", "w").write("\n".join(val) + "\n")
            open(dst + "/test.txt", "w").write("\n".join(te) + "\n")
            open(dst + "/data.yaml", "w").write(
                "path: %s\ntrain: %s/fold%d/train.txt\nval: %s/fold%d/val.txt\ntest: %s/fold%d/test.txt\n"
                "names:\n  0: vehicle\n  1: person\n" % (base, base, k, base, k, base, k))
        open(mixed + "/.built", "w").write("ok")
        note("C: 混合数据集构建完成")
    for k in range(1, 6):
        y = "%s/folds5_grouped_v1/fold%d/data.yaml" % (mixed, k)
        w = train(y, exp, "fold%d" % k)
        if w and os.path.exists(w):
            eval_test(w, y, exp, "fold%d" % k)
    mark("C")


def nonfog_dawn_list():
    fog_names = {n for n in os.listdir(FOG + "/images") if n.startswith("dawn__")}
    fog_stems = {n[len("dawn__"):].rsplit(".", 1)[0] for n in fog_names}
    src = PROC + "/dawn_2cls_clean_v1"
    keep = []
    for n in sorted(os.listdir(src + "/images")):
        stem = n.rsplit(".", 1)[0]
        if stem not in fog_stems:
            keep.append(n)
    return src, keep


def s_D_synth():
    if done("D"):
        return
    note("== D: 合成人工雾（DAWN 非雾子集） ==")
    src, keep = nonfog_dawn_list()
    work = OUT + "/synth_src"
    for sub in ("images", "labels"):
        os.makedirs(work + "/" + sub, exist_ok=True)
    for n in keep:
        s = src + "/images/" + n
        d = work + "/images/" + n
        if not os.path.exists(d):
            try:
                os.link(s, d)
            except OSError:
                shutil.copy2(s, d)
        ln = n.rsplit(".", 1)[0] + ".txt"
        s2, d2 = src + "/labels/" + ln, work + "/labels/" + ln
        if os.path.exists(s2) and not os.path.exists(d2):
            try:
                os.link(s2, d2)
            except OSError:
                shutil.copy2(s2, d2)
    note("D: 合成源 %d 张" % len(keep))
    sh("%s %s/scripts/synthesize_fog.py --root %s --out %s --per-image 3 "
       "--beta 1.0 3.5 --A 0.75 1.0 --depth vertical --noise 0.02 --seed 0"
       % (PY, REPO, work, work))
    n_fog = len(os.listdir(work + "/images_fog"))
    note("D: 生成合成雾 %d 张" % n_fog)
    mark("D")


def s_E_b2():
    exp = OUT + "/b2_synth_mix"
    if done("E"):
        return
    note("== E: B2 真实+合成雾混合训练，真实雾测 ==")
    work = OUT + "/synth_src"
    ds = PROC + "/fog_plus_synth_2cls_v1"
    if not os.path.exists(ds + "/.built"):
        os.makedirs(ds + "/images", exist_ok=True)
        os.makedirs(ds + "/labels", exist_ok=True)
        for n in os.listdir(FOG + "/images"):
            d = ds + "/images/" + n
            if not os.path.exists(d):
                try:
                    os.link(FOG + "/images/" + n, d)
                except OSError:
                    shutil.copy2(FOG + "/images/" + n, d)
        for n in os.listdir(work + "/images_fog"):
            d = ds + "/images/" + n
            if not os.path.exists(d):
                try:
                    os.link(work + "/images_fog/" + n, d)
                except OSError:
                    shutil.copy2(work + "/images_fog/" + n, d)
        for n in os.listdir(FOG + "/labels"):
            d = ds + "/labels/" + n
            if not os.path.exists(d):
                try:
                    os.link(FOG + "/labels/" + n, d)
                except OSError:
                    shutil.copy2(FOG + "/labels/" + n, d)
        for n in os.listdir(work + "/labels_fog"):
            d = ds + "/labels/" + n
            if not os.path.exists(d):
                try:
                    os.link(work + "/labels_fog/" + n, d)
                except OSError:
                    shutil.copy2(work + "/labels_fog/" + n, d)
        synth = sorted(work + "/images_fog/" + n for n in os.listdir(work + "/images_fog"))
        for k in range(1, 6):
            src = "%s/fold%d" % (FOGS, k)
            dst = "%s/folds5_grouped_v1/fold%d" % (ds, k)
            os.makedirs(dst, exist_ok=True)
            base = ds.rstrip("/")
            tr = [l.strip().replace(FOG + "/", base + "/") for l in open(src + "/train.txt") if l.strip()]
            val = [l.strip().replace(FOG + "/", base + "/") for l in open(src + "/val.txt") if l.strip()]
            te = [l.strip().replace(FOG + "/", base + "/") for l in open(src + "/test.txt") if l.strip()]
            extra = [p.replace(work + "/images_fog/", base + "/images/") for p in synth]
            open(dst + "/train.txt", "w").write("\n".join(tr + extra) + "\n")
            open(dst + "/val.txt", "w").write("\n".join(val) + "\n")
            open(dst + "/test.txt", "w").write("\n".join(te) + "\n")
            open(dst + "/data.yaml", "w").write(
                "path: %s\ntrain: %s/fold%d/train.txt\nval: %s/fold%d/val.txt\ntest: %s/fold%d/test.txt\n"
                "names:\n  0: vehicle\n  1: person\n" % (base, base, k, base, k, base, k))
        open(ds + "/.built", "w").write("ok")
        note("E: B2 数据集构建完成")
    for k in range(1, 6):
        y = "%s/folds5_grouped_v1/fold%d/data.yaml" % (ds, k)
        w = train(y, exp, "fold%d" % k)
        if w and os.path.exists(w):
            eval_test(w, y, exp, "fold%d" % k)
    mark("E")


def s_F_b3():
    exp = OUT + "/b3_synth_only"
    if done("F"):
        return
    note("== F: B3 仅合成雾训练，真实雾测 ==")
    work = OUT + "/synth_src"
    ds = PROC + "/synth_only_2cls_v1"
    if not os.path.exists(ds + "/.built"):
        os.makedirs(ds + "/images", exist_ok=True)
        os.makedirs(ds + "/labels", exist_ok=True)
        for n in os.listdir(work + "/images_fog"):
            d = ds + "/images/" + n
            if not os.path.exists(d):
                try:
                    os.link(work + "/images_fog/" + n, d)
                except OSError:
                    shutil.copy2(work + "/images_fog/" + n, d)
        for n in os.listdir(work + "/labels_fog"):
            d = ds + "/labels/" + n
            if not os.path.exists(d):
                try:
                    os.link(work + "/labels_fog/" + n, d)
                except OSError:
                    shutil.copy2(work + "/labels_fog/" + n, d)
        allimg = sorted(work + "/images_fog/" + n for n in os.listdir(work + "/images_fog"))
        # val 用真实雾 fold1 的 val（早停），test 用真实雾 fold1 test
        base = ds.rstrip("/")
        open(ds + "/train.txt", "w").write("\n".join(p.replace(work + "/images_fog/", base + "/images/") for p in allimg) + "\n")
        val = [l.strip().replace(FOG + "/", base + "/") for l in open(FOGS + "/fold1/val.txt") if l.strip()]
        te = [l.strip().replace(FOG + "/", base + "/") for l in open(FOGS + "/fold1/test.txt") if l.strip()]
        # 真实雾图片不在 ds/images 里，直接指向原路径
        val = [l.strip() for l in open(FOGS + "/fold1/val.txt") if l.strip()]
        te = [l.strip() for l in open(FOGS + "/fold1/test.txt") if l.strip()]
        open(ds + "/val.txt", "w").write("\n".join(val) + "\n")
        open(ds + "/test.txt", "w").write("\n".join(te) + "\n")
        open(ds + "/data.yaml", "w").write(
            "path: %s\ntrain: %s/train.txt\nval: %s/val.txt\ntest: %s/test.txt\n"
            "names:\n  0: vehicle\n  1: person\n" % (base, base, base, base))
        open(ds + "/.built", "w").write("ok")
    y = ds + "/data.yaml"
    w = train(y, exp, "fold1")
    if w and os.path.exists(w):
        for k in range(1, 6):
            yk = "%s/folds5_grouped_v1/fold%d/data.yaml" % (FOG, k)
            eval_test(w, yk, exp, "test_fold%d" % k)
    mark("F")


def s_G_subsets():
    if done("G"):
        return
    note("== G: 构造 600/1200/2300 嵌套子集（按来源组，从各折 train 池内部抽）==")
    import csv
    import random
    groups = {}
    with open(FOGS + "/group_manifest.csv", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            groups.setdefault(row["group_id"], []).append(row["name"])
    d = OUT + "/subsets"
    os.makedirs(d, exist_ok=True)
    order = sorted(groups)
    random.Random(0).shuffle(order)
    for k in range(1, 6):
        tr = [l.strip() for l in open("%s/fold%d/train.txt" % (FOGS, k)) if l.strip()]
        pool = {p.split("/")[-1] for p in tr}
        for target in (600, 1200, 2300):
            chosen, n = [], 0
            for g in order:
                if n >= target:
                    break
                n += sum(1 for nm in groups[g] if nm in pool)
                chosen.append(g)
            names = {nm for g in chosen for nm in groups[g]}
            keep = [p for p in tr if p.split("/")[-1] in names]
            open("%s/fold%d_train_%d.txt" % (d, k, target), "w").write("\n".join(keep) + "\n")
            note("G: fold%d target=%d -> 实取 %d 张 (组 %d)" % (k, target, len(keep), len(chosen)))
    mark("G")


def s_H_r0():
    exp = OUT + "/r0_zeroshot"
    if done("H"):
        return
    note("== H: R0 零样本评测 ==")
    for k in range(1, 6):
        y = "%s/fold%d/data.yaml" % (FOGS, k)
        eval_test(YOLO11N, y, exp, "fold%d" % k)
    mark("H")


def small_set_yaml(k, target, tag, extra_train=None):
    """生成小样本 data.yaml（train 用子集，val/test 沿用原折）。"""
    d = "%s/subsets/yaml_%s_%d_fold%d" % (OUT, tag, target, k)
    os.makedirs(d, exist_ok=True)
    tr = "%s/subsets/fold%d_train_%d.txt" % (OUT, k, target)
    lines = [l.strip() for l in open(tr) if l.strip()]
    if extra_train:
        lines += [l.strip() for l in open(extra_train) if l.strip()]
    open(d + "/train.txt", "w").write("\n".join(lines) + "\n")
    shutil.copy2("%s/fold%d/val.txt" % (FOGS, k), d + "/val.txt")
    shutil.copy2("%s/fold%d/test.txt" % (FOGS, k), d + "/test.txt")
    open(d + "/data.yaml", "w").write(
        "path: %s\ntrain: %s/train.txt\nval: %s/val.txt\ntest: %s/test.txt\n"
        "names:\n  0: vehicle\n  1: person\n" % (d, d, d, d))
    return d + "/data.yaml"


def s_I_r1():
    exp = OUT + "/r1_full_finetune_600"
    if done("I"):
        return
    note("== I: R1 600 张全网络微调 五折 ==")
    for k in range(1, 6):
        y = small_set_yaml(k, 600, "r1")
        w = train(y, exp, "fold%d" % k, extra="epochs=80")
        if w and os.path.exists(w):
            eval_test(w, y, exp, "fold%d" % k)
    mark("I")


def s_J_r2():
    exp = OUT + "/r2_frozen_600"
    if done("J"):
        return
    note("== J: R2 600 张冻结骨干 五折 ==")
    for k in range(1, 6):
        y = small_set_yaml(k, 600, "r2")
        w = train(y, exp, "fold%d" % k, extra="epochs=80 freeze=10")
        if w and os.path.exists(w):
            eval_test(w, y, exp, "fold%d" % k)
    mark("J")


def s_K_r3():
    exp = OUT + "/r3_two_stage_600"
    if done("K"):
        return
    note("== K: R3 600 张两阶段（冻结→解冻）五折 ==")
    for k in range(1, 6):
        y = small_set_yaml(k, 600, "r3")
        w1 = train(y, exp, "fold%d_s1" % k, extra="epochs=30 freeze=10 lr0=0.001")
        if not (w1 and os.path.exists(w1)):
            continue
        w2 = os.path.join(exp, "runs", "fold%d_s2" % k, "weights", "best.pt")
        if not os.path.exists(w2):
            # 阶段2：从阶段1权重继续，解冻全网络 + 低学习率
            cmd = ("%s detect train model=%s data=%s project=%s/runs name=fold%d_s2 %s freeze=0 lr0=0.0008 epochs=70"
                   % (YOLO_BIN, w1, y, exp, k, CFG))
            note("TRAIN fold%d_s2" % k)
            sh(cmd)
        if os.path.exists(w2):
            eval_test(w2, y, exp, "fold%d" % k)
    mark("K")


def s_L_summary():
    if done("L"):
        return
    note("== L: 汇总 ==")
    rows = []
    for exp in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, exp, "test")
        if not os.path.isdir(p):
            continue
        for f in sorted(os.listdir(p)):
            if not f.endswith(".json"):
                continue
            d = json.load(open(os.path.join(p, f)))
            rows.append({
                "experiment": exp, "fold": f.replace(".json", ""),
                "precision": round(d.get("precision", 0), 4),
                "recall": round(d.get("recall", 0), 4),
                "map50": round(d.get("map50", 0), 4),
                "map50_95": round(d.get("map50_95", 0), 4),
                "vehicle_map50_95": round(d.get("per_class_map50_95", {}).get("vehicle", 0), 4),
                "person_map50_95": round(d.get("per_class_map50_95", {}).get("person", 0), 4),
            })
    os.makedirs(OUT, exist_ok=True)
    import csv
    with open(OUT + "/all_results.csv", "w", newline="", encoding="utf-8") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    # 汇总 mean±std
    import statistics as st
    lines = ["# 夜间批量实验汇总（自动生成）", "", "| 实验 | fold | P | R | mAP50 | mAP50-95 | vehicle 50-95 | person 50-95 |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| {experiment} | {fold} | {precision} | {recall} | {map50} | {map50_95} | {vehicle_map50_95} | {person_map50_95} |".format(**r))
    lines += ["", "## 各实验均值", "", "| 实验 | mAP50 (mean±std) | mAP50-95 (mean±std) | n |", "|---|---|---|---|"]
    by = {}
    for r in rows:
        by.setdefault(r["experiment"], []).append(r)
    for exp, rs in by.items():
        m50 = [x["map50"] for x in rs]
        m5095 = [x["map50_95"] for x in rs]
        sd = (lambda v: st.stdev(v) if len(v) > 1 else 0.0)
        lines.append("| %s | %.4f ± %.4f | %.4f ± %.4f | %d |" %
                     (exp, st.mean(m50), sd(m50), st.mean(m5095), sd(m5095), len(rs)))
    open(OUT + "/SUMMARY.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    note("L: 汇总 %d 条" % len(rows))
    # 提交到 GitHub
    dst = REPO + "/reports/overnight_v3"
    os.makedirs(dst, exist_ok=True)
    for f in ("all_results.csv", "SUMMARY.md"):
        shutil.copy2(OUT + "/" + f, dst + "/" + f)
    sh("cd %s && git add -A && git -c user.name='函钩' -c user.email='tjdx1225230685@tju.edu.cn' "
       "commit -m 'feat(exp): 夜间自动流水线结果（去雾/合成雾/小样本）' || true" % REPO)
    sh("cd %s && git push origin main || true" % REPO)
    mark("L")


STAGES = [
    ("A", s_A_dehaze), ("B", s_B_d1), ("C", s_C_d2), ("D", s_D_synth),
    ("E", s_E_b2), ("F", s_F_b3), ("G", s_G_subsets), ("H", s_H_r0),
    ("I", s_I_r1), ("J", s_J_r2), ("K", s_K_r3), ("L", s_L_summary),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    note("############ PIPELINE START ############")
    for tag, fn in STAGES:
        if done(tag):
            note("skip stage %s (done)" % tag)
            continue
        try:
            fn()
        except Exception as e:
            note("!! stage %s 异常: %r" % (tag, e))
    note("############ PIPELINE FINISHED ############")
    open(OUT + "/FINISHED", "w").write(time.strftime("%F %T"))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        only = set(sys.argv[1:])
        for tag, fn in STAGES:
            if tag in only:
                note("### 单独执行阶段 %s ###" % tag)
                fn()
    else:
        main()
