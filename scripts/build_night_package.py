#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""夜间五折 baseline 交付包构建：组装目录 + 生成 README/工作记录 + SHA256SUMS + zip。"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import zipfile
from datetime import datetime

EXP = "/data/awod_data/exp_night5fold_v2"
FOLDS = "/data/awod_data/processed/night_2cls_clean_v2/folds5_grouped_v1"
REPO = "/data/fog_detect"
NAME = "awod_yolo11n_night5fold_v2"
STAGE = "/data/" + NAME
ZIP = "/data/交付包_" + NAME + "_于乘骁.zip"

summ = json.load(open(os.path.join(EXP, "test", "test_summary.json")))
folds = summ["folds"]
agg = summ["aggregate"]


def f(x, n=4):
    return f"{x:.{n}f}"


# ---------- 时间线 ----------
timeline = []
logp = os.path.join(EXP, "logs", "run_5fold.log")
if os.path.exists(logp):
    for line in open(logp, errors="replace"):
        m = re.match(r"#+ fold(\d) (train|test eval)  (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", line.strip())
        if m:
            timeline.append(m.groups())
all_done = ""
if os.path.exists(logp):
    for line in open(logp, errors="replace"):
        if line.startswith("ALL_DONE"):
            all_done = line.strip().split(" ", 1)[1]
            break


def git(*args):
    try:
        return subprocess.check_output(["git", "-C", REPO, *args], text=True).strip()
    except Exception:
        return ""


commit = git("rev-parse", "--short", "HEAD")

# ---------- 复制文件 ----------
if os.path.exists(STAGE):
    shutil.rmtree(STAGE)
for d in ("configs", "curves", "repro", "results", "splits", "weights"):
    os.makedirs(os.path.join(STAGE, d))

for k in range(1, 6):
    run = os.path.join(EXP, "runs", f"fold{k}")
    shutil.copy2(os.path.join(run, "args.yaml"), os.path.join(STAGE, "configs", f"fold{k}_args.yaml"))
    shutil.copy2(os.path.join(run, "results.csv"), os.path.join(STAGE, "results", f"fold{k}_results.csv"))
    w = os.path.join(run, "weights", "best.pt")
    if os.path.exists(w):
        shutil.copy2(w, os.path.join(STAGE, "weights", f"fold{k}_best.pt"))
    cd = os.path.join(STAGE, "curves", f"fold{k}")
    os.makedirs(cd, exist_ok=True)
    for png, dest in (("results.png", "results.png"), ("BoxPR_curve.png", "BoxPR_curve.png"),
                      ("confusion_matrix_normalized.png", "confusion_matrix_normalized.png")):
        src = os.path.join(EXP, "runs", f"fold{k}_test", png) if png != "results.png" else os.path.join(run, png)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(cd, dest))
    sd = os.path.join(STAGE, "splits", f"fold{k}")
    os.makedirs(sd, exist_ok=True)
    for fn in ("data.yaml", "train.txt", "val.txt", "test.txt"):
        shutil.copy2(os.path.join(FOLDS, f"fold{k}", fn), os.path.join(sd, fn))

for fn in ("fold_summary.json", "group_manifest.csv", "split_manifest.csv", "README.md"):
    src = os.path.join(FOLDS, fn)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(STAGE, "splits", fn))

for fn, dest in (("run_night_fivefold.sh", "run_night_fivefold.sh"), ("summarize_test.py", "summarize_test.py")):
    src = os.path.join("/data/fog_detect/scripts", fn)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(STAGE, "repro", dest))

for fn in ("test_summary.csv", "test_summary.json"):
    shutil.copy2(os.path.join(EXP, "test", fn), os.path.join(STAGE, "results", fn))
shutil.copy2("/data/数据集交付说明.md", os.path.join(STAGE, "数据集交付说明.md"))

env = subprocess.run(
    "source ~/anaconda3/etc/profile.d/conda.sh && conda activate torch2.4_cuda12.1 && "
    "python -c \"import torch,ultralytics;print('python',__import__('sys').version.split()[0]);"
    "print('torch',torch.__version__);print('cuda',torch.version.cuda);"
    "print('ultralytics',ultralytics.__version__);print('gpu',torch.cuda.get_device_name(0))\" && nvidia-smi",
    shell=True, executable="/bin/bash", capture_output=True, text=True).stdout
with open(os.path.join(STAGE, "repro", "environment.txt"), "w") as fh:
    fh.write(env + f"\nrepo_commit: {commit}\nhost: {os.uname().nodename} {os.uname().machine}\n"
                   f"build_time: {datetime.now():%Y-%m-%d %H:%M:%S}\n")

# ---------- README ----------
classes = [c for c in ("vehicle", "person")
           if any(c in d["per_class_map50_95"] for d in folds.values())] or \
          sorted({c for d in folds.values() for c in d["per_class_map50_95"]})
res_rows = "\n".join(
    f"| {int(k[4:])} | {f(old['precision'])} | {f(old['recall'])} | {f(old['map50'])} | {f(old['map50_95'])} | "
    + " | ".join(f(old['per_class_map50_95'].get(c, float('nan'))) for c in classes) + " |"
    for k, old in [(k, folds[k]) for k in sorted(folds, key=lambda s: int(s[4:]))]
)
agg_rows = "\n".join(
    f"| {label} | **{f(agg[key]['mean'])} ± {f(agg[key]['std'])}** | {f(agg[key]['min'])} ~ {f(agg[key]['max'])} |"
    for key, label in [("Precision", "Precision"), ("Recall", "Recall"), ("mAP50", "**mAP50**"),
                       ("mAP50-95", "**mAP50-95**")]
)
class_agg = "\n".join(
    f"| {c} mAP50 | {f(agg[c+' mAP50']['mean'])} ± {f(agg[c+' mAP50']['std'])} | {f(agg[c+' mAP50']['min'])} ~ {f(agg[c+' mAP50']['max'])} |\n"
    f"| {c} mAP50-95 | {f(agg[c+' mAP50-95']['mean'])} ± {f(agg[c+' mAP50-95']['std'])} | {f(agg[c+' mAP50-95']['min'])} ~ {f(agg[c+' mAP50-95']['max'])} |"
    for c in classes
)
head_cols = " | ".join(f"{c} mAP50-95" for c in classes)

readme = f"""# AWOD 夜间目标检测 · YOLO11n 分组五折 baseline 交付包

- **交付人**：于乘骁
- **构建时间**：{datetime.now():%Y-%m-%d %H:%M}
- **数据集**：`night_2cls_clean_v2`（NOD 2669 + RTTS 夜间 79 + DAWN 夜间 32，共 2780 张）
- **类别**：`0 = vehicle`，`1 = person`
- **模型**：YOLO11n（COCO 预训练 → 各折微调 100 epoch）
- **协议**：分组五折（59 个来源组整组不拆），train/val/test 三集合两两不重叠，
  每张图恰好当一次 test；val 仅用于早停与选 checkpoint，test 只在最后评测一次。
- **代码提交**：`{commit}`

## 结果（留出测试集，5 折 mean ± std，std 为样本标准差 ddof=1）

| fold | P | R | mAP50 | mAP50-95 | {head_cols} |
|---|---|---|---|---|---|
{res_rows}

| 指标 | mean ± std | 区间 |
|---|---|---|
{agg_rows}
{class_agg}

## 复现

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate torch2.4_cuda12.1
# 环境: 见 repro/environment.txt
# 训练（5 折，约 1 小时 / RTX 4090）
bash repro/run_night_fivefold.sh /data/awod_data/exp_night5fold_v2
# 汇总 mean ± std
python repro/summarize_test.py /data/awod_data/exp_night5fold_v2
```

> **注意**：`splits/fold*/data.yaml` 与各 `*.txt` 内是**绝对路径** `/data/awod_data/...`。
> 换机器或改目录名时必须同步改路径，否则无法训练/评测。

## 目录

```
{NAME}/
├── README.md                     本文件
├── 工作记录.md                    完整过程记录
├── 数据集交付说明.md               数据集口径（原样附带）
├── configs/foldN_args.yaml       每折完整训练超参
├── results/
│   ├── test_summary.csv/.json    5 折 test 指标 + mean±std
│   └── foldN_results.csv         每折逐轮训练/验证指标
├── splits/                       固化划分（可复现）
│   ├── foldN/{{data.yaml,train.txt,val.txt,test.txt}}
│   ├── fold_summary.json         各折规模
│   ├── group_manifest.csv        每图所属来源组
│   └── split_manifest.csv        完整划分清单
├── weights/foldN_best.pt         5 折最优权重
├── curves/foldN/                 训练曲线、PR 曲线、混淆矩阵
├── repro/                        训练/汇总脚本 + 环境信息
└── SHA256SUMS                    完整性校验（sha256sum -c SHA256SUMS）
```

## 局限

1. **各折 test 规模不均衡（340 ~ 781 张）**：NOD 只有 8 个拍摄日，整天必须同折，
   单日拍摄量差异大。fold1 的 test 为 781 张、train 相应偏小。报告时请注明，
   或另报"合并全部 test 预测"的 pooled mAP 以降低方差影响。
2. **person 明显弱于 vehicle**：夜间行人目标小、遮挡多，mAP50-95 差距见上表。
3. **vehicle 语义偏窄**：NOD 只标 `car`，RTTS 的 vehicle 含 bus/motorbike；合并后主要指小轿车。
4. **30 张未清洗图**：`manifest.csv` 中 `flag=not_in_clean_set`，见数据集交付说明第六节。
5. 本包为**基线**，未做去雾、小样本、跨域等更复杂实验。
"""
open(os.path.join(STAGE, "README.md"), "w").write(readme)

# ---------- 工作记录 ----------
tl_lines = "\n".join(f"| fold{m[0]} {m[1]} | {m[2]} |" for m in timeline)
log = f"""# AWOD 夜间目标检测 · 五折 baseline 工作记录

- **交付人**：于乘骁
- **日期**：{datetime.now():%Y-%m-%d}
- **任务**：在夜间数据集 `night_2cls_clean_v2` 上完成 YOLO11n 可复现、无泄漏的分组五折 baseline，
  产出带留出测试指标的交付包
- **环境**：Ubuntu 22.04 / RTX 4090 24G / conda `torch2.4_cuda12.1` / PyTorch 2.4.1 + CUDA 12.1 / ultralytics 8.4.175
- **代码提交**：`{commit}`

---

## 1. 任务目标

夜间（NOD/RTTS/DAWN）两类目标检测的 baseline：与雾天交付包 v2 采用同一套协议和口径，
使两份交付可以横向比较。

---

## 2. 数据与协议

### 2.1 数据集

`/data/awod_data/processed/night_2cls_clean_v2/`：2780 图 / 2780 标签，
NOD 2669 + RTTS 夜间 79 + DAWN 夜间 32；框数 vehicle 3507、person 16624。

### 2.2 分组五折

`folds5_grouped_v1`：59 个来源组整组不拆（NOD 按 EXIF 拍摄日、RTTS 按视频源/城市、
DAWN 按天气序列、全部按感知哈希近重复）。审计结果：组跨折 = 0，每张图恰好被 test 一次（2780/2780）。

| fold | train | val | test |
|---|---|---|---|
| 1 | 1695 | 304 | 781 |
| 2 | 2047 | 393 | 340 |
| 3 | 1943 | 271 | 566 |
| 4 | 1989 | 271 | 520 |
| 5 | 1936 | 271 | 573 |

- `val.txt`：早停 / 选 checkpoint（`patience=30`）。
- `test.txt`：**只用于最终评测一次**，不参与任何调参。

### 2.3 训练配置（五折完全一致）

```
model=yolo11n.pt epochs=100 imgsz=640 batch=16 seed=0 lr0=0.001 patience=30
amp=True cache=True workers=8 device=0
```

---

## 3. 执行过程

### 3.1 环境核验

`nvidia-smi` RTX 4090 空闲；torch 2.4.1 + CUDA 12.1 可用；磁盘 65G 可用。
训练前校验：五折清单图片全部存在（`missing=0`），images/labels 2780 对 2780 配对正常。

### 3.2 训练脚本

`/data/fog_detect/scripts/run_night_fivefold.sh`（已入仓库）：逐折 train → test 评测一次 → 最后汇总。

### 3.3 时间线

| 阶段 | 时间 |
|---|---|
{tl_lines}
| 全部完成 | {all_done} |

### 3.4 汇总

`repro/summarize_test.py` 逐折加载 `best.pt`，在各自 test 上评测，采集总体与 per-class 指标，
计算 mean ± std（ddof=1）。

---

## 4. 结果（留出测试集）

### 4.1 逐折

| fold | P | R | mAP50 | mAP50-95 | {head_cols} |
|---|---|---|---|---|---|
{res_rows}

### 4.2 mean ± std

| 指标 | mean ± std | 区间 |
|---|---|---|
{agg_rows}
{class_agg}

**结论**：五折 mAP50 = {f(agg['mAP50']['mean'])} ± {f(agg['mAP50']['std'])}，
mAP50-95 = {f(agg['mAP50-95']['mean'])} ± {f(agg['mAP50-95']['std'])}。
标准差小说明该 baseline 稳定可信；夜间场景整体低于雾天 baseline（雾天 mAP50 0.8240 / mAP50-95 0.5452），
与夜间低照度、小目标密集的特点一致。

---

## 5. 交付物结构

见 README.md 第「目录」节；`SHA256SUMS` 可用 `sha256sum -c SHA256SUMS` 校验。

---

## 6. 遗留问题与下一步

1. **各折 test 规模不均衡（340~781）**：报告 mean±std 时需注明；建议补报 pooled mAP。
2. **person 弱于 vehicle**：可做小目标漏检分析。
3. **DAWN 未作纯外部测试**：本五折已含 32 张 DAWN 夜间图；严格跨域验证需另建 RTTS+NOD 五折、
   把 DAWN 整体留为外部测试。
4. **未做的研究方向**：去雾 D0–D2、小样本 R0–R3、学习曲线，均需另立实验。

---

## 附录：关键命令

```bash
conda activate torch2.4_cuda12.1
bash /data/fog_detect/scripts/run_night_fivefold.sh /data/awod_data/exp_night5fold_v2
python /data/fog_detect/scripts/summarize_test.py /data/awod_data/exp_night5fold_v2
```
"""
open(os.path.join(STAGE, "工作记录.md"), "w").write(log)

# ---------- SHA256SUMS + zip ----------
lines = []
for root, dirs, files in os.walk(STAGE):
    dirs.sort()
    for fn in sorted(files):
        p = os.path.join(root, fn)
        rel = os.path.relpath(p, STAGE)
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        lines.append(f"{h}  {rel}")
open(os.path.join(STAGE, "SHA256SUMS"), "w").write("\n".join(lines) + "\n")

if os.path.exists(ZIP):
    os.remove(ZIP)
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for root, dirs, files in os.walk(STAGE):
        dirs.sort()
        for fn in sorted(files):
            p = os.path.join(root, fn)
            z.write(p, os.path.join(NAME, os.path.relpath(p, STAGE)))

print("PACKAGE", STAGE)
print("ZIP", ZIP, round(os.path.getsize(ZIP) / 1e6, 1), "MB")
print(json.dumps({k: round(v["mean"], 4) for k, v in agg.items()}, ensure_ascii=False, indent=2))
