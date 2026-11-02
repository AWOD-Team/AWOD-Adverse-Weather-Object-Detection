# AWOD 雾天目标检测 · 五折 baseline 工作记录

- **交付人：于乘骁**
- **日期：2026-10-14**
- **任务：修复上一版交付的问题，产出可复现、无泄漏、含分组五折交叉验证的 YOLO11n baseline 交付包**
- **环境：Ubuntu 22.04.4 / RTX 4090 24G / conda `torch2.4_cuda12.1` / PyTorch 2.4.1 + CUDA 12.1 / ultralytics 8.4.175**

---

## 1. 任务目标

上一版交付（v1，交付人周紫晗）经审查发现存在**测试集泄漏、没有五折、实验未跑完、不可复现**等问题。本次任务：

1. 定位并逐条修复 v1 的问题；
2. 在既有数据与分组五折基础上，完成 YOLO11n 的**完整五折 baseline 训练与留出测试**；
3. 汇总 5 折 mAP 的 mean ± std；
4. 重建一个**可复现、自包含、署名 于乘骁**的交付包。

---

## 2. 接手时的现状

### 2.1 仓库与数据

| 路径 | 内容 |
|---|---|
| `/data/fog_detect` | 项目 git 仓库（含 `scripts/`、`tools/`、`core/`、`train/` 等） |
| `/data/awod_data/processed/rtts_2cls_clean_v1` | RTTS 清洗版，2419 张，2 类 |
| `/data/awod_data/processed/fog_2cls_clean_v1` | 合并集，2681 张（RTTS 2419 + DAWN 262） |
| `/data/awod_data/processed/fog_2cls_clean_v1/folds5_grouped_v1` | **分组五折划分（已存在）** |
| `/data/交付包_awod_yolo11n_rtts2419_v1.zip` | v1 交付包（8 个文件） |

### 2.2 v1 实验目录 `/data/awod_data/exp_rtts_small`

包含 `lists/`（train_all、train_full、train_small_500、val、val_192、test）、多个 data yaml、`run.sh` / `run_all.sh` / `after.sh` / `eval_final.sh`，以及 `runs/detect/runs/full_ft` 一个已完成训练。

---

## 3. 对 v1 交付包的审查（问题清单）

### 3.1 严重问题

**P1 — 测试集泄漏（硬伤）**

三个清单实测关系（basename 集合运算）：

```
train_all.txt (2234) = train_full.txt (1863) + test.txt (371)   ← 完全相等
train_full ∩ test = 0，train_full ∪ test = train_all
1863 + 185(val) + 371(test) = 2419  ✅
```

本意应为 `train 1863 / val 185 / test 371`。但 v1 交付模型由 `data_all.yaml` 指定 `train: train_all.txt(2234)` 训练——**把整个 test 集 371 张也学进了模型**。`val(185)` 与 `train_all` 不重叠，故报告数字本身未被污染，但该模型**没有真正的留出测试集**，test.txt 一旦用于评测即为泄漏。

**P2 — 没有五折**

v1 仅单次 `2234 / 185` 划分，验证集只有 185 张。这正是导师指出的"验证集太小、指标震荡"问题，未按导师要求做五折。

**P3 — 实验未跑完**

`run.sh` 规划的 R1/R2/R3（小样本）与 FULL 无产物，`runs/` 下只有 `full_ft` 一个结果；`after.log` 为空，说明 `after.sh` 从未执行；`after.sh` 引用的 `data_dawn_fog.yaml` **不存在**，DAWN 跨域测试未做。

### 3.2 中等问题

**P4 — 不可复现**：zip 不含划分清单、不含数据集、不含跨域 yaml；`data_all.yaml` 用绝对路径指向 `/data/awod_data/exp_rtts_small`，换机即失效。

**P5 — 文件/口径不一致**：`val_192.txt` 名为 192 实为 185 行，且与 `val.txt` 内容完全相同（冗余）；README 称 AdamW，而 `args.yaml` 实为 `optimizer: auto`。

**P6 — 局限轻描淡写**：person mAP50-95 仅 0.484，README 一句"属数据特性"带过，无小目标/漏检细分。

---

## 4. 方案对齐

### 4.1 "五折 = 4:1" 的确认

导师口径：把数据分 5 份，每次 4 份训练、1 份测试，轮 5 次，报 mean ± std。

复核既有 `folds5_grouped_v1`，完全符合：

| 折 | train | val（早停用） | test（留出） | (train+val):test |
|---|---|---|---|---|
| fold1 | 1877 | 268 | 536 | 2145:536 = 4.00:1 |
| fold2 | 1876 | 268 | 537 | 2144:537 = 3.99:1 |
| fold3 | 1877 | 268 | 536 | 4.00:1 |
| fold4 | 1877 | 268 | 536 | 4.00:1 |
| fold5 | 1877 | 268 | 536 | 4.00:1 |

- 共 2681 张，分组规则：同 `hv*` 视频帧绑定 + 感知哈希距离 ≤ 4 的近重复绑定（含跨数据集）。
- `group_leaks = 0`；三集合两两不重叠；每张图恰好被测试一次。
- `val.txt` 仅用于选 checkpoint / 早停，`test.txt` 只做最终评测，天然避免"自选自评"，无需再固定 epoch。

### 4.2 数据口径决策

按用户确认，**直接沿用现有五折**（含 DAWN 262）。因此本版是"RTTS+DAWN 混合"五折，DAWN 未留作纯外部跨域集——此点已写入交付包局限。

---

## 5. 执行过程

### 5.1 环境核验

- `nvidia-smi`：RTX 4090 空闲；`torch2.4_cuda12.1` 环境 torch 2.4.1 + CUDA 12.1 + ultralytics 8.4.175。
- 验证既有 `fold*/data.yaml` 指向的图片路径有效，三个集合互不重叠。

### 5.2 冒烟测试

在 fold1 上跑 1 epoch，确认 train 1877 / val 268、标签与保存目录正常，GPU 显存占用约 4.5G。

### 5.3 训练脚本 `run_5fold.sh`

对每折：用 fold 的 `data.yaml` 训练 100 epoch（`patience=30`），再在 `test` 上评测一次。

```bash
CFG="imgsz=640 batch=16 seed=0 amp=True cache=True workers=8 device=0 exist_ok=True patience=30 lr0=0.001"
for k in 1 2 3 4 5; do
  yolo detect train model=/data/fog_detect/yolo11n.pt data=$FOLDS/fold$k/data.yaml epochs=100 project=$EXP/runs name=fold$k $CFG
  yolo detect val   model=$EXP/runs/fold$k/weights/best.pt data=$FOLDS/fold$k/data.yaml split=test project=$EXP/runs name=fold${k}_test exist_ok=True
done
```

### 5.4 训练时间线

| 阶段 | 开始 | 结束 |
|---|---|---|
| fold1 train | 10:54:19 | 11:05:48 |
| fold2 train | 11:05:57 | 11:17:25 |
| fold3 train | 11:17:34 | 11:29:01 |
| fold4 train | 11:29:11 | 11:40:39 |
| fold5 train | 11:40:48 | 11:52:13 |
| 全部完成 | — | 11:52:23 |

单折约 11.8 分钟（含 test 评测），五折合计约 58 分钟。

### 5.5 汇总

`summarize_test.py` 逐折加载 `best.pt`，在各自 `test` 集评测，采集总体与 per-class 指标，计算 mean ± std（ddof=1）。

---

## 6. 结果（留出测试集）

### 6.1 逐折

| fold | P | R | mAP50 | mAP50-95 | vehicle mAP50-95 | person mAP50-95 |
|---|---|---|---|---|---|---|
| 1 | 0.8701 | 0.7491 | 0.8370 | 0.5536 | 0.6042 | 0.5031 |
| 2 | 0.8707 | 0.7291 | 0.8268 | 0.5486 | 0.6019 | 0.4953 |
| 3 | 0.8656 | 0.7147 | 0.8149 | 0.5366 | 0.6009 | 0.4724 |
| 4 | 0.8615 | 0.7054 | 0.8027 | 0.5267 | 0.5900 | 0.4633 |
| 5 | 0.8846 | 0.7517 | 0.8385 | 0.5606 | 0.6118 | 0.5094 |

### 6.2 mean ± std

| 指标 | mean ± std | 区间 |
|---|---|---|
| Precision | 0.8705 ± 0.0087 | 0.8615 ~ 0.8846 |
| Recall | 0.7300 ± 0.0205 | 0.7054 ~ 0.7517 |
| **mAP50** | **0.8240 ± 0.0152** | 0.8027 ~ 0.8385 |
| **mAP50-95** | **0.5452 ± 0.0136** | 0.5267 ~ 0.5606 |
| vehicle mAP50 | 0.8793 ± 0.0063 | 0.8710 ~ 0.8865 |
| vehicle mAP50-95 | 0.6018 ± 0.0078 | 0.5900 ~ 0.6118 |
| person mAP50 | 0.7687 ± 0.0251 | 0.7344 ~ 0.7905 |
| person mAP50-95 | 0.4887 ± 0.0199 | 0.4633 ~ 0.5094 |

**结论**：五折 mAP50 标准差仅 0.015，指标高度稳定，说明该 baseline 可信；车辆明显优于行人。

---

## 7. 交付物结构

`交付包_awod_yolo11n_fog5fold_v2_于乘骁.zip`

```
awod_yolo11n_fog5fold_v2/
├── README.md                      说明与结果（署名 于乘骁）
├── 工作记录.md                     本文件
├── configs/foldN_args.yaml        每折完整训练配置
├── results/
│   ├── test_summary.csv/.json     5 折 test 指标 + mean±std
│   └── foldN_results.csv          每折逐轮指标
├── splits/                        固化划分（完整可复现）
│   ├── foldN/{data.yaml,train.txt,val.txt,test.txt}
│   ├── fold_summary.json          每折统计与均衡度
│   ├── group_manifest.csv         每张图所属组 / 在哪折当 test
│   └── split_manifest.csv         完整划分清单
├── weights/foldN_best.pt          5 折最优权重
├── curves/foldN/                  训练曲线、PR 曲线、混淆矩阵
├── repro/                         训练+汇总脚本、环境信息
└── SHA256SUMS                     完整性校验
```

---

## 8. 与 v1 的对照

| 维度 | v1 | v2（本次） |
|---|---|---|
| 划分 | 单次 2234/185 | 分组五折，train:test = 4:1 |
| 测试集 | test(371) 泄漏进训练集 | 每折 test 全程留出 |
| 指标 | 单次 mAP50 0.827 / mAP50-95 0.546 | 5 折 0.8240 ± 0.0152 / 0.5452 ± 0.0136 |
| 实验完整性 | 仅 1 个 full_ft，跨域缺失 | 5 折训练 + 5 折留出测试 + 汇总 |
| 可复现 | 无清单，绝对路径 | 清单/配置/脚本/环境齐全 |
| 署名 | 周紫晗 | 于乘骁 |

---

## 9. 遗留问题与下一步

1. **DAWN 未作纯外部测试**：本版五折含 262 张 DAWN；如需严格跨域验证，应另建 RTTS-only 五折，将 DAWN 整体留作外部测试。
2. **person 偏弱**（mAP50-95 0.49）：person 框少且小目标多，后续可做小目标漏检分析。
3. **导师其余方向未做**：小样本（R0–R3）、合成雾增强、去雾 vs 直接检测，均待推进。

---

## 附录：关键命令

```bash
# 环境
conda activate torch2.4_cuda12.1

# 五折训练 + 留出测试
bash /data/awod_data/exp_fog5fold_v2/run_5fold.sh

# 汇总 mean±std
python /data/awod_data/exp_fog5fold_v2/summarize_test.py
```
