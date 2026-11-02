# AWOD 雾天目标检测 · 五折交叉验证交付包 v2

- **交付人：于乘骁**
- 生成时间：2026-10-14
- 任务：雾天 vehicle / person 两类目标检测 baseline（分组五折）

---

## 一、本版相对 v1 的修正

v1（交付人周紫晗）的问题与本版修正：

| # | v1 问题 | v2 修正 |
|---|---|---|
| 1 | **测试集泄漏**：`train_all.txt(2234)` = `train_full(1863)` + `test(371)`，训练时把 test 图学进去了 | 改用分组五折，每折 `test.txt` 为**从未参与训练**的留出折 |
| 2 | **没有五折**：只有单次 2234/185 划分，验证集仅 185 张 | 分组五折，train:test = 4:1，报告 5 折 mean ± std |
| 3 | **实验没跑完**：规划里的小样本、跨域等实验无产物 | 本包含 5 折完整训练、留出测试与汇总 |
| 4 | **不可复现**：交付包不含划分清单，yaml 用绝对路径 | 本包内含全部划分清单、每折 data.yaml、训练/评测脚本、环境信息 |
| 5 | **命名/口径不一致**：`val_192.txt` 实为 185 行、与 `val.txt` 重复；README 称 AdamW 而 args 为 auto | 清单与实际一致，指标口径统一，配置以 `args.yaml` 为准 |

---

## 二、数据与划分

| 项 | 内容 |
|---|---|
| 数据 | `fog_2cls_clean_v1`：2681 张（RTTS 2419 + DAWN 262），2 类 vehicle/person |
| 划分 | 分组五折（`folds5_grouped_v1`），seed 42 |
| 分组规则 | 同 `hv*` 视频帧绑定；感知哈希距离 ≤ 4 的近重复绑定（含跨数据集） |
| 泄漏 | `group_leaks = 0`（同组图不跨折） |
| 每折 | train 1877 / val 268 / test 536（fold2 为 1876/268/537） |
| 比例 | (train+val) : test = **4 : 1**；每张图恰好被测试一次 |

- `val.txt` 仅用于选 checkpoint / 早停；`test.txt` 只做最终评测，训练全程不可见。

---

## 三、训练配置

| 参数 | 值 |
|---|---|
| 模型 | YOLO11n（COCO 预训练 `yolo11n.pt`） |
| 输入 | 640 × 640 |
| Epoch | 100，`patience=30` |
| Batch | 16 |
| 优化器 | Ultralytics `optimizer=auto` → AdamW(lr=0.001667) |
| 初始学习率 | 0.001 |
| 随机种子 | 0 |
| AMP | 开启 |
| 环境 | Ubuntu 22.04 / RTX 4090 24G / PyTorch 2.4.1 + CUDA 12.1 / ultralytics 8.4.175 |

每折独立训练，checkpoint 取 `val` 上最优；训练完成后在 `test` 上评测一次。

---

## 四、结果（留出测试集，5 折 mean ± std）

| 指标 | mean ± std | 最小 ~ 最大 |
|---|---|---|
| Precision | 0.8705 ± 0.0087 | 0.8615 ~ 0.8846 |
| Recall | 0.7300 ± 0.0205 | 0.7054 ~ 0.7517 |
| mAP50 | **0.8240 ± 0.0152** | 0.8027 ~ 0.8385 |
| mAP50-95 | **0.5452 ± 0.0136** | 0.5267 ~ 0.5606 |
| vehicle mAP50 | 0.8793 ± 0.0063 | 0.8710 ~ 0.8865 |
| vehicle mAP50-95 | 0.6018 ± 0.0078 | 0.5900 ~ 0.6118 |
| person mAP50 | 0.7687 ± 0.0251 | 0.7344 ~ 0.7905 |
| person mAP50-95 | 0.4887 ± 0.0199 | 0.4633 ~ 0.5094 |

逐折明细见 `results/test_summary.csv` 与 `results/test_summary.json`。

结论：五折 mAP50 标准差仅 0.015，指标高度稳定；车辆明显优于行人。

---

## 五、已知局限（如实写明）

1. **DAWN 参与了训练**：本版按现有 `folds5_grouped_v1` 五折，其中含 262 张 DAWN 雾图。
   因此本结果**不是**纯 RTTS 域内结果，也**未**保留 DAWN 作为独立跨域测试集。
   如需跨域验证，应另建 RTTS-only 五折并将 DAWN 整体留作外部测试。
2. **person 偏弱**：person 框少且小目标多，mAP50-95 仅 0.49，属数据特性。
3. **分组为启发式**：基于文件名与感知哈希，无法证明所有视频序列都已完全分离（已验证 `group_leaks=0`）。
4. 仅做了雾天、仅两类、仅 YOLO11n；未涉及雨/雪/夜间、注意力模块、去雾与合成雾增强。

---

## 六、复现

数据划分已固化在本包 `splits/`，直接使用每折的 `data.yaml`：

```bash
conda activate torch2.4_cuda12.1
# 训练单折（以 fold1 为例）
yolo detect train model=yolo11n.pt data=splits/fold1/data.yaml \
  epochs=100 imgsz=640 batch=16 seed=0 lr0=0.001 patience=30 \
  amp=True cache=True workers=8 device=0 project=runs name=fold1
# 留出测试集评测
yolo detect val model=runs/fold1/weights/best.pt data=splits/fold1/data.yaml split=test
```

或直接运行 `repro/run_5fold.sh`（连续跑 5 折）与 `repro/summarize_test.py`（汇总 mean±std）。

---

## 七、文件说明

```
awod_yolo11n_fog5fold_v2/
├── README.md                      本文件（署名 于乘骁）
├── configs/foldN_args.yaml        每折完整训练配置（复现用）
├── results/
│   ├── test_summary.csv/.json     5 折 test 指标与 mean±std 汇总
│   └── foldN_results.csv          每折逐轮指标
├── splits/                        固化数据划分（完整可复现）
│   ├── foldN/{data.yaml,train.txt,val.txt,test.txt}
│   ├── fold_summary.json          每折统计与均衡度
│   ├── group_manifest.csv         每张图所属组 / 在哪折当 test
│   └── split_manifest.csv         完整划分清单
├── weights/foldN_best.pt          5 折最优权重（推理可直接用）
├── curves/foldN/                  训练曲线、PR 曲线、混淆矩阵
├── repro/                         训练 + 汇总脚本、环境信息
└── SHA256SUMS                     完整性校验
```

推理示例：

```python
from ultralytics import YOLO
model = YOLO("weights/fold1_best.pt")
model.predict("fog.jpg", conf=0.25, save=True)
```
