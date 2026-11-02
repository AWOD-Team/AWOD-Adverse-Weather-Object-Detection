# AWOD 夜间目标检测 · YOLO11n 分组五折 baseline 交付包

- **交付人**：于乘骁
- **构建时间**：2026-11-02 23:44
- **数据集**：`night_2cls_clean_v2`（NOD 2669 + RTTS 夜间 79 + DAWN 夜间 32，共 2780 张）
- **类别**：`0 = vehicle`，`1 = person`
- **模型**：YOLO11n（COCO 预训练 → 各折微调 100 epoch）
- **协议**：分组五折（59 个来源组整组不拆），train/val/test 三集合两两不重叠，
  每张图恰好当一次 test；val 仅用于早停与选 checkpoint，test 只在最后评测一次。
- **代码提交**：`a4adbac`

## 结果（留出测试集，5 折 mean ± std，std 为样本标准差 ddof=1）

| fold | P | R | mAP50 | mAP50-95 | vehicle mAP50-95 | person mAP50-95 |
|---|---|---|---|---|---|
| 1 | 0.8615 | 0.5440 | 0.6583 | 0.4346 | 0.4292 | 0.4400 |
| 2 | 0.7839 | 0.5224 | 0.6173 | 0.3728 | 0.3691 | 0.3766 |
| 3 | 0.8568 | 0.5648 | 0.6673 | 0.4501 | 0.4636 | 0.4367 |
| 4 | 0.8430 | 0.5490 | 0.6552 | 0.4193 | 0.3955 | 0.4431 |
| 5 | 0.7615 | 0.5398 | 0.6041 | 0.4010 | 0.3165 | 0.4855 |

| 指标 | mean ± std | 区间 |
|---|---|---|
| Precision | **0.8213 ± 0.0456** | 0.7615 ~ 0.8615 |
| Recall | **0.5440 ± 0.0153** | 0.5224 ~ 0.5648 |
| **mAP50** | **0.6404 ± 0.0279** | 0.6041 ~ 0.6673 |
| **mAP50-95** | **0.4156 ± 0.0300** | 0.3728 ~ 0.4501 |
| vehicle mAP50 | 0.6040 ± 0.0644 | 0.4963 ~ 0.6648 |
| vehicle mAP50-95 | 0.3948 ± 0.0564 | 0.3165 ~ 0.4636 |
| person mAP50 | 0.6769 ± 0.0392 | 0.6145 ~ 0.7120 |
| person mAP50-95 | 0.4364 ± 0.0389 | 0.3766 ~ 0.4855 |

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
awod_yolo11n_night5fold_v2/
├── README.md                     本文件
├── 工作记录.md                    完整过程记录
├── 数据集交付说明.md               数据集口径（原样附带）
├── configs/foldN_args.yaml       每折完整训练超参
├── results/
│   ├── test_summary.csv/.json    5 折 test 指标 + mean±std
│   └── foldN_results.csv         每折逐轮训练/验证指标
├── splits/                       固化划分（可复现）
│   ├── foldN/{data.yaml,train.txt,val.txt,test.txt}
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
