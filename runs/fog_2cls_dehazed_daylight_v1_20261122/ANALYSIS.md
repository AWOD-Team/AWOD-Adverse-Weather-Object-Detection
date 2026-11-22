# fog_2cls_dehazed_daylight_v1 五折训练分析

## 1. 本次实验口径

- 训练时间：2026-11-22
- 数据根目录：`D:/dachuang/AWOD-Adverse-Weather-Object-Detection/awod_data/processed/fog_2cls_dehazed_daylight_v1/folds5_grouped_v1`
- 模型：YOLO11n，仅使用 `yolo11n.pt` 预训练权重初始化
- 类别：`0: vehicle`、`1: person`
- 输入尺寸：640
- 训练轮数：100，`patience=30`
- Batch：16
- Seed：0
- AMP：开启
- 设备：NVIDIA GeForce RTX 5060 Laptop GPU
- 训练环境：Python 3.11.18、PyTorch 2.14.1、CUDA 13.4、Ultralytics 8.4.216
- 模型选择：每折只用 `val` 选择 `best.pt`
- 最终评测：每折训练完成后，`test` 仅评测一次

完整命令：

```powershell
D:\conda\envs\yolo-env\python.exe -u scripts\train_dehazed_fivefold.py `
  --data-root D:\dachuang\AWOD-Adverse-Weather-Object-Detection\awod_data\processed\fog_2cls_dehazed_daylight_v1\folds5_grouped_v1 `
  --model D:\dachuang\AWOD-Adverse-Weather-Object-Detection\yolo11n.pt `
  --output D:\dachuang\AWOD-Adverse-Weather-Object-Detection\runs\fog_2cls_dehazed_daylight_v1_20261122 `
  --epochs 100 --imgsz 640 --batch 16 --seed 0 --lr0 0.001 `
  --patience 30 --workers 4 --device 0
```

注意：当前 Ultralytics 对 `optimizer=auto` 自动选择了 AdamW，初始学习率为 `0.001667`，传入的 `lr0=0.001` 被自动优化器逻辑覆盖。该行为在五折中一致。

## 2. 数据完整性

- 图片总数：2,681
- 分组数：2,468
- 视频来源：45 个，共 240 帧
- 分组泄漏：0
- 每折训练/验证/测试：约 1,877 / 268 / 536 张
- 分割条目校验：13,405 个，全部可以解析到图片和标签
- 语义比例：vehicle 目标框约 18,605-18,622 个，person 目标框约 3,758-3,763 个，约为 `4.95:1`

各折规模：

| Fold | Train | Val | Test |
| --- | ---: | ---: | ---: |
| 1 | 1,877 | 268 | 536 |
| 2 | 1,876 | 268 | 537 |
| 3 | 1,877 | 268 | 536 |
| 4 | 1,877 | 268 | 536 |
| 5 | 1,877 | 268 | 536 |

## 3. 五折 test 结果

| Fold | Precision | Recall | mAP50 | mAP50-95 | person mAP50 | vehicle mAP50 | person mAP50-95 | vehicle mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.8855 | 0.7390 | 0.8385 | 0.5500 | 0.7911 | 0.8858 | 0.5003 | 0.5997 |
| 2 | 0.8596 | 0.7371 | 0.8323 | 0.5499 | 0.7859 | 0.8787 | 0.4974 | 0.6025 |
| 3 | 0.8758 | 0.6982 | 0.8050 | 0.5340 | 0.7335 | 0.8764 | 0.4684 | 0.5995 |
| 4 | 0.8427 | 0.7191 | 0.8105 | 0.5298 | 0.7468 | 0.8742 | 0.4652 | 0.5944 |
| 5 | 0.8804 | 0.7472 | 0.8484 | 0.5625 | 0.8051 | 0.8917 | 0.5111 | 0.6138 |
| **Mean ± Std** | **0.8688 ± 0.0175** | **0.7281 ± 0.0196** | **0.8269 ± 0.0185** | **0.5452 ± 0.0133** | **0.7725 ± 0.0307** | **0.8814 ± 0.0072** | **0.4885 ± 0.0205** | **0.6020 ± 0.0072** |

### 最佳验证轮次与 val/test 差异

| Fold | 最佳 epoch | 最佳 val mAP50 | 最佳 val mAP50-95 | test-val mAP50 | test-val mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 90 | 0.8504 | 0.5732 | -0.0119 | -0.0232 |
| 2 | 89 | 0.8506 | 0.5608 | -0.0183 | -0.0109 |
| 3 | 83 | 0.8320 | 0.5530 | -0.0270 | -0.0190 |
| 4 | 99 | 0.8189 | 0.5409 | -0.0084 | -0.0110 |
| 5 | 90 | 0.8363 | 0.5504 | +0.0121 | +0.0121 |
| **Mean** | **90.2** | - | - | **-0.0107** | **-0.0104** |

val 到 test 的平均下降约 1 个百分点，且第 5 折 test 还略高于 val。没有观察到明显的验证集过拟合或 test 异常崩塌。

## 4. 结果分析

### 4.1 五折整体稳定，可以作为当前数据上的可信基线

五折 test mAP50 为 `0.8269 ± 0.0185`，mAP50-95 为 `0.5452 ± 0.0133`。mAP50 的变异系数约为 2.2%，说明不同来源组重新分折后，整体检测能力没有出现数量级变化。逐折结果落在 0.8050-0.8484 之间，Fold 3 和 Fold 4 较弱。

### 4.2 主要瓶颈是 recall，而不是单纯的 precision

五折平均 Precision 为 0.8688，但 Recall 只有 0.7281，差距约 0.141。说明模型漏检多于误检。实际部署时如果优先减少漏检，应基于独立验证集重新校准置信度阈值，并检查 person 和远处小目标的漏检，而不能继续只参考默认阈值下的 mAP。

### 4.3 高 IoU 定位质量仍有明显提升空间

mAP50 与 mAP50-95 相差约 0.282。模型对较宽松 IoU 的检测已经较好，但在 IoU 0.5-0.95 的严格区间下降明显，说明边界框定位和置信度排序仍是主要限制之一。

### 4.4 person 是稳定性和精度的共同短板

- vehicle mAP50 为 `0.8814 ± 0.0072`，person 为 `0.7725 ± 0.0307`。
- vehicle mAP50-95 为 `0.6020 ± 0.0072`，person 为 `0.4885 ± 0.0205`。
- person 的平均 mAP50 比 vehicle 低约 10.9 个百分点。
- person 的标准差约为 vehicle 的 4.3 倍，说明 person 结果更依赖具体测试折。
- test 中每折 person 框约 1,071-1,076 个，而 vehicle 框约 5,308-5,327 个，类别数量相差约 5 倍。

Fold 3、Fold 4 的下降也主要来自 person：两折 person mAP50 分别为 0.7335 和 0.7468，明显低于 Fold 5 的 0.8051。下一步应优先分析 person 的小目标、遮挡和夜间/低照度边界样本。

### 4.5 没有明显过拟合，但训练末端收益趋缓

各折最后 10 轮的 train box loss 约为 0.94-0.95，val box loss 约为 1.08-1.14。最终 val box loss 相对各自最低值平均只高约 0.0036，没有出现持续、明显的验证损失反弹。最佳 epoch 平均为 90.2，说明将训练轮数设为 100 基本合理；Fold 4 的最佳 epoch 为 99，但其余四折在 83-90 轮附近已达到最佳。

### 4.6 结论边界

本次是去雾数据 `fog_2cls_dehazed_daylight_v1` 的干净重训结果。昨天的实验输出已删除且未参与本次分析，因此本报告不声称相对任何旧实验“提升了多少”。要判断去雾是否优于原雾图，后续必须在同一套分组五折和同一评测协议下重新训练 `fog_2cls_clean_v1`，并比较 test 的五折 mean ± std。

## 5. 建议的下一组实验

1. 在完全相同五折上训练原雾图 `fog_2cls_clean_v1`，建立去雾前后唯一变量对比。
2. 针对 person 做误差拆分：按目标框面积、遮挡程度和亮度统计漏检率，确认是分辨率、类别不平衡还是场景难度导致。
3. 测试 768 或 960 输入尺寸，重点观察 person mAP50-95 与推理延迟，而不是只看总 mAP。
4. 在验证集上扫描置信度阈值，给出 Precision-Recall 工作点；当前 recall 偏低，默认阈值未必适合部署。
5. 对 Fold 3、Fold 4 的 test 来源组做困难样本审计，避免把折间难度差异误判为训练波动。

## 6. 产物位置

- 汇总：`test_summary.csv`、`test_summary.json`
- 每折权重：`train/foldN/weights/best.pt`
- 每折 test 指标：`metrics/foldN.json`
- 每折训练曲线：`train/foldN/results.csv`
- 每折图表：`train/foldN/`、`test/foldN/`
