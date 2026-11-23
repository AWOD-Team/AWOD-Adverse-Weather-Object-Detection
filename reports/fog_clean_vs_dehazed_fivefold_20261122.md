# 雾天原图与去雾图五折实验对比

- 对比日期：2026-11-23
- 原图基线：`awod_yolo11n_fog5fold_v2`
- 去雾实验：`fog_2cls_dehazed_daylight_v1_20261122`
- 模型与评测协议：YOLO11n，640 输入，分组五折，`val` 选权重，`test` 五折留出评测

## 数据来源

| 实验 | 数据集 | 结果文件 |
| --- | --- | --- |
| 优化前原图基线 | `fog_2cls_clean_v1` | [`reports/fivefold_baseline_v2/test_summary.json`](fivefold_baseline_v2/test_summary.json) |
| 去雾后实验 | `fog_2cls_dehazed_daylight_v1` | [`runs/fog_2cls_dehazed_daylight_v1_20261122/test_summary.json`](../runs/fog_2cls_dehazed_daylight_v1_20261122/test_summary.json) |

两次实验使用相同的 2,681 张图片、相同文件名、相同五折划分和相同标签。三份五折清单
（`group_manifest.csv`、`split_manifest.csv`、`fold_summary.json`）的 SHA256 完全一致，
因此两版结果可以在同一测试集上做配对比较。

## 五折均值

括号内为标准差。变化单位为百分点（percentage point，pp）。

| 指标 | 优化前原图 | 去雾后 | 变化 |
| --- | ---: | ---: | ---: |
| Precision | 0.8705 (0.0087) | 0.8688 (0.0175) | -0.17 pp |
| Recall | 0.7300 (0.0205) | 0.7281 (0.0196) | -0.19 pp |
| mAP50 | 0.8240 (0.0152) | 0.8269 (0.0185) | +0.29 pp |
| mAP50-95 | 0.5452 (0.0136) | 0.5452 (0.0133) | +0.00 pp |
| vehicle mAP50 | 0.8793 (0.0063) | 0.8814 (0.0072) | +0.21 pp |
| vehicle mAP50-95 | 0.6018 (0.0078) | 0.6020 (0.0072) | +0.02 pp |
| person mAP50 | 0.7687 (0.0251) | 0.7725 (0.0307) | +0.38 pp |
| person mAP50-95 | 0.4887 (0.0199) | 0.4885 (0.0205) | -0.02 pp |

## 逐折 mAP50

| Fold | 优化前原图 | 去雾后 | 变化 |
| --- | ---: | ---: | ---: |
| 1 | 0.8370 | 0.8385 | +0.15 pp |
| 2 | 0.8268 | 0.8323 | +0.55 pp |
| 3 | 0.8149 | 0.8050 | -0.99 pp |
| 4 | 0.8027 | 0.8105 | +0.78 pp |
| 5 | 0.8385 | 0.8484 | +0.99 pp |

逐折 mAP50 差值的均值为 `+0.30 pp`，五折配对均值的 95% 置信区间约为
`[-0.68 pp, +1.27 pp]`，包含 0。去雾后 mAP50 在 5 折中有 4 折上涨、1 折下降，
但 mAP50-95、Precision 和 Recall 基本持平。

## 结论

本次去雾处理对五折整体结果表现为**接近持平、mAP50 略有正向趋势**，尚不足以证明稳定
提升。person mAP50 有轻微改善，但 person mAP50-95 没有改善；高 IoU 定位质量仍是主要
短板。

需要注意，两次训练并非完全相同的软件环境：原图基线使用 Ultralytics 8.4.175、
Python 3.10、`cache=True`、`workers=8`；去雾实验使用 Ultralytics 8.4.216、
Python 3.11、`cache=False`、`workers=4`。因此本对比适合作为当前工程记录，不能将全部
差异严格归因于去雾算法本身。

## 复现入口

```text
runs/fog_2cls_dehazed_daylight_v1_20261122/ANALYSIS.md
runs/fog_2cls_dehazed_daylight_v1_20261122/test_summary.json
runs/fog_2cls_dehazed_daylight_v1_20261122/test_summary.csv
reports/fivefold_baseline_v2/test_summary.json
reports/fivefold_baseline_v2/test_summary.csv
```
