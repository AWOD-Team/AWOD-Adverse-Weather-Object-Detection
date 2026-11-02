# night_2cls_clean_v2 — 夜间（低照度）两类目标检测主数据集

夜间训练主数据集，结构对齐 `fog_2cls_clean_v1`。由三个已清洗来源合并，YOLO 两类。

## 规模

| 来源 | 图片 | 说明 |
| --- | --- | --- |
| NOD (Nikon D750) | 2669 | 夜间街道实拍；已按规则清洗（见 `nod_2cls_clean_v1`） |
| RTTS 夜间 | 79 | 按亮度筛选的真实雾霾/夜间图 |
| DAWN 夜间 | 32 | 按亮度筛选的 DAWN 夜间子集 |
| **合计** | **2780** | vehicle 3507 框、person 16624 框 |

- 类别：`0 vehicle`、`1 person`（`nc: 2`）
- 文件名带来源前缀：`nod__DSC_xxxx.JPG` / `rtts__*.png` / `dawn__*.jpg`

## 文件

| 文件 | 说明 |
| --- | --- |
| `images/`、`labels/` | YOLO 数据（硬链接，勿原地修改） |
| `classes.yaml` | 类别定义 |
| `all.txt` | 全部图片绝对路径 |
| `manifest.csv` | 逐图台账：来源、原名、亮度、框数、`flag` |
| `folds5_grouped_v1/` | **分组五折划分**（`foldN/data.yaml` 直接可用） |
| `SHA256SUMS` | 清单文件校验 |

## 五折划分要点

- **59 个来源组**，整组不拆：
  - NOD 按 **EXIF 拍摄日**（8 个拍摄日，2020-06-09 ~ 06-24）
  - RTTS 按 **hv 视频源 / 城市_来源**
  - DAWN 按天气序列名
  - 感知哈希近重复（4 对）强制同组
- **审计结果：组跨折 = 0，每张图恰好被 test 一次（2780/2780）**
- ⚠️ **各折 test 规模不均衡（340 ~ 781 张）**：NOD 单日拍摄量差异大且整天不可拆，
  这是分组划分的必然代价。`fold1` 的 test 是 781 张（最大的一天），其 train 相应偏小。
  报告指标时请注明这一点，或额外报告"合并全部 test 预测"的总体 mAP。

## 注意

1. `manifest.csv` 中 `flag=not_in_clean_set` 的 **30 张**（RTTS/DAWN 夜间）未经清洗，
   名单见 `night_2cls_clean_v1/not_in_clean.txt`，训练前建议剔除或复核。
2. 未做横向/纵向翻转之外的额外增强；划分与增强分开，避免影响可比性。

_生成：2026-10-25_
