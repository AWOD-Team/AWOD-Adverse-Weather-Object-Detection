# processed/ 数据集索引

> 原始数据从哪下载、下载后怎么变成这些目录：见 [`../数据集来源清单.md`](../数据集来源清单.md)。

所有数据集均为 YOLO 两类格式：**`0 vehicle` / `1 person`**（见各目录 `classes.yaml`）。

两条主线（均可直接用于训练）：

| 用途 | 主数据集 | 图片 | 五折 |
| --- | --- | --- | --- |
| **雾天** | `fog_2cls_clean_v1/` | 2681 | `folds5_grouped_v1/` |
| **夜间 / 低照度** | `night_2cls_clean_v2/` | 2780 | `folds5_grouped_v1/` |

## 目录

### 雾天

| 目录 | 含义 | 图片数 |
| --- | --- | --- |
| `fog_2cls_clean_v1/` | **雾天主数据集** = RTTS 2419 + DAWN 雾类 262，含分组五折 | 2681 |

### 夜间 / 低照度

| 目录 | 含义 | 图片数 |
| --- | --- | --- |
| `night_2cls_clean_v2/` | **夜间主数据集** = NOD 2669 + RTTS 夜间 79 + DAWN 夜间 32，含分组五折 | 2780 |
| `nod_2cls_clean_v1/` | NOD 清洗版（COCO→YOLO），含逐图剔除台账与复核 contact sheet | 2669 |
| `night_2cls_clean_v1/` | RTTS+DAWN 夜间合并的**中间产物**（已并入 v2，保留供追溯） | 111 |

### 原始 / 中间产物

| 目录 | 含义 | 图片数 |
| --- | --- | --- |
| `rtts/` | 原始 RTTS，VOC 格式（`Annotations/*.xml` + `JPEGImages/`） | 4322 |
| `rtts_2cls_v1/` | RTTS → YOLO 两类，**未清洗** | 4322 |
| `rtts_2cls_clean_v1/` | RTTS 清洗版（剔人工语义排除 1422、若干整段视频源、近重复、相邻帧） | 2419 |
| `dawn_2cls_v1/` | DAWN → YOLO 两类，**未清洗**，附 fog/rain/snow/sand 子集清单 | 1027 |
| `dawn_2cls_clean_v1/` | DAWN 清洗版（去近重复 105、完全重复 22、空标签 1） | 899 |
| `dawn_2cls_night_v1/` | DAWN 按平均亮度筛出的夜间候选 | 32 |
| `rtts_2cls_night_v1/` | RTTS 按平均亮度筛出的夜间候选（含未清洗图） | 79 |
| `rtts_2cls_night_screen_v1/` | RTTS 夜间/晨昏子集，已人工视觉筛选 | 23 |
| `dawn_2cls_dup.dup.json` / `.hash.json` | `dawn_2cls_v1` 的感知哈希去重结论与哈希值 | — |

## 血缘

```
雾天：
  rtts/ ─→ rtts_2cls_v1 ─→ rtts_2cls_clean_v1 ─┐
                                                ├─→ fog_2cls_clean_v1（+ folds5_grouped_v1）
  dawn_2cls_v1 ─→ dawn_2cls_clean_v1 ───────────┘

夜间：
  rtts_2cls_night_v1 ─┐
  dawn_2cls_night_v1 ─┴─→ night_2cls_clean_v1 ─┐
                                                ├─→ night_2cls_clean_v2（+ folds5_grouped_v1）
  NOD COCO JSON ─→ nod_2cls_clean_v1 ──────────┘
```

## 使用注意

1. **雾天训练/评测只用 `fog_2cls_clean_v1/`；夜间只用 `night_2cls_clean_v2/`**。
   两者的五折都在各自的 `folds5_grouped_v1/`，每折含 `data.yaml` + `train/val/test` 清单。
2. **不要把雾天和夜间混进同一个五折**——那是两个天气变量，混起来结论会互相污染。
3. `night_2cls_clean_v2/manifest.csv` 里 `flag=not_in_clean_set` 的 **30 张**（RTTS/DAWN 夜间）未经清洗，
   训练前建议剔除或复核，名单见 `night_2cls_clean_v1/not_in_clean.txt`。
4. `rtts_2cls_night_v1/`、`dawn_2cls_night_v1/` 是从**未清洗全量**按亮度筛的，含脏图，**不要直接训练**。
5. `rtts_2cls_v1/data.yaml` 与 `dawn_2cls_v1/data.yaml` **不可直接用于训练**（`train` 与 `val` 指向同一目录）。
6. `selection.csv` / `exclusion_ledger.csv` / `manifest.csv` 是筛选台账，`SHA256SUMS` 可校验；
   `images/`、`labels/` 多为硬链接，**不要原地修改**。

_最后更新：2026-10-25_
