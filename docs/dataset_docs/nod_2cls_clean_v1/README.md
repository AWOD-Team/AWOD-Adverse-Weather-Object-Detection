# nod_2cls_clean_v1 — NOD 夜间数据集（已清洗，YOLO 两类）

来源：NOD (Night Object Detection, BMVC 2021) 的 Nikon D750 子集 COCO 标注，
图片引用 `/data/awod_data/Nikon750_1280`（硬链接，未复制像素）。

## 清洗规则（阈值可通过脚本参数调整）

| 规则 | 阈值 | 剔除数 |
| --- | --- | --- |
| 亮度过低（肉眼难分辨） | `mean_brightness < 10` | 739 |
| 目标数量太少 | `car + person 框数 < 2` | 587 |
| 疑似自拍/人像摆拍 | `person 最大框面积占比 ≥ 0.20` 且 `框数 ≤ 3` | 11 |

另：**bicycle 类整体丢弃**（与 RTTS/DAWN 的两类口径一致），共 5511 个框未保留。

## 规模

| 项 | 值 |
| --- | --- |
| 原始标注图片 | 4006 |
| **保留** | **2669**（66.6%） |
| 剔除 | 1337 |
| 保留框数 | vehicle(car) 2886、person 16406 |
| 保留图亮度 | 10.0 ~ 中位 26.4（p5=11.5） |
| 框数分布 | 2 框 451、3 框 352、≥4 框 1866 |

## 文件

| 文件 | 说明 |
| --- | --- |
| `images/`、`labels/` | YOLO 数据，文件名沿用 NOD 原名（如 `DSC_4705.JPG` / `.txt`） |
| `classes.yaml` | `0 vehicle`、`1 person`，`nc: 2` |
| `all.txt` | 保留图片的绝对路径 |
| `selection.csv` | 保留图片台账（亮度、各类框数、最大 person 面积） |
| `exclusion_ledger.csv` | **逐图剔除台账**（含 `reason`） |
| `review/` | 剔除样本与保留样本的 contact sheet，供人工复核 |
| `SHA256SUMS` | 清单类文件校验 |

## 说明

1. **未划分 train/val/test**。NOD 是连续拍摄（`DSC_xxxx` 编号相邻，帧间高度相似），
   划分必须按拍摄 session/时间分组，不能随机分。原始 split 记录在 `selection.csv` 的 `split` 列，仅供追溯。
2. 清洗是**基于规则**的自动筛选；`review/` 里是抽样的 contact sheet。
   若发现漏网的摆拍/无意义图片，把文件名给我即可追加剔除。
3. **格式转换**：NOD 原始是 COCO JSON（`[x,y,w,h]` 绝对像素），本数据集已转为 YOLO
   （归一化 `cx,cy,w,h`）。全量逐框校验通过（最大误差 < 5e-7）；唯一差异是
   `test/DSC_0674.JPG` 中 1 个 `h=0` 的退化标注被正确跳过。
4. `images/`、`labels/` 为硬链接，**不要原地修改**。

_生成：2026-10-25，脚本：`/data/fog_detect/scripts/clean_nod.py`_
