# awod-dataset1 — 真实雾天 RTTS 全量

AWOD 恶劣天气目标检测项目的**真实雾天**数据集，原始 RTTS（RESIDE 测试子集）。

## 内容

```
RTTS/
├── JPEGImages/          4322 张真实雾天道路图（PNG）
├── Annotations/         4322 个 VOC XML 标注（与图片一一对应）
└── ImageSets/Main/
    ├── test.txt         4322 行，全部图片清单
    └── demo_test.txt    6 行，演示子集
```

## 核对结果

| 项 | 值 |
| --- | --- |
| 图片 / 标注 | **4322 / 4322，双向 0 缺失** |
| 标注格式 | PASCAL VOC（`<xmin,ymin,xmax,ymax>` 绝对像素） |
| 原始类别 | car / person / bus / truck / bicycle / motorbike |
| 框总数 | 41,203 |
| difficult 标记 | 11,604（28%） |
| 小目标占比 | 37.1% |
| 图片体积 | 1.06 GB |

## 来源

- 数据集：**RTTS**，RESIDE 的标准真实测试子集（Real-world Task-driven Testing Set）
- 论文：*Benchmarking Single-Image Dehazing and Beyond*，Li et al.，IEEE TIP 2018，arXiv:1712.04143
- 发布页：<https://sites.google.com/view/reside-dehaze-datasets/reside-standard>
- 许可：学术研究用途；商用请回原发布页确认

## 在项目中的用途

本数据集是项目**唯一**的真实雾天标注数据。经人工语义筛选后保留 2419 张，与 DAWN 雾类 262 张合并成 `fog_2cls_clean_v1`（2681 张），完成分组五折交叉验证。

> 注：早期误认为是真实雾的 417 张（`images_1`/`images_2`）经 md5 比对实为 OTS 合成雾，不在此仓库，见 `awod-dataset2` 的 `legacy417/`。

## 转 YOLO 两类

项目的统一口径是 `0=vehicle, 1=person`：

| 原始类别 | 归并到 |
| --- | --- |
| car, bus, truck, bicycle, motorbike | vehicle |
| person | person |

转换脚本见主仓库 `scripts/rtts_to_yolo.py`。

## 相关仓库

| 仓库 | 内容 |
| --- | --- |
| [awod-dataset1](https://github.com/AWOD-Team/awod-dataset1) | 真实雾天 RTTS（本仓库） |
| [awod-dataset2](https://github.com/AWOD-Team/awod-dataset2) | DAWN / ExDark / OTS 清晰源 / 早期 417 |
| [awod-dataset3](https://github.com/AWOD-Team/awod-dataset3) | 合成雾 + 不可再生台账与训练权重 |
| [AWOD-Adverse-Weather-Object-Detection](https://github.com/AWOD-Team/AWOD-Adverse-Weather-Object-Detection) | 代码与实验报告 |
