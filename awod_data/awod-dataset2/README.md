# awod-dataset2 — 其余公开数据（DAWN / ExDark / OTS 清晰源 / 早期 417）

AWOD 项目的辅助数据集。**私有仓库**，因为 ExDark 等数据的再分发条款未逐项确认。

## 内容

| 目录 | 内容 | 数量 | 标注 |
| --- | --- | --- | --- |
| `DAWN/` | DAWN 恶劣天气：Fog / Rain / Sand / Snow | 1027 张 | 300+200+323+204 个 VOC xml + YOLO txt |
| `ExDark/` | 低照度检测数据集，5 类分目录 | 2929 张 | **无标签** |
| `clear/clear_images/` | RESIDE-OTS 的清晰原图（合成雾的源图） | 5949 张 | 无 |
| `legacy417/` | 早期 417 张 + 标签（经 md5 确认是 OTS 合成雾） | 209+208 张 | 419 个 txt（8 类） |

## 各数据集说明

### DAWN

- 全称：DAWN: Vehicle Detection in Adverse Weather Nature Dataset（Kenk & Hassaballah, 2020，arXiv:2008.05402）
- 下载：<https://www.kaggle.com/datasets/orvile/dawn-detection-in-adverse-weather-nature>
- 目录结构：`DAWN/<weather>/<weather>/`，内含 `*.jpg` + `<W>_PASCAL_VOC/*.xml` + `<W>_YOLO_darknet/*.txt`
- 核对：四类 jpg / xml / txt **数量完全一致**
- 已知：`Fog_YOLO_darknet/haze-012.txt` 为空 —— 对应 XML 本身 0 个目标，属正常
- 压缩包 MD5（原始下载校验）：
  - `Fog.zip`  `27c16c02f6cc64c2211c49a24eae3b8f`
  - `Rain.zip` `c367148491dd99dbc3770b3304eb7aa9`
  - `Sand.zip` `f2081f5a1ba80156842b78d8047e0a1d`
  - `Snow.zip` `f465236b46501efd215c6556b4a81705`

### ExDark

- 低照度/夜间目标检测数据集，5 类分目录（Bicycle / Bus / Car / Motorbike / People）
- ⚠️ **本仓库这份没有标签文件**，需要时请另行获取官方标注
- ⚠️ `Car/2015_02963.png`、`Motorbike/2015_06092.png` 扩展名是 `.png` 但内容是 JPEG（图片本身完好，PIL 可正常读取）

### clear/clear_images

RESIDE-OTS 合成流程的清晰原图，5949 张，与合成雾集（见 `awod-dataset3`）的 5949 个源图 id **完全对应**。

> 注：原目录里还有个 `clear_images.zip`（1.2 GB，含 8970 张，比解压目录多 3021 张），因为单文件超 GitHub 100 MB 限制未入库。

### legacy417

项目早期用的 417 张（`images_1` 209 + `images_2` 208），带 419 个 YOLO 标签（8 类）。

**这 417 张经 md5 逐字节比对，确认是 OTS 合成雾（A=0.95, β=0.2），不是真实雾。** 早期把它当真实雾用过，已从正式数据集中移除。保留在此仅为追溯。

## 核对结果

| 项 | 值 |
| --- | --- |
| 文件总数 | 12,795 |
| 体积 | 1.78 GB |
| 单文件 >100 MB | 0 |
| 图片/标注配对 | DAWN 四类全部一致；legacy417 图缺标 0 |

## 相关仓库

| 仓库 | 内容 |
| --- | --- |
| [awod-dataset1](https://github.com/AWOD-Team/awod-dataset1) | 真实雾天 RTTS |
| [awod-dataset2](https://github.com/AWOD-Team/awod-dataset2) | 本仓库 |
| [awod-dataset3](https://github.com/AWOD-Team/awod-dataset3) | 合成雾 + 不可再生台账与训练权重 |
| [AWOD-Adverse-Weather-Object-Detection](https://github.com/AWOD-Team/AWOD-Adverse-Weather-Object-Detection) | 代码与实验报告 |
