# 边缘部署包 · 香橙派 Zero 3W（Allwinner H618）

恶劣天气车/人检测模型（YOLO11n，2 类：`0=vehicle` / `1=person`），面向**无 NPU 的 ARM CPU 设备**。

## 一、硬件现实（先看这个）

| 项目 | 香橙派 Zero 3W |
|---|---|
| SoC | Allwinner H618，4× Cortex-A53 @ 1.5 GHz |
| NPU | **没有**（对比：Orange Pi 5 的 RK3588 有 6 TOPS NPU） |
| 内存 | 1 / 2 / 4 GB 版本 |
| 结论 | 纯 CPU 推理。**640 分辨率做不到实时；320/416 可以做到可用帧率** |

模型本身只有 **2.6 M 参数 / 6.4 GFLOPs**，已经是 YOLO 系列最小的检测模型之一。

## 二、包内容

| 文件 | 说明 | 体积 |
|---|---|---|
| `yolo11n_aw_320.onnx` | **推荐首选**：320 输入，CPU 最快 | 10.5 MB |
| `yolo11n_aw_416.onnx` | 精度/速度折中，416 输入 | 10.6 MB |
| `aw320_ncnn_model/` | ncnn 格式 @320（ARM 上通常更快，需装 ncnn） | 10.4 MB |
| `aw640_ncnn_model/` | ncnn 格式 @640 | 10.5 MB |
| `bench_edge.py` | 推理基准 + 单图检测，**只依赖 onnxruntime**（不需要 torch） | — |

> ⚠️ 尝试过 int8 静态量化（3.15 MB），但量化后**置信度分支被破坏，检测结果全为空**，已从包中移除。
> 若确实需要 int8，需要带 QDQ 的量化训练或 QAT，不是一次校准能解决的。

## 三、服务器（Xeon 8368Q）实测，供换算参考

| 配置 | 实测延迟 | 帧率 |
|---|---|---|
| ONNX fp32 @320 | **11.9 ms** | 84 FPS |
| ONNX fp32 @416 | **19.3 ms** | 52 FPS |
| ONNX fp32 @640 | 29.5 ms | 34 FPS |
| ncnn @320（x86） | 71.5 ms | 14 FPS |
| ncnn @640（x86） | 147.5 ms | 6.8 FPS |

## 四、Zero 3W 上的预期（A53 约为 Xeon 单核的 1/8 ~ 1/15）

| 配置 | 预期延迟 | 预期帧率 |
|---|---|---|
| ONNX @320 | 100 ~ 180 ms | **5 ~ 10 FPS** |
| ONNX @416 | 160 ~ 300 ms | 3 ~ 6 FPS |
| ncnn @320（4 线程） | 80 ~ 200 ms | 5 ~ 12 FPS |
| ncnn @640 | 400 ~ 900 ms | 1 ~ 2.5 FPS |

> 这些是换算估计，**务必在设备上实测**（下面第 5 节有现成脚本），把实测数字记下来再决定用哪档。

## 五、在 Zero 3W 上跑

```bash
# 1) 依赖（arm64 均有预编译轮子，不需要 torch）
sudo apt update && sudo apt install -y python3-pip
pip3 install onnxruntime numpy opencv-python-headless

# 2) 把整个 edge_zero3w 目录拷到设备
scp -r edge_zero3w orangepi@<设备IP>:~/awod/

# 3) 基准（纯推理耗时）
cd ~/awod/edge_zero3w
python3 bench_edge.py --model yolo11n_aw_320.onnx --imgsz 320 --n 30

# 4) 单图检测（带后处理 + 画框，验证精度）
python3 bench_edge.py --model yolo11n_aw_320.onnx --imgsz 320 \
    --image demo.jpg --out demo_out.jpg

# 5) 限制线程数对比（A53 四核，试 1/2/4）
python3 bench_edge.py --model yolo11n_aw_320.onnx --imgsz 320 --n 30 --threads 4
```

`bench_edge.py` 的参数：`--model` `--imgsz` `--n` `--threads` `--image` `--out` `--conf`。

ncnn 路线（更快但依赖重）：

```bash
pip3 install ncnn ultralytics   # 会连带装 torch，约 1 GB，2G 内存版本不建议
python3 -c "from ultralytics import YOLO; m=YOLO('aw320_ncnn_model'); print(m('demo.jpg'))"
```

## 六、如果帧率不够，三条路

1. **降分辨率**：320 → 256（可让服务器再导出），帧率约再涨 1.5×
2. **换硬件**：Orange Pi 5（RK3588，6 TOPS NPU）≈ 640 int8 下 20~30 FPS，是这类任务真正合适的边缘板
3. **边缘协同**：Zero 3W 只做采集/推流，推理放在同网段的 4070 笔记本，局域网延迟 ~10 ms，整体仍在 100 ms 内

## 七、精度参考（部署模型，留出测试集）

| 指标 | 值 |
|---|---|
| mAP50 | 0.7703 |
| mAP50-95 | 0.5124 |
| vehicle mAP50-95 | 0.5742 |
| person mAP50-95 | 0.4506 |

训练覆盖 **雾天 + 夜间**；沙尘/雪/雨数据已就绪（DAWN，沙尘 210 / 雪 203 / 雨 188 张，标注完整），
若并入训练后重新导出，本包模型可直接替换。
