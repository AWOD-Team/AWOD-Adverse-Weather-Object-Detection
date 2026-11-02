# 分组五折划分（night）

- 共 2780 张、59 个来源组；组内图片**整组进同一折**，绝不拆分。
- 每组：`test` = 留出折（仅最终评测）；`val` = 从其余组中按组挑约 12.5%（早停/选 checkpoint）；
  剩余为 `train`。
- 数字标定期望：**epoch 与超参只能用 val 决定，test 只在最后评测一次。**

## 各组规模

| fold | train | val | test | test 来源 |
| --- | --- | --- | --- | --- |
| 1 | 1695 | 304 | 781 | nod 781 |
| 2 | 2047 | 393 | 340 | nod 271 + rtts 39 + dawn 30 |
| 3 | 1943 | 271 | 566 | nod 566 |
| 4 | 1989 | 271 | 520 | nod 496 + rtts 22 + dawn 2 |
| 5 | 1936 | 271 | 573 | nod 555 + rtts 18 |

## 审计

- 组跨折：**0**
- 每张图被 test 的次数：恰好 1（2780/2780）
- 近重复对（4 对）已并入同组

## 用法

```bash
yolo detect train model=yolo11n.pt data=$PWD/fold1/data.yaml epochs=100 imgsz=640 batch=16
yolo detect val   model=runs/.../weights/best.pt data=$PWD/fold1/data.yaml split=test
```

_生成：2026-10-25_
