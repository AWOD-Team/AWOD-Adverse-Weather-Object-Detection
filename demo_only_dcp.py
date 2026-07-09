"""Demo: 纯图像去雾处理（不含目标检测）—— 对比 DCP / Retinex / Hybrid 三种方法."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from core.dehaze import DarkChannelPrior, Retinex, HybridDehazer


def imread_rgb(path: str) -> np.ndarray:
    """读取图像并转为 RGB."""
    img = cv2.imread(path)
    if img is None:
        print(f"[ERROR] 无法读取图像: {path}")
        sys.exit(1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
    parser = argparse.ArgumentParser(description="图像去雾 Demo — 对比三种算法")
    parser.add_argument("--image", type=str, required=True, help="输入图像路径")
    parser.add_argument("--method", type=str, default="all",
                        choices=["dcp", "retinex", "hybrid", "all"],
                        help="去雾方法 (默认 all=三种都跑)")
    parser.add_argument("--output", type=str, default=None,
                        help="输出图像路径 (默认: dehaze_result.jpg)")
    parser.add_argument("--omega", type=float, default=0.95,
                        help="DCP 去雾强度 (0~1, 默认 0.95)")
    parser.add_argument("--fusion-weight", type=float, default=0.5,
                        help="Hybrid 融合权重: DCP 占比 (默认 0.5)")
    args = parser.parse_args()

    # 读取图像
    img_rgb = imread_rgb(args.image)
    h, w = img_rgb.shape[:2]
    print(f"图像尺寸: {w}×{h}")

    # 初始化去雾器
    dcp = DarkChannelPrior(omega=args.omega)
    retinex = Retinex()
    hybrid = HybridDehazer(omega=args.omega, fusion_weight=args.fusion_weight)

    results = {}  # 存放各方法的结果

    if args.method in ("dcp", "all"):
        results["DCP (暗通道先验)"] = dcp.dehaze(img_rgb)
    if args.method in ("retinex", "all"):
        results["Retinex (MSRCR)"] = retinex.enhance(img_rgb)
    if args.method in ("hybrid", "all"):
        results["Hybrid (DCP+Retinex)"] = hybrid.dehaze(img_rgb)

    # 拼接显示：原始 + 各方法结果
    panels = [img_rgb] + list(results.values())
    titles = ["Original (原始)"] + list(results.keys())

    # 统一高度
    target_h = min(p.shape[0] for p in panels)
    panels_resized = []
    for p in panels:
        if p.shape[0] != target_h:
            new_w = int(p.shape[1] * target_h / p.shape[0])
            p = cv2.resize(p, (new_w, target_h))
        panels_resized.append(p)

    side_by_side = cv2.hconcat(panels_resized)
    side_by_side_bgr = cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR)

    # 在图像顶部添加标题文字
    font = cv2.FONT_HERSHEY_SIMPLEX
    x = 10
    for i, title in enumerate(titles):
        offset = sum(p.shape[1] for p in panels_resized[:i])
        cv2.putText(side_by_side_bgr, title, (offset + 10, 30),
                    font, 0.7, (0, 255, 0), 2)

    # 保存结果
    out_path = args.output or "dehaze_result.jpg"
    cv2.imwrite(out_path, side_by_side_bgr)
    print(f"结果已保存至: {out_path}")

    # 显示
    cv2.imshow("Dehaze Demo | " + " | ".join(titles), side_by_side_bgr)
    print("按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
