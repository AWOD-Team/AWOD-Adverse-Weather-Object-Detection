"""Demo: 视频流实时去雾处理 —— 支持摄像头或视频文件."""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from core.dehaze import DarkChannelPrior, Retinex, HybridDehazer


def put_text(img: np.ndarray, text: str, org: tuple, color=(0, 255, 0)):
    """在图像上叠加文字（带黑色描边，提高可读性）."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def main():
    parser = argparse.ArgumentParser(description="视频流实时去雾 Demo")
    parser.add_argument("--source", type=str, default="0",
                        help="视频源: 摄像头编号(默认0) 或 视频文件路径")
    parser.add_argument("--method", type=str, default="dcp",
                        choices=["dcp", "retinex", "hybrid"],
                        help="去雾方法 (默认 dcp)")
    parser.add_argument("--omega", type=float, default=0.95,
                        help="DCP 去雾强度 (0~1, 默认 0.95)")
    parser.add_argument("--fusion-weight", type=float, default=0.5,
                        help="Hybrid 融合权重: DCP 占比 (默认 0.5)")
    parser.add_argument("--threads", type=int, default=2,
                        help="Hybrid 并行线程数 (默认 2, 1=顺序)")
    parser.add_argument("--width", type=int, default=640,
                        help="处理宽度 (默认 640, 越小越快)")
    parser.add_argument("--output", type=str, default=None,
                        help="可选: 将处理结果保存为视频文件")
    args = parser.parse_args()

    # ---- 打开视频源 ----
    source = args.source
    is_camera = source.isdigit()

    if is_camera:
        source = int(source)
        # Windows 上优先用 DirectShow 后端，兼容性更好
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if not cap.isOpened():
            # 回退到默认后端
            cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[ERROR] 无法打开摄像头 #{source}，请尝试 --source 0")
            sys.exit(1)
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[ERROR] 无法打开视频文件: {args.source}")
            sys.exit(1)

    # ---- 初始化去雾器 ----
    method_name = {
        "dcp": "DCP (暗通道先验)",
        "retinex": "Retinex (MSRCR)",
        "hybrid": "Hybrid (DCP+Retinex)",
    }[args.method]

    dehazer = {
        "dcp": DarkChannelPrior(omega=args.omega),
        "retinex": Retinex(),
        "hybrid": HybridDehazer(omega=args.omega, fusion_weight=args.fusion_weight,
                                num_workers=args.threads),
    }[args.method]

    print(f"视频源: {args.source}  |  方法: {method_name}  |  omega={args.omega}")
    print("按键: [q]退出  [s]截图保存  [1/2/3]切换方法(dcp/retinex/hybrid)")

    # ---- 可选: 视频写入器 ----
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, 20.0, (args.width * 2, args.width))

    # ---- 主循环 ----
    fps_times = []  # 用于计算实时帧率
    current_method = args.method

    while True:
        t0 = time.time()

        ret, frame = cap.read()
        if not ret:
            print("视频流结束或中断.")
            break

        # 缩放到指定宽度以加速处理
        h0, w0 = frame.shape[:2]
        target_h = int(h0 * args.width / w0)
        frame = cv2.resize(frame, (args.width, target_h))

        # BGR → RGB → 去雾 → BGR
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if current_method == "retinex":
            dehazed_rgb = dehazer.enhance(frame_rgb)
        else:
            dehazed_rgb = dehazer.dehaze(frame_rgb)
        dehazed_bgr = cv2.cvtColor(dehazed_rgb, cv2.COLOR_RGB2BGR)

        elapsed = (time.time() - t0) * 1000  # ms

        # 帧率计算（滑动窗口平均）
        fps_times.append(time.time())
        fps_times = [t for t in fps_times if time.time() - t < 2.0]
        fps = len(fps_times) / 2.0 if len(fps_times) > 1 else 0

        # ---- 拼接显示 ----
        put_text(frame, "Original", (10, 25))
        put_text(dehazed_bgr, method_name, (10, 25))

        side_by_side = cv2.hconcat([frame, dehazed_bgr])

        # 顶部叠加信息栏
        info_bar = np.zeros((40, side_by_side.shape[1], 3), dtype=np.uint8)
        put_text(info_bar, f"FPS: {fps:.1f}  |  {elapsed:.0f}ms/frame", (10, 28), (255, 255, 255))
        put_text(info_bar, "[Q]uit  [S]ave  [1]DCP  [2]Retinex  [3]Hybrid",
                 (side_by_side.shape[1] - 480, 28), (200, 200, 200))

        display = cv2.vconcat([info_bar, side_by_side])

        cv2.imshow("Dehaze Video Stream", display)

        # 写入输出视频
        if writer:
            writer.write(side_by_side)

        # ---- 键盘控制 ----
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            stamp = time.strftime("%Y%m%d_%H%M%S")
            fname = f"screenshot_{stamp}.jpg"
            cv2.imwrite(fname, side_by_side)
            print(f"截图已保存: {fname}")
        elif key == ord("1"):
            current_method = "dcp"
            dehazer = DarkChannelPrior(omega=args.omega)
            method_name = "DCP (暗通道先验)"
            print(">>> 切换到: DCP")
        elif key == ord("2"):
            current_method = "retinex"
            dehazer = Retinex()
            method_name = "Retinex (MSRCR)"
            print(">>> 切换到: Retinex")
        elif key == ord("3"):
            current_method = "hybrid"
            dehazer = HybridDehazer(omega=args.omega, fusion_weight=args.fusion_weight,
                                    num_workers=args.threads)
            method_name = "Hybrid (DCP+Retinex)"
            print(">>> 切换到: Hybrid")

    # ---- 清理 ----
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("结束.")


if __name__ == "__main__":
    main()
