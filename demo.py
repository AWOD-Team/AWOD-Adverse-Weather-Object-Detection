import argparse
import sys
from pathlib import Path

import cv2

from core.pipeline import AWODPipeline
from core.dehaze.hybrid_dehaze import HybridDehazer


def main():
    parser = argparse.ArgumentParser(description="AWOD: Adverse Weather Object Detection")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="YOLO model path or name")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--fusion-weight", type=float, default=0.5,
                        help="DCP(1.0) vs Retinex(0.0) fusion weight")
    parser.add_argument("--no-dehaze", action="store_true", help="Skip dehazing (baseline test)")
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"[ERROR] Cannot read image: {args.image}")
        sys.exit(1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    dehazer = HybridDehazer(fusion_weight=args.fusion_weight)
    pipeline = AWODPipeline(
        yolo_model_path=args.model,
        dehazer=dehazer,
        enable_dehaze=not args.no_dehaze,
    )

    raw_drawn, enhanced_drawn, timing = pipeline.process_and_draw(img)

    print(f"Detections: {len(pipeline.process(img)['detections'])} objects")
    print(f"Dehaze time: {timing['dehaze_ms']:.1f}ms | Detect time: {timing['detect_ms']:.1f}ms")
    print(f"Total: {(timing['dehaze_ms'] + timing['detect_ms']):.1f}ms")

    side_by_side = cv2.hconcat([raw_drawn, enhanced_drawn])
    side_by_side_bgr = cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR)

    out_path = args.output or "awod_result.jpg"
    cv2.imwrite(out_path, side_by_side_bgr)
    print(f"Result saved to: {out_path}")

    cv2.imshow("AWOD: Original (left) vs Dehazed (right)", side_by_side_bgr)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
