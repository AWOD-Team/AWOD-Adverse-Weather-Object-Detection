import argparse
import sys
from pathlib import Path

import cv2

from core.pipeline import AWODPipeline
from core.dehaze.hybrid_dehaze import HybridDehazer


def main():
    parser = argparse.ArgumentParser(description="AWOD: Adverse Weather Object Detection")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="YOLO model path or name")
    parser.add_argument("--image", type=str, required=False, help="Path to input image")
    parser.add_argument("--fusion-weight", type=float, default=0.5,
                        help="DCP(1.0) vs Retinex(0.0) fusion weight")
    parser.add_argument("--dehaze", action="store_true", help="Enable dehazing (use only if provided)")
    parser.add_argument("--input-dir", type=str, default=str(Path('..') / 'dataset_built' / 'testset'),
                        help="Input directory (relative path by default)")
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    args = parser.parse_args()

    # prepare dehazer and pipeline (dehazer only created when requested)
    dehazer = HybridDehazer(fusion_weight=args.fusion_weight) if args.dehaze else None

    pipeline = AWODPipeline(
        yolo_model_path=args.model,
        dehazer=dehazer,
        enable_dehaze=args.dehaze,
    )

    # decide between single image and input directory modes
    input_dir = Path(args.input_dir)
    out_arg = args.output

    # helper to process one image path and save with same name
    def process_and_save(image_path: Path):
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"[ERROR] Cannot read image: {image_path}")
            return
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        raw_drawn, enhanced_drawn, timing = pipeline.process_and_draw(img_rgb)

        print(f"Detections: {len(pipeline.process(img_rgb)['detections'])} objects")
        if args.dehaze:
            print(f"Dehaze time: {timing['dehaze_ms']:.1f}ms | Detect time: {timing['detect_ms']:.1f}ms")
            # side_by_side = cv2.hconcat([raw_drawn, enhanced_drawn])
            out_img_rgb = enhanced_drawn
        else:
            print(f"Detect time: {timing['detect_ms']:.1f}ms")
            out_img_rgb = raw_drawn

        out_img_bgr = cv2.cvtColor(out_img_rgb, cv2.COLOR_RGB2BGR)

        # determine output directory and filename
        if out_arg:
            out_dir = Path(out_arg)
        else:
            out_dir = Path("output_dehaze" if args.dehaze else "output")
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / image_path.name
        cv2.imwrite(str(out_path), out_img_bgr)
        print(f"Result saved to: {out_path}")

    # If input_dir exists and is a directory, process all images inside it
    if input_dir and Path(input_dir).is_dir():
        p = Path(input_dir)
        imgs = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
            imgs.extend(sorted(p.glob(ext)))
        if not imgs:
            print(f"[WARN] No images found in {input_dir}")
            return
        for im_path in imgs:
            process_and_save(im_path)
    elif args.image:
        process_and_save(Path(args.image))
    else:
        print("[ERROR] No input specified. Use --image or ensure --input-dir has images.")
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
