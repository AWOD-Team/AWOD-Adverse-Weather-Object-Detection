import argparse
from pathlib import Path

import cv2

from core.pipeline import AWODPipeline
from core.dehaze.hybrid_dehaze import HybridDehazer


def main():
    parser = argparse.ArgumentParser(description="AWOD: Adverse Weather Object Detection - YOLO label output")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="YOLO model path or name")
    parser.add_argument("--image", type=str, required=False, help="Path to input image")
    parser.add_argument("--fusion-weight", type=float, default=0.5,
                        help="DCP(1.0) vs Retinex(0.0) fusion weight")
    parser.add_argument("--dehaze", action="store_true", help="Enable dehazing")
    parser.add_argument("--input-dir", type=str, default=str(Path('..') / 'dataset_built' / 'testset'),
                        help="Input directory")
    parser.add_argument("--output", type=str, default=None, help="Output directory for YOLO label files")
    args = parser.parse_args()

    dehazer = HybridDehazer(fusion_weight=args.fusion_weight) if args.dehaze else None

    pipeline = AWODPipeline(
        yolo_model_path=args.model,
        dehazer=dehazer,
        enable_dehaze=args.dehaze,
    )

    out_dir = Path(args.output) if args.output else Path("labels_dehaze" if args.dehaze else "labels")

    input_dir = Path(args.input_dir)

    def process_and_save_label(image_path: Path):
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"[ERROR] Cannot read image: {image_path}")
            return
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w = img.shape[:2]

        result = pipeline.process(img_rgb)
        detections = result["detections"]

        out_dir.mkdir(parents=True, exist_ok=True)
        txt_path = out_dir / f"{image_path.stem}.txt"

        lines = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cx = (x1 + x2) / 2.0 / img_w
            cy = (y1 + y2) / 2.0 / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            lines.append(f"{det['class_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"Detections: {len(detections)} objects | Saved to: {txt_path}")

    if input_dir and input_dir.is_dir():
        imgs = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
            imgs.extend(sorted(input_dir.glob(ext)))
        if not imgs:
            print(f"[WARN] No images found in {input_dir}")
            return
        for im_path in imgs:
            process_and_save_label(im_path)
    elif args.image:
        process_and_save_label(Path(args.image))
    else:
        print("[ERROR] No input specified. Use --image or ensure --input-dir has images.")


if __name__ == "__main__":
    main()
