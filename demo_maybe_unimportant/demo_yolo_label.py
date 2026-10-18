import argparse
from pathlib import Path

import cv2

from core.pipeline import AWODPipeline
from core.dehaze.hybrid_dehaze import HybridDehazer

TARGET_CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def resolve_model_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or path.suffix:
        return str(resolve_repo_path(value))
    return value


def main():
    parser = argparse.ArgumentParser(description="AWOD: Adverse Weather Object Detection - YOLO label output")
    parser.add_argument("--model", type=str, default=str(PROJECT_ROOT / "yolo11n.pt"), help="YOLO model path or name")
    parser.add_argument("--image", type=str, required=False, help="Path to input image")
    parser.add_argument("--fusion-weight", type=float, default=0.5,
                        help="DCP(1.0) vs Retinex(0.0) fusion weight")
    parser.add_argument("--dehaze", action="store_true", help="Enable dehazing")
    parser.add_argument("--input-dir", type=str, default=str(PROJECT_ROOT / "dataset_built" / "成员yu"),
                        help="Input directory")
    parser.add_argument("--output", type=str, default=None, help="Output directory for YOLO label files")
    parser.add_argument("--single-file", action="store_true",
                        help="Output all labels into a single txt file (one line per detection)")
    args = parser.parse_args()

    dehazer = HybridDehazer(fusion_weight=args.fusion_weight) if args.dehaze else None

    pipeline = AWODPipeline(
        yolo_model_path=resolve_model_path(args.model),
        dehazer=dehazer,
        enable_dehaze=args.dehaze,
    )

    out_dir = resolve_repo_path(args.output) if args.output else PROJECT_ROOT / ("labels_dehaze" if args.dehaze else "labels")
    input_dir = resolve_repo_path(args.input_dir)

    all_lines = []

    def process_single(image_path: Path):
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"[ERROR] Cannot read image: {image_path}")
            return
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w = img.shape[:2]

        result = pipeline.process(img_rgb)
        detections = result["detections"]

        for det in detections:
            if det["class_id"] not in TARGET_CLASSES:
                continue
            x1, y1, x2, y2 = det["bbox"]
            cx = (x1 + x2) / 2.0 / img_w
            cy = (y1 + y2) / 2.0 / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            all_lines.append(f"{image_path.name} {det['class_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

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
            if det["class_id"] not in TARGET_CLASSES:
                continue
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

        if args.single_file:
            for im_path in imgs:
                process_single(im_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            combined_path = out_dir / "labels.txt"
            with open(combined_path, "w", encoding="utf-8") as f:
                f.write("\n".join(all_lines))
            print(f"Total: {len(all_lines)} detections | Saved to: {combined_path}")
        else:
            for im_path in imgs:
                process_and_save_label(im_path)

    elif args.image:
        if args.single_file:
            process_single(resolve_repo_path(args.image))
            out_dir.mkdir(parents=True, exist_ok=True)
            combined_path = out_dir / "labels.txt"
            with open(combined_path, "w", encoding="utf-8") as f:
                f.write("\n".join(all_lines))
            print(f"Total: {len(all_lines)} detections | Saved to: {combined_path}")
        else:
            process_and_save_label(resolve_repo_path(args.image))
    else:
        print("[ERROR] No input specified. Use --image or ensure --input-dir has images.")


if __name__ == "__main__":
    main()
