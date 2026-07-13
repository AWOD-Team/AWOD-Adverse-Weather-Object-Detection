import argparse
import json
import sys
from pathlib import Path

import cv2

from core.pipeline import AWODPipeline
from core.dehaze.hybrid_dehaze import HybridDehazer


def main():
    parser = argparse.ArgumentParser(description="AWOD: Adverse Weather Object Detection - JSON output")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="YOLO model path or name")
    parser.add_argument("--image", type=str, required=False, help="Path to input image")
    parser.add_argument("--fusion-weight", type=float, default=0.5,
                        help="DCP(1.0) vs Retinex(0.0) fusion weight")
    parser.add_argument("--dehaze", action="store_true", help="Enable dehazing")
    parser.add_argument("--input-dir", type=str, default=str(Path('..') / 'dataset_built' / 'testset'),
                        help="Input directory")
    parser.add_argument("--output", type=str, default=None, help="Output directory for JSON results")
    parser.add_argument("--combine", action="store_true",
                        help="Combine all results into a single JSON file (COCO-style)")
    args = parser.parse_args()

    dehazer = HybridDehazer(fusion_weight=args.fusion_weight) if args.dehaze else None

    pipeline = AWODPipeline(
        yolo_model_path=args.model,
        dehazer=dehazer,
        enable_dehaze=args.dehaze,
    )

    input_dir = Path(args.input_dir)
    out_arg = args.output

    def process_and_save_json(image_path: Path):
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"[ERROR] Cannot read image: {image_path}")
            return None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = pipeline.process(img_rgb)
        detections = result["detections"]
        timing = result["timing"]

        img_h, img_w = img.shape[:2]

        annotation = {
            "image": image_path.name,
            "width": img_w,
            "height": img_h,
            "timing": timing,
            "num_detections": len(detections),
            "detections": [],
        }

        for det in detections:
            annotation["detections"].append({
                "class_id": det["class_id"],
                "class_name": det["class_name"],
                "confidence": det["confidence"],
                "bbox": det["bbox"],
            })

        if out_arg:
            out_dir = Path(out_arg)
        else:
            out_dir = Path("output_json_dehaze" if args.dehaze else "output_json")
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / f"{image_path.stem}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(annotation, f, indent=2, ensure_ascii=False)

        print(f"Detections: {len(detections)} objects | Saved to: {json_path}")
        return annotation

    if input_dir and Path(input_dir).is_dir():
        p = Path(input_dir)
        imgs = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
            imgs.extend(sorted(p.glob(ext)))
        if not imgs:
            print(f"[WARN] No images found in {input_dir}")
            return

        if args.combine:
            coco_output = {
                "images": [],
                "annotations": [],
                "categories": [],
            }
            category_map = {}
            cat_id_counter = 1
            ann_id_counter = 1

            for img_id, im_path in enumerate(imgs, start=1):
                img = cv2.imread(str(im_path))
                if img is None:
                    print(f"[ERROR] Cannot read image: {im_path}")
                    continue
                img_h, img_w = img.shape[:2]
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                result = pipeline.process(img_rgb)
                detections = result["detections"]
                timing = result["timing"]

                coco_output["images"].append({
                    "id": img_id,
                    "file_name": im_path.name,
                    "width": img_w,
                    "height": img_h,
                    "timing": timing,
                })

                for det in detections:
                    if det["class_name"] not in category_map:
                        category_map[det["class_name"]] = {
                            "id": cat_id_counter,
                            "name": det["class_name"],
                        }
                        cat_id_counter += 1

                    x1, y1, x2, y2 = det["bbox"]
                    coco_output["annotations"].append({
                        "id": ann_id_counter,
                        "image_id": img_id,
                        "category_id": category_map[det["class_name"]]["id"],
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "area": (x2 - x1) * (y2 - y1),
                        "score": det["confidence"],
                    })
                    ann_id_counter += 1

                print(f"{im_path.name}: {len(detections)} objects")

            coco_output["categories"] = list(category_map.values())

            if out_arg:
                out_dir = Path(out_arg)
            else:
                out_dir = Path("output_json_dehaze" if args.dehaze else "output_json")
            out_dir.mkdir(parents=True, exist_ok=True)

            combined_path = out_dir / "annotations.json"
            with open(combined_path, "w", encoding="utf-8") as f:
                json.dump(coco_output, f, indent=2, ensure_ascii=False)
            print(f"Combined annotations saved to: {combined_path}")
        else:
            for im_path in imgs:
                process_and_save_json(im_path)

    elif args.image:
        process_and_save_json(Path(args.image))
    else:
        print("[ERROR] No input specified. Use --image or ensure --input-dir has images.")
        return


if __name__ == "__main__":
    main()
