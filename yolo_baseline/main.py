import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

TARGET_CLASSES = {0: "person", 1: "bicycle", 2: "car", 5: "bus"}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def collect_images(dataset_dir: Path) -> list[Path]:
    images = set()
    for ext in IMAGE_EXTS:
        for p in dataset_dir.glob(f"*{ext}"):
            images.add(p)
        for p in dataset_dir.glob(f"*{ext.upper()}"):
            images.add(p)
    return sorted(images)


def draw_boxes(img: np.ndarray, detections: list[dict]) -> np.ndarray:
    img_out = img.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        label = f"{det['class_name']} {det['confidence']:.2f}"
        cv2.rectangle(img_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_out, (x1, y1 - th - 4), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(img_out, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img_out


def main():
    parser = argparse.ArgumentParser(description="YOLO11 baseline detection on adverse weather dataset")
    parser.add_argument("--dataset", type=str,
                        default=str(Path("G:/leetcode/dataset_built/OTS_selected")),
                        help="Path to dataset directory")
    parser.add_argument("--output", type=str,
                        default=str(Path("G:/leetcode/dataset_built/OTS_selected_yolo_baseline")),
                        help="Output directory for annotated images and results")
    parser.add_argument("--model", type=str,
                        default=str(Path("F:/yolov11/yolo11n.pt")),
                        help="YOLO model name or path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--show", action="store_true", help="Display each image during processing")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    if not dataset_dir.is_dir():
        print(f"[ERROR] Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(dataset_dir)
    print(f"Found {len(images)} images in {dataset_dir}")

    model = YOLO(args.model)
    print(f"Model loaded: {args.model}")

    all_results = []
    class_counts = {name: 0 for name in TARGET_CLASSES.values()}
    total_detections = 0

    for idx, img_path in enumerate(images, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Cannot read: {img_path.name}")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = model(img_rgb, conf=args.conf, iou=args.iou, verbose=False, classes=list(TARGET_CLASSES.keys()))

        detections = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            for box, cls, conf in zip(boxes, classes, confs):
                cls_name = TARGET_CLASSES.get(int(cls), str(cls))
                detections.append({
                    "bbox": box.tolist(),
                    "class_id": int(cls),
                    "class_name": cls_name,
                    "confidence": float(conf),
                })
                class_counts[cls_name] += 1
                total_detections += 1

        drawn = draw_boxes(img_rgb, detections)
        drawn_bgr = cv2.cvtColor(drawn, cv2.COLOR_RGB2BGR)
        out_path = out_dir / f"{img_path.stem}_det.jpg"
        cv2.imwrite(str(out_path), drawn_bgr)

        all_results.append({
            "image": img_path.name,
            "num_detections": len(detections),
            "detections": detections,
        })

        print(f"[{idx:4d}/{len(images)}] {img_path.name}: {len(detections)} objects")

        if args.show:
            cv2.imshow("YOLO Baseline", drawn_bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()

    summary = {
        "model": args.model,
        "conf_threshold": args.conf,
        "iou_threshold": args.iou,
        "total_images": len(images),
        "total_detections": total_detections,
        "per_class_counts": class_counts,
        "results": all_results,
    }

    json_path = out_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults JSON saved to: {json_path}")

    print(f"\n{'='*50}")
    print(f"Summary")
    print(f"{'='*50}")
    print(f"Total images:     {len(images)}")
    print(f"Total detections: {total_detections}")
    print(f"Per-class counts:")
    for cls_name, count in class_counts.items():
        print(f"  {cls_name:10s}: {count}")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
