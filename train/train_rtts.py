import argparse
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from random import Random

from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR / "dataset_rtts"
RANDOM_SEED = 42
VAL_SPLIT = 0.2

RTTS_ROOT = Path("G:/leetcode/dataset_built/RTTS")
CLASSES = ["bicycle", "bus", "car", "motorbike", "person"]


def convert_xml_to_yolo(xml_dir: Path, txt_dir: Path, class_map: dict):
    txt_dir.mkdir(parents=True, exist_ok=True)
    for xml_path in xml_dir.glob("*.xml"):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find("size")
        img_w = int(size.find("width").text)
        img_h = int(size.find("height").text)

        lines = []
        for obj in root.findall("object"):
            cls_name = obj.find("name").text.strip()
            if cls_name not in class_map:
                continue
            cls_id = class_map[cls_name]
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            cx = ((xmin + xmax) / 2) / img_w
            cy = ((ymin + ymax) / 2) / img_h
            w = (xmax - xmin) / img_w
            h = (ymax - ymin) / img_h
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if lines:
            with open(txt_dir / f"{xml_path.stem}.txt", "w") as f:
                f.write("\n".join(lines))


def prepare_dataset():
    class_map = {name: i for i, name in enumerate(CLASSES)}
    img_dir = RTTS_ROOT / "JPEGImages"
    xml_dir = RTTS_ROOT / "Annotations"

    img_paths = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
    print(f"RTTS 图片总数: {len(img_paths)}")

    txt_dir = SCRIPT_DIR / "tmp_rtts_labels"
    if txt_dir.exists():
        shutil.rmtree(txt_dir)
    txt_dir.mkdir()
    print(f"转换 XML -> YOLO 格式...")
    convert_xml_to_yolo(xml_dir, txt_dir, class_map)

    pairs = []
    skipped = 0
    for img_path in img_paths:
        label_path = txt_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            pairs.append((img_path, label_path))
        else:
            skipped += 1
    print(f"有效图-标签对: {len(pairs)} (跳过 {skipped})")

    rng = Random(RANDOM_SEED)
    rng.shuffle(pairs)
    split_idx = int(len(pairs) * (1 - VAL_SPLIT))
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    print(f"train={len(train_pairs)}, val={len(val_pairs)}")

    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    DATASET_DIR.mkdir(parents=True)

    for subset, subset_pairs in [("train", train_pairs), ("val", val_pairs)]:
        img_dst = DATASET_DIR / "images" / subset
        lbl_dst = DATASET_DIR / "labels" / subset
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)
        for img_p, lbl_p in subset_pairs:
            shutil.copy2(img_p, img_dst / img_p.name)
            shutil.copy2(lbl_p, lbl_dst / lbl_p.name)

    shutil.rmtree(txt_dir)

    yaml_path = SCRIPT_DIR / "dataset_rtts.yaml"
    yaml_path.write_text(
        "path: G:/leetcode/AWOD-Adverse-Weather-Object-Detection/train/dataset_rtts\n"
        "train: images/train\n"
        "val: images/val\n"
        f"\nnc: {len(CLASSES)}\n"
        "names:\n" +
        "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASSES))
    )
    print(f"数据集已准备: {DATASET_DIR}")
    print(f"配置文件:     {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description="YOLOv11 RTTS 真实雾图训练")
    parser.add_argument("--model", type=str, default="yolo11s.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--project", type=str, default="runs/train_rtts")
    parser.add_argument("--name", type=str, default="yolo11s_rtts")
    parser.add_argument("--skip-prepare", action="store_true")
    args = parser.parse_args()

    if not args.skip_prepare:
        print("=" * 50)
        print("[1/2] 准备 RTTS 数据集...")
        print("=" * 50)
        prepare_dataset()

    yaml_path = SCRIPT_DIR / "dataset_rtts.yaml"
    if not yaml_path.exists():
        print(f"[ERROR] 未找到: {yaml_path}")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("[2/2] 开始训练...")
    print("=" * 50)
    print(f"  模型:   {args.model}")
    print(f"  数据集: {yaml_path}")
    print(f"  轮数:   {args.epochs}")
    print(f"  批次:   {args.batch}")
    print("=" * 50)

    model = YOLO(args.model)
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        patience=args.patience,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        pretrained=True,
        amp=True,
        cos_lr=True,
    )


if __name__ == "__main__":
    main()
