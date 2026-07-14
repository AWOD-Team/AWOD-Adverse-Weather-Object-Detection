import argparse
import shutil
import sys
from pathlib import Path
from random import Random

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset_foggy"
RANDOM_SEED = 42
VAL_SPLIT = 0.2

SOURCES = [
    {
        "images": Path("G:/leetcode/dataset_built/images_1"),
        "labels": Path("G:/leetcode/dataset_built/annonation/labels_1"),
    },
    {
        "images": Path("G:/leetcode/dataset_built/images_2"),
        "labels": Path("G:/leetcode/dataset_built/annonation/labels_2"),
    },
]


def prepare_dataset():
    """合并 images_1 + images_2 并按 80/20 划分 train/val"""

    pairs = []
    for src in SOURCES:
        for img_path in src["images"].glob("*"):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            label_path = src["labels"] / f"{img_path.stem}.txt"
            if not label_path.exists():
                print(f"  [WARN] 无标签文件: {img_path.name}, 跳过")
                continue
            pairs.append((img_path, label_path))

    rng = Random(RANDOM_SEED)
    rng.shuffle(pairs)
    split_idx = int(len(pairs) * (1 - VAL_SPLIT))
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    print(f"总计匹配: {len(pairs)} 张 (train={len(train_pairs)}, val={len(val_pairs)})")

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    for subset, subset_pairs in [("train", train_pairs), ("val", val_pairs)]:
        img_dir = DATASET_DIR / "images" / subset
        lbl_dir = DATASET_DIR / "labels" / subset
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path, lbl_path in subset_pairs:
            shutil.copy2(img_path, img_dir / img_path.name)
            shutil.copy2(lbl_path, lbl_dir / lbl_path.name)

    print(f"数据集已准备完毕: {DATASET_DIR}")


def main():
    parser = argparse.ArgumentParser(description="YOLOv11 雾天目标检测训练")
    parser.add_argument("--model", type=str, default="yolo11l.pt",
                        help="预训练权重路径 (默认: yolo11l.pt)")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图像尺寸")
    parser.add_argument("--batch", type=int, default=8, help="批次大小")
    parser.add_argument("--lr0", type=float, default=0.01, help="初始学习率")
    parser.add_argument("--patience", type=int, default=20, help="早停 patience")
    parser.add_argument("--workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--device", type=str, default="0", help="设备 (0=cuda:0, cpu)")
    parser.add_argument("--project", type=str, default="runs/train_foggy",
                        help="训练结果保存目录")
    parser.add_argument("--name", type=str, default="yolo11l_foggy",
                        help="实验名称")
    parser.add_argument("--skip-prepare", action="store_true",
                        help="跳过数据集准备 (复用已有文件夹)")
    args = parser.parse_args()

    if not args.skip_prepare:
        print("=" * 50)
        print("[1/2] 准备数据集...")
        print("=" * 50)
        if DATASET_DIR.exists():
            shutil.rmtree(DATASET_DIR)
        prepare_dataset()

    yaml_path = PROJECT_ROOT / "dataset_foggy.yaml"
    if not yaml_path.exists():
        print(f"[ERROR] 未找到数据集配置: {yaml_path}")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("[2/2] 开始训练 YOLOv11 Large...")
    print("=" * 50)
    print(f"  模型:      {args.model}")
    print(f"  数据集:    {yaml_path}")
    print(f"  轮数:      {args.epochs}")
    print(f"  图像尺寸:  {args.imgsz}")
    print(f"  批次:      {args.batch}")
    print(f"  学习率:    {args.lr0}")
    print(f"  设备:      {args.device}")
    print(f"  输出目录:  {args.project}/{args.name}")
    print("=" * 50)

    model = YOLO(args.model)
    results = model.train(
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

    print("\n" + "=" * 50)
    print("训练完成!")
    print(f"最佳权重: {results.save_dir}/weights/best.pt")
    print("=" * 50)


if __name__ == "__main__":
    main()
