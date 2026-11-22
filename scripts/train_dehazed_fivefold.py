#!/usr/bin/env python3
"""Train YOLO11n on the daylight-dehazed fog five-fold dataset.

The validation split selects the best checkpoint. The test split is evaluated
only after each fold finishes training.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = (
    ROOT
    / "awod_data"
    / "processed"
    / "fog_2cls_dehazed_daylight_v1"
    / "folds5_grouped_v1"
)
DEFAULT_MODEL = ROOT / "yolo11n.pt"
DEFAULT_OUTPUT = ROOT / "runs" / "fog_dehazed_daylight_fivefold"


def validate_folds(data_root: Path, folds: list[int]):
    checked = 0
    for fold in sorted(set(folds)):
        fold_dir = data_root / f"fold{fold}"
        for split in ("train", "val", "test"):
            split_path = fold_dir / f"{split}.txt"
            if not split_path.is_file():
                raise FileNotFoundError(f"Split file not found: {split_path}")
            for line in split_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                image = (fold_dir / line.strip()).resolve()
                if not image.is_file():
                    raise FileNotFoundError(f"Image not found: {image}")
                parts = list(image.parts)
                image_index = parts.index("images")
                label = Path(
                    *parts[:image_index],
                    "labels",
                    *parts[image_index + 1 :],
                ).with_suffix(".txt")
                if not label.is_file():
                    raise FileNotFoundError(f"Label not found: {label}")
                checked += 1
    print(f"Validated {checked} split entries", flush=True)


def _as_list(value):
    if value is None:
        return []
    try:
        return list(value)
    except TypeError:
        return []


def evaluate(model_path: Path, data_yaml: Path, output: Path, fold: int, args):
    model = YOLO(str(model_path))
    result = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(output / "test"),
        name=f"fold{fold}",
        exist_ok=True,
        plots=True,
        verbose=False,
    )
    box = result.box
    names = result.names or model.names
    class_index = _as_list(getattr(box, "ap_class_index", None))
    maps = _as_list(getattr(box, "maps", None))
    ap50 = _as_list(getattr(box, "ap50", None))

    metrics = {
        "fold": fold,
        "weights": str(model_path),
        "data": str(data_yaml),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "per_class_map50": {
            str(names[int(class_id)]): float(ap50[index])
            for index, class_id in enumerate(class_index)
            if index < len(ap50)
        },
        "per_class_map50_95": {
            str(names[int(class_id)]): float(maps[index])
            for index, class_id in enumerate(class_index)
            if index < len(maps)
        },
    }
    metrics_dir = output / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / f"fold{fold}.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metrics


def write_summary(rows: list[dict], output: Path):
    metric_keys = ("precision", "recall", "map50", "map50_95")
    classes = sorted(
        {
            class_name
            for row in rows
            for class_name in row["per_class_map50_95"]
        }
    )
    summary = {"folds": rows, "aggregate": {}}
    for key in metric_keys:
        values = [row[key] for row in rows]
        summary["aggregate"][key] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "n": len(values),
        }
    for class_name in classes:
        for key in ("per_class_map50", "per_class_map50_95"):
            values = [
                row[key][class_name]
                for row in rows
                if class_name in row[key]
            ]
            if not values:
                continue
            summary["aggregate"][f"{class_name}_{key}"] = {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
                "n": len(values),
            }

    output.mkdir(parents=True, exist_ok=True)
    (output / "test_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    with (output / "test_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["fold", *metric_keys]
            + [f"{name}_map50" for name in classes]
            + [f"{name}_map50_95" for name in classes]
        )
        for row in rows:
            writer.writerow(
                [row["fold"], *(f"{row[key]:.6f}" for key in metric_keys)]
                + [
                    f"{row['per_class_map50'].get(name, float('nan')):.6f}"
                    for name in classes
                ]
                + [
                    f"{row['per_class_map50_95'].get(name, float('nan')):.6f}"
                    for name in classes
                ]
            )
        for label, reducer in (
            ("mean", statistics.mean),
            ("std", statistics.stdev if len(rows) > 1 else None),
        ):
            values = []
            for key in metric_keys:
                values.append(
                    f"{reducer([row[key] for row in rows]):.6f}"
                    if reducer is not None
                    else "0.000000"
                )
            for key in ("per_class_map50", "per_class_map50_95"):
                for class_name in classes:
                    class_values = [
                        row[key][class_name]
                        for row in rows
                        if class_name in row[key]
                    ]
                    values.append(
                        f"{reducer(class_values):.6f}"
                        if reducer is not None and class_values
                        else "0.000000"
                    )
            writer.writerow([label, *values])

    print(json.dumps(summary["aggregate"], indent=2, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--cache",
        choices=("false", "ram", "disk"),
        default="false",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        choices=range(1, 6),
        default=list(range(1, 6)),
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    model_path = args.model.resolve()
    output = args.output.resolve()
    if not model_path.is_file():
        parser.error(f"Model not found: {model_path}")
    cache = False if args.cache == "false" else args.cache
    rows = []
    validate_folds(data_root, args.folds)

    for fold in sorted(set(args.folds)):
        data_yaml = data_root / f"fold{fold}" / "data.yaml"
        if not data_yaml.is_file():
            parser.error(f"Fold configuration not found: {data_yaml}")

        best = output / "train" / f"fold{fold}" / "weights" / "best.pt"
        if not (args.skip_existing and best.is_file()):
            print(f"\n===== Train fold {fold}: {data_yaml} =====", flush=True)
            model = YOLO(str(model_path))
            model.train(
                data=str(data_yaml),
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                seed=args.seed,
                lr0=args.lr0,
                patience=args.patience,
                amp=not args.no_amp,
                cache=cache,
                workers=args.workers,
                device=args.device,
                project=str(output / "train"),
                name=f"fold{fold}",
                exist_ok=True,
                plots=True,
                verbose=True,
            )
        if not best.is_file():
            raise FileNotFoundError(f"Training did not create {best}")

        metrics_path = output / "metrics" / f"fold{fold}.json"
        if args.skip_existing and metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            print(f"\n===== Test fold {fold}: {best} =====", flush=True)
            metrics = evaluate(best, data_yaml, output, fold, args)
        rows.append(metrics)
        print(
            f"fold{fold}: mAP50={metrics['map50']:.4f} "
            f"mAP50-95={metrics['map50_95']:.4f}",
            flush=True,
        )

    if rows:
        write_summary(rows, output)
    print(f"\nResults: {output}", flush=True)


if __name__ == "__main__":
    main()
