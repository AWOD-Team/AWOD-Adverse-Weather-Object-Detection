"""Restore AWOD night images, labels, and the archived five-fold membership."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import yaml

from scripts.rtts_to_yolo import link_or_copy
from tools.dehaze_images import read_rgb, write_rgb
from tools.restore_fog_fivefold import CLASSES, PARTS, load_splits, read_csv, restore_labels


def load_nikon_labels(annotation_dir):
    annotations = {}
    for split in PARTS:
        file = annotation_dir / f"NOD_Nikon750_{split}.json"
        data = json.loads(file.read_bytes())
        classes = {row["id"]: {"car": 0, "person": 1}.get(row["name"])
                   for row in data["categories"]}
        by_image = defaultdict(list)
        for row in data["annotations"]:
            by_image[row["image_id"]].append(row)
        for image in data["images"]:
            width, height = image["width"], image["height"]
            lines, counts = [], Counter()
            for row in by_image[image["id"]]:
                cls = classes[row["category_id"]]
                x, y, w, h = row["bbox"]
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(width, x + w), min(height, y + h)
                if cls is None or x1 >= x2 or y1 >= y2:
                    continue
                coords = ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                          (x2 - x1) / width, (y2 - y1) / height)
                lines.append(f"{cls} " + " ".join(f"{value:.6f}" for value in coords) + "\n")
                counts[CLASSES[cls]] += 1
            name = image["file_name"]
            if name in annotations:
                raise ValueError(f"Nikon image appears more than once: {name}")
            annotations[name] = ("".join(lines), counts)
    return annotations


def restore(data_root, archive, annotation_dir=None, destination=None):
    default_destination = data_root / "processed/night_2cls_clean_v2"
    destination = destination or default_destination
    if destination.exists():
        raise FileExistsError(destination)
    if annotation_dir is None:
        annotation_dir = default_destination / "nod_annotations"
        if not annotation_dir.is_dir():
            annotation_dir = data_root / "nod_annotations"
    samples = {row["name"]: row for row in read_csv(archive / "group_manifest.csv")}
    splits, summary = load_splits(archive, samples)
    nikon = data_root / "awod-dataset4"
    nikon_hashes = {row["filename"]: row["sha256"] for row in read_csv(nikon / "manifest.csv")}
    nikon_labels = load_nikon_labels(annotation_dir)
    rtts = data_root / "awod-dataset1/RTTS"
    dawn_files = defaultdict(list)
    for directory in (data_root / "awod-dataset2/DAWN").iterdir():
        for file in directory.rglob("*"):
            if file.is_file() and file.suffix.lower() in {".jpg", ".xml"}:
                dawn_files[file.name].append(file)

    inputs, label_counts = {}, Counter()
    for name in samples:
        source, original = name.split("__", 1)
        xml = Path(original).with_suffix(".xml").name
        if source == "nod":
            image = nikon / "Nikon750" / original
            label, counts = nikon_labels[original]
            if not label:
                raise ValueError(f"Archived Nikon sample has no car or person boxes: {name}")
        elif source == "rtts":
            image = rtts / "JPEGImages" / original
            label, counts = restore_labels(rtts / "Annotations" / xml)
        elif source == "dawn":
            image, = dawn_files[original]
            annotation, = dawn_files[xml]
            label, counts = restore_labels(annotation)
        else:
            raise ValueError(f"Unknown source: {source}")
        if not image.is_file() or Path(name).name != name:
            raise ValueError(f"Invalid source image: {image}")
        inputs[name] = (source, image, label)
        label_counts.update(counts)

    sources = Counter(source for source, _, _ in inputs.values())
    print(f"Matched images: {dict(sources)}; boxes: {dict(label_counts)}", flush=True)
    if sources != {"nod": 2669, "rtts": 79, "dawn": 32}:
        raise ValueError("Sources differ from the night delivery")
    if label_counts != {"vehicle": 3507, "person": 16624}:
        raise ValueError("Reconstructed box counts differ from the night delivery")

    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    with TemporaryDirectory(prefix=".restore_night_", dir=destination.parent) as temp:
        stage = Path(temp) / destination.name
        (stage / "images").mkdir(parents=True)
        (stage / "labels").mkdir()
        (stage / "nod_annotations").mkdir()
        for part in PARTS:
            name = f"NOD_Nikon750_{part}.json"
            copy2(annotation_dir / name, stage / "nod_annotations" / name)

        def write_sample(item):
            name, (source, image, label) = item
            output = stage / "images" / name
            if source == "nod":
                data = image.read_bytes()
                if sha256(data).hexdigest() != nikon_hashes[image.name]:
                    raise ValueError(f"Nikon source checksum mismatch: {image}")
                # The available Nikon copy is 2048px. Match the historical
                # training resolution without changing normalized box coordinates.
                rgb = cv2.imdecode(np.frombuffer(data, np.uint8),
                                   cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                height, width = rgb.shape[:2]
                size = (1280, round(height * 1280 / width))
                rgb = cv2.cvtColor(cv2.resize(rgb, size, interpolation=cv2.INTER_AREA),
                                   cv2.COLOR_BGR2RGB)
                write_rgb(output, rgb)
                if read_rgb(output).shape[:2] != size[::-1]:
                    raise ValueError(f"Unexpected restored dimensions: {output}")
            else:
                if cv2.imdecode(np.fromfile(image, np.uint8), cv2.IMREAD_UNCHANGED) is None:
                    raise ValueError(f"Unreadable source image: {image}")
                link_or_copy(image, output)
            (stage / "labels" / Path(name).with_suffix(".txt")).write_text(
                label, encoding="utf-8", newline="\n")

        with ThreadPoolExecutor(max_workers=6) as pool:
            for index, _ in enumerate(pool.map(write_sample, inputs.items()), 1):
                if index % 500 == 0 or index == len(inputs):
                    print(f"Restored {index}/{len(inputs)}", flush=True)

        splits_root = stage / "folds5_grouped_v1"
        for (fold, part), names in splits.items():
            folder = splits_root / f"fold{fold}"
            folder.mkdir(parents=True, exist_ok=True)
            if len(names) != summary[fold][part]:
                raise ValueError(f"Incorrect fold count: {fold}/{part}")
            (folder / f"{part}.txt").write_text(
                "".join(f"./../../images/{name}\n" for name in names), encoding="utf-8")
            if part == "test":
                config = {**{key: f"{key}.txt" for key in PARTS}, "names": CLASSES}
                (folder / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False),
                                                  encoding="utf-8")
        for name in ("group_manifest.csv", "split_manifest.csv", "fold_summary.json"):
            (splits_root / name).write_bytes((archive / name).read_bytes())
        (stage / "classes.yaml").write_text(yaml.safe_dump({"nc": 2, "names": CLASSES}),
                                            encoding="utf-8")
        (stage / "all.txt").write_text("".join(f"./images/{name}\n" for name in samples),
                                        encoding="utf-8")
        (stage / "restore_audit.json").write_text(json.dumps({
            "images": len(samples), "sources": sources, "boxes": label_counts,
            "fold_entries": sum(map(len, splits.values())),
            "fold_counts": {fold: {part: row[part] for part in PARTS}
                            for fold, row in summary.items()},
            "nikon_pixels": "1280px long edge regenerated from awod-dataset4 2048px JPEGs",
            "nikon_annotations": "Official NOD Nikon750 COCO train/val/test",
        }, indent=2) + "\n", encoding="utf-8")
        stage.rename(destination)
    print(f"Complete: {destination}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("awod_data"))
    parser.add_argument("--archive", type=Path,
                        default=Path("deliverables/awod_yolo11n_night5fold_v2/splits"))
    parser.add_argument("--ann-dir", type=Path, help="Directory containing the official Nikon COCO JSON files")
    parser.add_argument("--output", type=Path, help="New directory for the restored dataset")
    args = parser.parse_args()
    restore(args.data_root.resolve(), args.archive.resolve(),
            args.ann_dir.resolve() if args.ann_dir else None,
            args.output.resolve() if args.output else None)
