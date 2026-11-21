"""Restore the original AWOD fog five-fold dataset from local RTTS and DAWN.

Run from the project root with python -m tools.restore_fog_fivefold.
Uses the archived membership and order without rerunning any screening or split.
"""

import argparse
import csv
import json
import math
import shutil
from collections import Counter
from pathlib import Path, PurePosixPath
from tempfile import mkdtemp
from xml.etree import ElementTree

import yaml

from scripts.rtts_to_yolo import NAME_TO_ID, link_or_copy
from tools.make_grouped_fivefold import perceptual_hash


PARTS = ("train", "val", "test")
CLASSES = {0: "vehicle", 1: "person"}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def restore_labels(xml_path):
    # Apply the same clipping, min_side=1 and precision as rtts_to_yolo.py.
    root = ElementTree.parse(xml_path).getroot()
    width = float(root.findtext("size/width"))
    height = float(root.findtext("size/height"))
    if not all(math.isfinite(value) and value > 0 for value in (width, height)):
        raise ValueError(f"Invalid annotation dimensions: {xml_path}")
    lines = []
    counts = Counter()
    for obj in root.findall("object"):
        class_id = NAME_TO_ID.get((obj.findtext("name") or "").strip().lower())
        if class_id is None:
            continue
        x1, y1, x2, y2 = (float(obj.findtext(f"bndbox/{axis}"))
                          for axis in ("xmin", "ymin", "xmax", "ymax"))
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        coords = (((x1 + x2) / 2) / width, ((y1 + y2) / 2) / height,
                  (x2 - x1) / width, (y2 - y1) / height)
        lines.append(f"{class_id} " + " ".join(f"{value:.6f}" for value in coords))
        counts[CLASSES[class_id]] += 1
    return "\n".join(lines) + ("\n" if lines else ""), counts


def load_splits(archive, samples):
    summary = json.loads((archive / "fold_summary.json").read_text(encoding="utf-8"))
    if len(samples) != summary["images"]:
        raise ValueError("Unexpected sample count in group manifest")
    split_index = {(int(row["fold"]), row["name"]): row["split"]
                   for row in read_csv(archive / "split_manifest.csv")}
    splits = {}
    for number in range(1, 6):
        seen = {}
        groups = {}
        for part in PARTS:
            file = archive / f"fold{number}" / f"{part}.txt"
            names = [PurePosixPath(line).name for line in file.read_text(encoding="utf-8").splitlines()
                     if line.strip()]
            for name in names:
                row = samples[name]
                group = row["group_id"]
                if (name in seen or groups.get(group, part) != part
                        or split_index.get((number, name)) != part
                        or (part == "test") != (number == int(row["test_fold"]))):
                    raise ValueError(f"Archive disagrees or overlaps: fold {number}, {name}")
                seen[name] = groups[group] = part
            splits[number, part] = names
        if seen.keys() != samples.keys():
            raise ValueError(f"Fold {number} does not cover the sample set")
    if len(split_index) != len(samples) * 5:
        raise ValueError("Unexpected membership count in split manifest")
    return splits, {row["fold"]: row for row in summary["folds"]}


def write_split(stage, number, part, names, samples, annotations):
    fold_dir = stage / "folds5_grouped_v1" / f"fold{number}"
    (fold_dir / "images" / part).mkdir(parents=True)
    (fold_dir / "labels" / part).mkdir(parents=True)
    stats = Counter()
    for name in names:
        label = Path(name).with_suffix(".txt")
        link_or_copy(stage / "images" / name, fold_dir / "images" / part / name)
        link_or_copy(stage / "labels" / label, fold_dir / "labels" / part / label)
        stats[samples[name]["source"]] += 1
        counts = annotations[name]
        stats["vehicle_boxes"] += counts["vehicle"]
        stats["person_boxes"] += counts["person"]
        stats["person_images"] += counts["person"] > 0
    stats["images"] = len(names)
    (fold_dir / f"{part}.txt").write_text(
        "".join(f"./images/{part}/{name}\n" for name in names), encoding="utf-8")
    return dict(stats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("awod_data"))
    parser.add_argument("--splits", type=Path,
                        default=Path("deliverables/awod_yolo11n_fog5fold_v2/splits"))
    args = parser.parse_args()
    destination = args.data_root.resolve() / "processed" / "fog_2cls_clean_v1"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    rows = read_csv(args.splits / "group_manifest.csv")
    samples = {row["name"]: row for row in rows}
    if len(samples) != len(rows) or len({Path(name).stem.casefold() for name in samples}) != len(rows):
        raise ValueError("Duplicate sample name or label stem")
    splits, expected_stats = load_splits(args.splits, samples)
    rtts = args.data_root / "awod-dataset1" / "RTTS"
    dawn = args.data_root / "awod-dataset2" / "DAWN" / "Fog" / "Fog"
    roots = {"rtts": (rtts / "JPEGImages", rtts / "Annotations"),
             "dawn": (dawn, dawn / "Fog_PASCAL_VOC")}
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(mkdtemp(prefix=".fog_2cls_clean_v1-", dir=destination.parent))
    print(f"Restoration staging directory: {stage}", flush=True)
    for directory in ("images", "labels", "source_annotations", "folds5_grouped_v1"):
        (stage / directory).mkdir()

    annotations = {}
    for index, (name, row) in enumerate(samples.items(), 1):
        original = row["original_name"]
        if Path(name).name != name or Path(original).name != original:
            raise ValueError(f"Expected plain filenames in group manifest: {name}, {original}")
        image_root, xml_root = roots[row["source"]]
        image = image_root / original
        xml = xml_root / Path(original).with_suffix(".xml")
        if perceptual_hash(image) != int(row["phash"], 16):
            raise ValueError(f"Image pHash differs from original manifest: {image}")
        labels, annotations[name] = restore_labels(xml)
        # Keep the originals independent of the source clones; folds link into this copy.
        shutil.copy2(image, stage / "images" / name)
        shutil.copy2(xml, stage / "source_annotations" / Path(name).with_suffix(".xml"))
        (stage / "labels" / Path(name).with_suffix(".txt")).write_text(
            labels, encoding="utf-8", newline="\n")
        if index % 500 == 0:
            print(f"Restored {index}/{len(samples)} original image/annotation pairs", flush=True)

    audit = {"images": len(samples), "matching_archived_phashes": len(samples),
             "source_counts": dict(Counter(row["source"] for row in rows)), "folds": {}}
    for item in ("group_manifest.csv", "split_manifest.csv", "fold_summary.json"):
        shutil.copy2(args.splits / item, stage / "folds5_grouped_v1" / item)
    for (number, part), names in splits.items():
        stats = write_split(stage, number, part, names, samples, annotations)
        expected = expected_stats[number][part]
        differences = {field: (expected[field], stats.get(field, 0))
                       for field in ("images", "rtts", "dawn", "vehicle_boxes",
                                     "person_boxes", "person_images")
                       if expected[field] != stats.get(field, 0)}
        if differences:
            raise ValueError(f"Historical counts differ: fold {number} {part}: {differences}")
        audit["folds"].setdefault(number, {})[part] = stats
        if part == "test":
            config = {"path": destination.as_posix(),
                      **{key: f"folds5_grouped_v1/fold{number}/{key}.txt" for key in PARTS},
                      "names": CLASSES}
            fold_dir = stage / "folds5_grouped_v1" / f"fold{number}"
            (fold_dir / "data.yaml").write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            print(f"Restored and verified fold {number}", flush=True)
    (stage / "restore_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    stage.rename(destination)
    print(f"Restored fog dataset: {destination}", flush=True)


if __name__ == "__main__":
    main()
