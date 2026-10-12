#!/usr/bin/env python3
"""Build a reproducible, road-domain RTTS small-sample training set.

The source dataset is never modified.  The script selects exactly ``--target``
images from an existing grouped train split, creates hard-linked image/label
files, and emits manifests, selection metadata, an exclusion ledger, and
contact sheets for visual review.

Selection policy:

* use only names already assigned to the supplied train split;
* require a valid image/label pair with at least one vehicle box;
* keep difficult foggy road images (blur is not an automatic exclusion);
* exclude person-only portraits and non-road person scenes from this version;
* use at most one image from each verified near-duplicate cluster;
* search deterministically for a subset that preserves person-box, small-box,
  density, and source diversity statistics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Processed two-class RTTS root (images/ + labels/).")
    parser.add_argument("--train-manifest", type=Path, required=True,
                        help="Existing leak-free train split; basename is used.")
    parser.add_argument("--val-manifest", type=Path,
                        help="Optional held-out split copied into val.txt.")
    parser.add_argument("--dup-json", type=Path,
                        help="dup_scan.py JSON containing filename->cluster mapping.")
    parser.add_argument("--manual-exclude", type=Path,
                        help="Optional text/CSV file whose first field is a filename.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--draws", type=int, default=2500)
    parser.add_argument("--max-per-source", type=int, default=8)
    parser.add_argument("--video-max-per-source", type=int, default=3,
                        help="Maximum selected frames from one hv* video source.")
    parser.add_argument("--video-min-frame-gap", type=int, default=30,
                        help="Minimum frame-number gap within one hv* video source.")
    parser.add_argument("--base-manifest", type=Path,
                        help="Prefer an already reviewed selection, replacing only rejected items.")
    parser.add_argument("--contact-size", type=int, default=50)
    return parser.parse_args()


def source_group(filename: str) -> str:
    stem = Path(filename).stem
    group = re.sub(r"[_\-.]?\d+$", "", stem)
    return group or stem


def video_frame(filename: str) -> tuple[str, int] | None:
    """Return (video id, frame number) for RTTS hv* sequential frames."""
    match = re.fullmatch(r"(hv\d+)[_.-](\d+)", Path(filename).stem, re.IGNORECASE)
    if not match:
        return None
    return match.group(1).lower(), int(match.group(2))


def read_manifest_names(path: Path) -> list[str]:
    names: list[str] = []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if line:
            names.append(Path(line).name)
    return names


def read_manual_excludes(path: Path | None) -> set[str]:
    if path is None or not path.is_file():
        return set()
    excluded: set[str] = set()
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.lower().startswith("filename"):
            continue
        excluded.add(Path(line.split(",", 1)[0].strip()).name)
    return excluded


def load_cluster_map(path: Path | None) -> tuple[dict[str, str], dict]:
    if path is None or not path.is_file():
        return {}, {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("clusters") or {}
    clusters = {Path(name).name: str(cluster) for name, cluster in raw.items()}
    meta = {key: payload.get(key) for key in (
        "thresh", "verify_thresh", "n_images", "n_pairs_verified",
        "n_pairs_within_thresh", "n_multi_image_clusters", "largest_clusters",
    )}
    return clusters, meta


def label_stats(path: Path) -> dict:
    counts = Counter()
    small = 0
    areas: list[float] = []
    invalid = 0
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            invalid += 1
            continue
        try:
            class_id = int(fields[0])
            cx, cy, width, height = map(float, fields[1:])
        except ValueError:
            invalid += 1
            continue
        if class_id not in (0, 1) or not (0 <= cx <= 1 and 0 <= cy <= 1) or width <= 0 or height <= 0:
            invalid += 1
            continue
        counts[class_id] += 1
        area = width * height
        areas.append(area)
        if area * 640 * 640 < 32 * 32:
            small += 1
    return {
        "vehicle": counts[0],
        "person": counts[1],
        "n_box": counts[0] + counts[1],
        "small": small,
        "max_person_area": max(
            (float(line.split()[3]) * float(line.split()[4])
             for line in path.read_text(encoding="utf-8-sig").splitlines()
             if len(line.split()) == 5 and line.split()[0] == "1"),
            default=0.0,
        ),
        "invalid": invalid,
    }


def density_bin(n_box: int) -> str:
    if n_box <= 3:
        return "sparse"
    if n_box <= 8:
        return "medium"
    return "dense"


def inspect_image(path: Path) -> tuple[bool, int, int, str]:
    try:
        from PIL import Image
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
        return width > 0 and height > 0, width, height, ""
    except Exception as exc:  # data audit must record rather than abort
        return False, 0, 0, f"decode_error:{type(exc).__name__}"


def summarize(records: list[dict]) -> dict:
    vehicle = sum(item["vehicle"] for item in records)
    person = sum(item["person"] for item in records)
    small = sum(item["small"] for item in records)
    boxes = vehicle + person
    density = Counter(item["density"] for item in records)
    sources = Counter(item["source"] for item in records)
    return {
        "images": len(records),
        "vehicle_boxes": vehicle,
        "person_boxes": person,
        "person_box_share": person / boxes if boxes else 0.0,
        "small_box_share": small / boxes if boxes else 0.0,
        "images_with_person_share": (
            sum(item["person"] > 0 for item in records) / len(records)
            if records else 0.0
        ),
        "mean_boxes_per_image": boxes / len(records) if records else 0.0,
        "density_share": {
            key: density[key] / len(records) if records else 0.0
            for key in ("sparse", "medium", "dense")
        },
        "unique_sources": len(sources),
        "max_images_per_source": max(sources.values(), default=0),
        "top_sources": sources.most_common(15),
    }


def score_subset(summary: dict, target: dict) -> float:
    score = 0.0
    score += 12.0 * abs(summary["person_box_share"] - target["person_box_share"])
    score += 7.0 * abs(summary["small_box_share"] - target["small_box_share"])
    score += 4.0 * abs(summary["images_with_person_share"] - target["images_with_person_share"])
    denom = max(target["mean_boxes_per_image"], 1.0)
    score += 2.0 * abs(summary["mean_boxes_per_image"] - target["mean_boxes_per_image"]) / denom
    score += 3.0 * sum(
        abs(summary["density_share"][key] - target["density_share"][key])
        for key in ("sparse", "medium", "dense")
    )
    score -= 0.002 * summary["unique_sources"]
    return score


def sample_once(records: list[dict], target_count: int, seed: int,
                max_per_source: int, video_max_per_source: int,
                video_min_frame_gap: int, preferred_names: set[str]) -> list[dict] | None:
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for item in records:
        buckets[item["source"]].append(item)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    sources = sorted(buckets)
    rng.shuffle(sources)
    source_counts = Counter()
    video_frames: dict[str, list[int]] = defaultdict(list)
    used_clusters: set[str] = set()
    selected: list[dict] = []

    def can_add(item: dict) -> bool:
        if item["cluster"] in used_clusters:
            return False
        video = video_frame(item["name"])
        if video:
            video_id, frame_number = video
            if len(video_frames[video_id]) >= video_max_per_source:
                return False
            if any(abs(frame_number - other) < video_min_frame_gap
                   for other in video_frames[video_id]):
                return False
        elif source_counts[item["source"]] >= max_per_source:
            return False
        return True

    def add(item: dict) -> None:
        selected.append(item)
        source_counts[item["source"]] += 1
        used_clusters.add(item["cluster"])
        video = video_frame(item["name"])
        if video:
            video_frames[video[0]].append(video[1])

    # Retain as much of the already reviewed v1 manifest as possible.  Randomize
    # its order per draw so the optimizer can choose among conflicting video
    # frames while all non-conflicting reviewed images remain stable.
    preferred = [item for item in records if item["name"] in preferred_names]
    rng.shuffle(preferred)
    for item in preferred:
        if len(selected) >= target_count:
            break
        if can_add(item):
            add(item)

    # Round-robin over shuffled source buckets.  This is O(n), gives broad
    # source coverage, and keeps each draw cheap enough for repeated search.
    while len(selected) < target_count:
        progressed = False
        rng.shuffle(sources)
        for source in sources:
            if len(selected) >= target_count:
                break
            if source_counts[source] >= max_per_source:
                continue
            bucket = buckets[source]
            while bucket:
                item = bucket.pop()
                if not can_add(item):
                    continue
                add(item)
                progressed = True
                break
        if not progressed:
            return None
    return selected


def choose_subset(records: list[dict], target_count: int, seed: int, draws: int,
                  max_per_source: int, video_max_per_source: int,
                  video_min_frame_gap: int,
                  preferred_names: set[str]) -> tuple[list[dict], float, dict]:
    target_summary = summarize(records)
    best: list[dict] | None = None
    best_score = math.inf
    for draw in range(draws):
        picked = sample_once(
            records, target_count, seed + draw * 7919, max_per_source,
            video_max_per_source, video_min_frame_gap, preferred_names,
        )
        if picked is None:
            continue
        summary = summarize(picked)
        score = score_subset(summary, target_summary)
        if score < best_score:
            best, best_score = picked[:], score
    if best is None:
        raise RuntimeError(
            f"Could not draw {target_count} images with max_per_source={max_per_source}"
        )
    return sorted(best, key=lambda item: item["name"]), best_score, target_summary


def link_file(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        try:
            destination.symlink_to(source)
            return "symlink"
        except OSError:
            shutil.copy2(source, destination)
            return "copy"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_contact_sheets(records: list[dict], output: Path, per_sheet: int) -> None:
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    output.mkdir(parents=True, exist_ok=True)
    columns = 5
    tile_width, image_height, label_height = 256, 150, 28
    rows = math.ceil(per_sheet / columns)
    font = ImageFont.load_default()
    for sheet_index, start in enumerate(range(0, len(records), per_sheet), 1):
        batch = records[start:start + per_sheet]
        canvas = Image.new("RGB", (columns * tile_width, rows * (image_height + label_height)), "white")
        draw = ImageDraw.Draw(canvas)
        for index, record in enumerate(batch):
            row, column = divmod(index, columns)
            x, y = column * tile_width, row * (image_height + label_height)
            with Image.open(record["image"]) as image:
                thumb = ImageOps.contain(image.convert("RGB"), (tile_width, image_height))
            px = x + (tile_width - thumb.width) // 2
            py = y + (image_height - thumb.height) // 2
            canvas.paste(thumb, (px, py))
            draw.rectangle((x, y + image_height, x + tile_width, y + image_height + label_height), fill="white")
            text = f"{record['name']}  v{record['vehicle']} p{record['person']}"
            draw.text((x + 3, y + image_height + 5), text[:40], fill="black", font=font)
        canvas.save(output / f"selected_{sheet_index:02d}.jpg", quality=90)


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    image_dir, label_dir = dataset / "images", dataset / "labels"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"Expected {image_dir} and {label_dir}")
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")

    train_names = read_manifest_names(args.train_manifest)
    val_names = read_manifest_names(args.val_manifest) if args.val_manifest else []
    if set(train_names) & set(val_names):
        raise ValueError("Train and validation manifests overlap by filename")

    clusters, duplicate_meta = load_cluster_map(args.dup_json)
    manual_excludes = read_manual_excludes(args.manual_exclude)
    # Known user-confirmed out-of-domain example remains explicit even when no
    # manual file has been created yet.
    manual_excludes.add("AM_Bing_274.png")
    preferred_names = (
        set(read_manifest_names(args.base_manifest))
        if args.base_manifest else set()
    )

    all_records: list[dict] = []
    ledger: list[dict] = []
    for name in train_names:
        image = image_dir / name
        label = label_dir / f"{Path(name).stem}.txt"
        record = {
            "name": name,
            "image": image,
            "label": label,
            "source": source_group(name),
            "cluster": clusters.get(name, f"single:{name}"),
        }
        reason = ""
        if name in manual_excludes:
            reason = "manual_out_of_domain"
        elif not image.is_file():
            reason = "missing_image"
        elif not label.is_file():
            reason = "missing_label"
        else:
            stats = label_stats(label)
            record.update(stats)
            valid, width, height, decode_reason = inspect_image(image)
            record.update({"width": width, "height": height})
            if not valid:
                reason = decode_reason
            elif stats["invalid"]:
                reason = "invalid_yolo_label"
            elif stats["n_box"] == 0:
                reason = "empty_label"
            elif stats["vehicle"] == 0:
                # Conservative small-sample v1: road-domain proxy.  This removes
                # selfies/portraits and other pure-person web-search noise while
                # mixed road scenes still supply person boxes.
                reason = "person_only_not_used_in_road_smallset_v1"
            else:
                record["density"] = density_bin(stats["n_box"])
        record["eligibility_reason"] = reason or "eligible"
        all_records.append(record)
        if reason:
            ledger.append({"filename": name, "status": "EXCLUDE_FROM_SMALL500", "reason": reason})

    eligible = [record for record in all_records if record["eligibility_reason"] == "eligible"]
    if len(eligible) < args.target:
        raise ValueError(f"Only {len(eligible)} eligible images for target {args.target}")

    selected, selection_score, target_summary = choose_subset(
        eligible, args.target, args.seed, args.draws, args.max_per_source,
        args.video_max_per_source, args.video_min_frame_gap, preferred_names,
    )
    selected_names = {record["name"] for record in selected}
    selected_clusters = {record["cluster"] for record in selected}
    val_clusters = {clusters.get(name, f"single:{name}") for name in val_names}
    cluster_overlap = sorted(selected_clusters & val_clusters)
    if cluster_overlap:
        raise ValueError(f"Selected/validation duplicate clusters overlap: {cluster_overlap[:10]}")

    for record in eligible:
        if record["name"] not in selected_names:
            ledger.append({"filename": record["name"], "status": "NOT_SELECTED", "reason": "sampling_balance"})

    output = args.output.resolve()
    out_images, out_labels = output / "images", output / "labels"
    out_images.mkdir(parents=True)
    out_labels.mkdir(parents=True)
    link_modes = Counter()
    for record in selected:
        link_modes[link_file(record["image"], out_images / record["name"])] += 1
        link_modes[link_file(record["label"], out_labels / record["label"].name)] += 1

    train_txt = output / "train.txt"
    train_txt.write_text(
        "\n".join(str((out_images / record["name"]).resolve()) for record in selected) + "\n",
        encoding="utf-8",
    )
    if val_names:
        val_txt = output / "val.txt"
        val_txt.write_text(
            "\n".join(str((image_dir / name).resolve()) for name in val_names) + "\n",
            encoding="utf-8",
        )
    else:
        val_txt = train_txt

    (output / "data.yaml").write_text(
        f"path: {output}\n"
        f"train: {train_txt}\n"
        f"val: {val_txt}\n"
        "names:\n  0: vehicle\n  1: person\n",
        encoding="utf-8",
    )

    fields = [
        "filename", "source", "cluster", "width", "height", "vehicle",
        "person", "n_box", "small", "density", "max_person_area",
    ]
    with (output / "selection.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in selected:
            row = {key: record.get(key, "") for key in fields}
            row["filename"] = record["name"]
            writer.writerow(row)

    with (output / "selection_ledger.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "status", "reason"])
        writer.writeheader()
        writer.writerows(sorted(ledger, key=lambda row: row["filename"]))

    selected_summary = summarize(selected)
    version = output.name
    report = {
        "version": version,
        "source_dataset": str(dataset),
        "train_manifest": str(args.train_manifest.resolve()),
        "validation_manifest": str(args.val_manifest.resolve()) if args.val_manifest else None,
        "target": args.target,
        "seed": args.seed,
        "draws": args.draws,
        "max_per_source": args.max_per_source,
        "policy": {
            "requires_vehicle": True,
            "blur_auto_excluded": False,
            "one_per_near_duplicate_cluster": True,
            "video_max_per_source": args.video_max_per_source,
            "video_min_frame_gap": args.video_min_frame_gap,
            "manual_excludes": sorted(manual_excludes),
        },
        "preferred_base_manifest": str(args.base_manifest.resolve()) if args.base_manifest else None,
        "preferred_images_retained": len(selected_names & preferred_names),
        "train_pool_images": len(train_names),
        "eligible_images": len(eligible),
        "excluded_from_eligibility": Counter(
            record["eligibility_reason"] for record in all_records
            if record["eligibility_reason"] != "eligible"
        ),
        "eligible_pool_summary": target_summary,
        "selected_summary": selected_summary,
        "selection_score": selection_score,
        "selected_validation_name_overlap": len(selected_names & set(val_names)),
        "selected_validation_cluster_overlap": len(cluster_overlap),
        "duplicate_scan": duplicate_meta,
        "link_modes_image_and_label": dict(link_modes),
    }
    # Counter is JSON serializable only after conversion.
    report["excluded_from_eligibility"] = dict(report["excluded_from_eligibility"])
    (output / "selection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    manifest_hash = sha256_file(train_txt)
    (output / "SHA256SUMS").write_text(
        f"{manifest_hash}  train.txt\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        f"# {version}\n\n"
        "Exactly 500 RTTS images selected from the existing fold-2 training pool.\n"
        "The source dataset was not modified. Images and labels are hard-linked when possible.\n\n"
        "Policy: valid two-class labels, at least one vehicle, manual visual exclusions, "
        "one image per verified near-duplicate cluster, and controlled spacing for hv* video frames.\n\n"
        "Review `contact_sheets/` before training. Record later replacements in a new versioned "
        "directory; do not silently edit this set.\n",
        encoding="utf-8",
    )
    write_contact_sheets(selected, output / "contact_sheets", args.contact_size)

    print(json.dumps({
        "output": str(output),
        "selected": len(selected),
        "eligible": len(eligible),
        "summary": selected_summary,
        "train_manifest_sha256": manifest_hash,
        "contact_sheets": math.ceil(len(selected) / args.contact_size),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
