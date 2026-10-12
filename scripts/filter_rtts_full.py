#!/usr/bin/env python3
"""Prepare or materialize a quality-controlled full RTTS two-class dataset.

The source dataset is read-only.  A previously reviewed mandatory manifest is
always retained.  Other images pass mechanical validation, verified
near-duplicate suppression and spacing of known ``hv*`` video frames before
they are presented as contact sheets for semantic review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mandatory-keep", type=Path, required=True)
    parser.add_argument(
        "--manual-exclude", type=Path, action="append", default=[],
        help="Manual exclusion list; repeat the option to merge multiple lists.",
    )
    parser.add_argument(
        "--exclude-video-source", action="append", default=[],
        help="Exclude every frame from a video source such as hv11; repeat as needed.",
    )
    parser.add_argument("--dup-json", type=Path)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        help="Materialize final hard-linked dataset after review.")
    parser.add_argument("--video-min-frame-gap", type=int, default=30)
    parser.add_argument("--contact-size", type=int, default=60)
    return parser.parse_args()


def read_names(path: Path | None) -> set[str]:
    if path is None or not path.is_file():
        return set()
    names: set[str] = set()
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.lower().startswith("filename"):
            continue
        names.add(Path(line.split(",", 1)[0].strip()).name)
    return names


def label_stats(path: Path) -> dict:
    counts = Counter()
    small = invalid = 0
    for raw in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        fields = raw.split()
        if not fields:
            continue
        if len(fields) != 5:
            invalid += 1
            continue
        try:
            class_id = int(fields[0])
            cx, cy, width, height = map(float, fields[1:])
        except ValueError:
            invalid += 1
            continue
        if (class_id not in (0, 1) or not (0 <= cx <= 1 and 0 <= cy <= 1)
                or width <= 0 or height <= 0 or width > 1 or height > 1):
            invalid += 1
            continue
        counts[class_id] += 1
        if width * height * 640 * 640 < 32 * 32:
            small += 1
    return {
        "vehicle": counts[0], "person": counts[1],
        "n_box": counts[0] + counts[1], "small": small, "invalid": invalid,
    }


def inspect_image(path: Path) -> tuple[bool, int, int, str]:
    try:
        from PIL import Image
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
        return width > 0 and height > 0, width, height, ""
    except Exception as exc:
        return False, 0, 0, f"decode_error:{type(exc).__name__}"


def video_frame(name: str) -> tuple[str, int] | None:
    match = re.fullmatch(r"(hv\d+)[_.-](\d+)", Path(name).stem, re.IGNORECASE)
    if not match:
        return None
    return match.group(1).lower(), int(match.group(2))


def load_clusters(path: Path | None) -> tuple[dict[str, str], dict]:
    if path is None or not path.is_file():
        return {}, {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    clusters = {
        Path(name).name: str(cluster)
        for name, cluster in (payload.get("clusters") or {}).items()
    }
    meta = {key: payload.get(key) for key in (
        "n_images", "n_pairs_verified", "n_pairs_within_thresh",
        "n_multi_image_clusters", "largest_clusters",
    )}
    return clusters, meta


def choose_duplicate_representatives(records: list[dict], mandatory: set[str]) -> tuple[list[dict], list[dict]]:
    by_cluster: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_cluster[record["cluster"]].append(record)
    kept: list[dict] = []
    rejected: list[dict] = []
    for cluster_records in by_cluster.values():
        forced = [r for r in cluster_records if r["name"] in mandatory]
        if len(forced) > 1:
            raise ValueError(
                f"Mandatory manifest contains multiple near duplicates: "
                f"{[r['name'] for r in forced]}"
            )
        if forced:
            chosen = forced[0]
        else:
            chosen = max(
                cluster_records,
                key=lambda r: (r["width"] * r["height"], r["n_box"], r["name"]),
            )
        kept.append(chosen)
        for record in cluster_records:
            if record is not chosen:
                record["reason"] = f"near_duplicate_of:{chosen['name']}"
                rejected.append(record)
    return kept, rejected


def thin_video_frames(records: list[dict], mandatory: set[str], gap: int) -> tuple[list[dict], list[dict]]:
    by_video: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    ordinary: list[dict] = []
    for record in records:
        parsed = video_frame(record["name"])
        if parsed:
            by_video[parsed[0]].append((parsed[1], record))
        else:
            ordinary.append(record)

    kept = ordinary[:]
    rejected: list[dict] = []
    for video_id, items in sorted(by_video.items()):
        forced = sorted((frame, r) for frame, r in items if r["name"] in mandatory)
        chosen: list[tuple[int, dict]] = forced[:]
        for frame, record in sorted(items):
            if record["name"] in mandatory:
                continue
            if all(abs(frame - other) >= gap for other, _ in chosen):
                chosen.append((frame, record))
            else:
                nearest = min(chosen, key=lambda pair: abs(frame - pair[0]))
                record["reason"] = f"adjacent_video_frame_of:{nearest[1]['name']}"
                rejected.append(record)
        kept.extend(record for _, record in chosen)
    return kept, rejected


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_contact_sheets(records: list[dict], output: Path, per_sheet: int) -> None:
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    output.mkdir(parents=True, exist_ok=True)
    columns = 5
    tile_width, image_height, label_height = 256, 150, 28
    rows = math.ceil(per_sheet / columns)
    font = ImageFont.load_default()
    for sheet_index, start in enumerate(range(0, len(records), per_sheet), 1):
        batch = records[start:start + per_sheet]
        canvas = Image.new(
            "RGB", (columns * tile_width, rows * (image_height + label_height)), "white"
        )
        draw = ImageDraw.Draw(canvas)
        for index, record in enumerate(batch):
            row, column = divmod(index, columns)
            x, y = column * tile_width, row * (image_height + label_height)
            with Image.open(record["image"]) as image:
                thumb = ImageOps.contain(image.convert("RGB"), (tile_width, image_height))
            canvas.paste(
                thumb,
                (x + (tile_width - thumb.width) // 2, y + (image_height - thumb.height) // 2),
            )
            draw.rectangle(
                (x, y + image_height, x + tile_width, y + image_height + label_height),
                fill="white",
            )
            marker = "M" if record["mandatory"] else "R"
            text = f"{marker} {record['name']} v{record['vehicle']} p{record['person']}"
            draw.text((x + 3, y + image_height + 5), text[:42], fill="black", font=font)
        canvas.save(output / f"review_{sheet_index:03d}.jpg", quality=90)


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


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    image_dir, label_dir = dataset / "images", dataset / "labels"
    mandatory = read_names(args.mandatory_keep)
    manual_excludes: set[str] = set()
    for exclusion_path in args.manual_exclude:
        manual_excludes.update(read_names(exclusion_path))
    excluded_video_sources = {source.lower() for source in args.exclude_video_source}
    conflict = mandatory & manual_excludes
    if conflict:
        raise ValueError(f"Mandatory/exclusion conflict: {sorted(conflict)[:20]}")
    clusters, duplicate_meta = load_clusters(args.dup_json)

    image_paths = sorted(
        (p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda p: p.name,
    )
    source_names = {p.name for p in image_paths}
    if not mandatory <= source_names:
        raise ValueError(f"Mandatory images missing from source: {sorted(mandatory-source_names)[:20]}")

    eligible: list[dict] = []
    rejected: list[dict] = []
    for image in image_paths:
        name = image.name
        label = label_dir / f"{image.stem}.txt"
        record = {
            "name": name, "image": image, "label": label,
            "cluster": clusters.get(name, f"single:{name}"),
            "mandatory": name in mandatory,
        }
        reason = ""
        parsed_video = video_frame(name)
        if (
            not record["mandatory"]
            and parsed_video
            and parsed_video[0].lower() in excluded_video_sources
        ):
            reason = f"excluded_video_source:{parsed_video[0].lower()}"
        elif name in manual_excludes:
            reason = "manual_semantic_exclusion"
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
        if reason:
            record["reason"] = reason
            rejected.append(record)
        else:
            eligible.append(record)

    mandatory_rejected = [r for r in rejected if r["mandatory"]]
    if mandatory_rejected:
        raise ValueError(f"Mandatory images failed audit: {mandatory_rejected[:10]}")

    deduped, duplicate_rejected = choose_duplicate_representatives(eligible, mandatory)
    rejected.extend(duplicate_rejected)
    candidates, video_rejected = thin_video_frames(
        deduped, mandatory, args.video_min_frame_gap
    )
    rejected.extend(video_rejected)
    candidates.sort(key=lambda r: r["name"])
    rejected.sort(key=lambda r: r["name"])

    candidate_names = {r["name"] for r in candidates}
    if not mandatory <= candidate_names:
        raise AssertionError("Mandatory images were lost during dedup/video thinning")

    review_dir = args.review_dir.resolve()
    if review_dir.exists():
        raise FileExistsError(f"Review directory already exists: {review_dir}")
    review_dir.mkdir(parents=True)
    (review_dir / "candidate_keep.txt").write_text(
        "\n".join(r["name"] for r in candidates) + "\n", encoding="utf-8"
    )
    fields = [
        "name", "mandatory", "width", "height", "vehicle", "person",
        "n_box", "small", "cluster", "reason",
    ]
    write_csv(review_dir / "candidate_metadata.csv", candidates, fields)
    write_csv(review_dir / "exclusion_ledger.csv", rejected, fields)

    # Mandatory images have already had a complete visual review.  Contact
    # sheets contain only the remaining candidates that still need a decision.
    to_review = [r for r in candidates if not r["mandatory"]]
    write_contact_sheets(to_review, review_dir / "contact_sheets", args.contact_size)

    report = {
        "source_dataset": str(dataset),
        "source_images": len(image_paths),
        "mandatory_keep": len(mandatory),
        "manual_excludes": len(manual_excludes),
        "excluded_video_sources": sorted(excluded_video_sources),
        "eligible_before_dedup": len(eligible),
        "candidate_keep": len(candidates),
        "nonmandatory_candidates": len(to_review),
        "auto_excluded": len(rejected),
        "exclusion_reasons": dict(Counter(r["reason"].split(":", 1)[0] for r in rejected)),
        "video_min_frame_gap": args.video_min_frame_gap,
        "duplicate_scan": duplicate_meta,
        "contact_sheets": math.ceil(len(to_review) / args.contact_size),
    }

    if args.output:
        output = args.output.resolve()
        if output.exists():
            raise FileExistsError(f"Output already exists: {output}")
        out_images, out_labels = output / "images", output / "labels"
        out_images.mkdir(parents=True)
        out_labels.mkdir(parents=True)
        modes = Counter()
        for record in candidates:
            modes[link_file(record["image"], out_images / record["name"])] += 1
            modes[link_file(record["label"], out_labels / record["label"].name)] += 1
        all_txt = output / "all.txt"
        all_txt.write_text(
            "\n".join(str((out_images / r["name"]).resolve()) for r in candidates) + "\n",
            encoding="utf-8",
        )
        (output / "classes.yaml").write_text(
            "names:\n  0: vehicle\n  1: person\n", encoding="utf-8"
        )
        (output / "SHA256SUMS").write_text(
            f"{sha256_file(all_txt)}  all.txt\n", encoding="utf-8"
        )
        shutil.copy2(review_dir / "candidate_metadata.csv", output / "selection.csv")
        shutil.copy2(review_dir / "exclusion_ledger.csv", output / "exclusion_ledger.csv")
        report["output"] = str(output)
        report["link_modes_image_and_label"] = dict(modes)

    (review_dir / "screening_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
