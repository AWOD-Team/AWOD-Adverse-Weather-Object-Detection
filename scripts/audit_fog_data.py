#!/usr/bin/env python3
"""Day-1 data audit for the fog-detection week (see 下一周任务计划.md §4.1/§8).

Reads the raw sources read-only and writes an audit report:

    reports/data_audit.json
    reports/data_audit.csv
    reports/<dataset>_pairs.csv   (per-image: size, boxes, classes, difficult)

Checks per dataset:
  * image/label (or XML) pairing, missing on either side
  * class histogram by NAME (integer class ids are never trusted)
  * difficult flag counts (VOC) and boxes outside the image / degenerate
  * exact duplicates (md5) and near-duplicate pairs (dHash)
  * image size stats, empty-label count, small-object (<32x32 px) ratio

Usage:
    python audit_fog_data.py \
        --rtts rtts --dawn datasets/dawn_yolo2 --ots dataset \
        --out-dir reports
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
VOC_CLASSES = ["bicycle", "bus", "car", "motorbike", "person"]
DAWN_CLASSES = ["vehicle", "person"]
# class order of the local 417-image OTS set (see classes.txt)
OTS_CLASSES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus",
               "train", "truck"]
SMALL_AREA = 32 * 32


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(path: Path, size: int = 8) -> int:
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("L").resize((size + 1, size), Image.LANCZOS)
        px = list(im.getdata())
    bits = k = 0
    for r in range(size):
        for c in range(size):
            if px[r * (size + 1) + c] > px[r * (size + 1) + c + 1]:
                bits |= 1 << k
            k += 1
    return bits


def near_dup_pairs(hashes: dict[str, int], thresh: int = 6):
    """Count pairs with hamming distance <= thresh.

    Uses a 7-band multi-index join (pigeonhole: distance <= 6 cannot touch all
    7 bands), so no pair within the threshold is missed.  A low-16-bit bucket
    join would silently drop pairs that differ inside the bucket key.
    """
    bands: list[dict[int, list[str]]] = [defaultdict(list) for _ in range(7)]
    for name, h in hashes.items():
        for b in range(7):
            bands[b][(h >> (b * 10)) & 0x3FF].append(name)
    seen: set[tuple[str, str]] = set()
    pairs = 0
    for name, h in hashes.items():
        cands: set[str] = set()
        for b in range(7):
            cands.update(bands[b][(h >> (b * 10)) & 0x3FF])
        for other in cands:
            if other == name:
                continue
            key = (name, other) if name < other else (other, name)
            if key in seen:
                continue
            seen.add(key)
            if bin(h ^ hashes[other]).count("1") <= thresh:
                pairs += 1
    return pairs


def image_size(p: Path) -> tuple[int, int]:
    from PIL import Image
    try:
        with Image.open(p) as im:
            return im.size
    except Exception:
        return (0, 0)


def audit_voc(images_dir: Path, ann_dir: Path) -> dict:
    imgs = {p.stem: p for p in sorted(images_dir.iterdir())
            if p.suffix.lower() in IMG_EXT}
    xmls = {p.stem: p for p in sorted(ann_dir.glob("*.xml"))}
    only_img, only_xml = sorted(set(imgs) - set(xmls)), sorted(set(xmls) - set(imgs))

    cls = Counter()
    difficult = 0
    bad_box = 0
    outside = 0
    empty = 0
    small = 0
    total = 0
    rows = []
    sizes = Counter()
    md5s: dict[str, str] = {}
    dup_md5 = 0
    hashes: dict[str, int] = {}

    for stem in sorted(set(imgs) & set(xmls)):
        img = imgs[stem]
        img_md5 = md5_of(img)
        if img_md5 in md5s:
            dup_md5 += 1
        else:
            md5s[img_md5] = img.name
        hashes[img.name] = dhash(img)

        root = ET.parse(xmls[stem]).getroot()
        size = root.find("size")
        w = float(size.find("width").text)
        h = float(size.find("height").text)
        sizes[(int(w), int(h))] += 1
        objs = root.findall("object")
        if not objs:
            empty += 1
        per = Counter()
        n_diff = 0
        for obj in objs:
            node = obj.find("name")
            name = (node.text or "").strip().lower() if node is not None else "<none>"
            cls[name] += 1
            total += 1
            d = obj.find("difficult")
            if d is not None and (d.text or "").strip() == "1":
                difficult += 1
                n_diff += 1
            bb = obj.find("bndbox")
            try:
                x1, y1, x2, y2 = (float(bb.find(k).text)
                                  for k in ("xmin", "ymin", "xmax", "ymax"))
            except (AttributeError, TypeError, ValueError):
                bad_box += 1
                continue
            if x2 <= x1 or y2 <= y1:
                bad_box += 1
                continue
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                outside += 1
            if (x2 - x1) * (y2 - y1) < SMALL_AREA:
                small += 1
            per[name] += 1
        rows.append({
            "image": str(img.relative_to(images_dir.parent)) if images_dir.parent else img.name,
            "width": int(w), "height": int(h), "n_boxes": len(objs),
            "n_difficult": n_diff,
            "classes": "|".join(f"{k}:{v}" for k, v in sorted(per.items())),
        })

    return {
        "kind": "voc",
        "images": len(imgs), "annotations": len(xmls),
        "paired": len(set(imgs) & set(xmls)),
        "only_image": len(only_img), "only_annotation": len(only_xml),
        "only_image_examples": only_img[:5],
        "only_annotation_examples": only_xml[:5],
        "boxes": total, "classes": dict(cls.most_common()),
        "difficult": difficult,
        "degenerate_or_bad_boxes": bad_box,
        "boxes_outside_image": outside,
        "empty_annotation_images": empty,
        "small_boxes_lt_32x32": small,
        "small_ratio": round(small / total, 4) if total else 0.0,
        "exact_duplicate_images": dup_md5,
        "near_duplicate_pairs_dhash6": near_dup_pairs(hashes),
        "distinct_sizes": len(sizes),
        "top_sizes": [{"w": k[0], "h": k[1], "n": v} for k, v in sizes.most_common(5)],
        "rows": rows,
    }


def audit_yolo(root: Path, names: list[str], subsets: dict[str, Path] | None = None) -> dict:
    img_dir = root / "images"
    lbl_dir = root / "labels"
    imgs = {p.stem: p for p in sorted(img_dir.rglob("*"))
            if p.suffix.lower() in IMG_EXT}
    lbls = {p.stem: p for p in sorted(lbl_dir.rglob("*.txt"))}
    cls = Counter()
    bad = 0
    empty = 0
    small = 0
    total = 0
    rows = []
    sizes = Counter()
    md5s: dict[str, str] = {}
    dup_md5 = 0
    hashes: dict[str, int] = {}

    for stem in sorted(imgs):
        img = imgs[stem]
        m = md5_of(img)
        if m in md5s:
            dup_md5 += 1
        else:
            md5s[m] = img.name
        hashes[img.name] = dhash(img)
        w, h = image_size(img)
        sizes[(w, h)] += 1
        lp = lbls.get(stem)
        per = Counter()
        n = 0
        if lp is not None:
            for line in lp.read_text(errors="ignore").splitlines():
                f = line.split()
                if len(f) < 5:
                    continue
                try:
                    c = int(float(f[0]))
                    cx, cy, bw, bh = (float(x) for x in f[1:5])
                except ValueError:
                    bad += 1
                    continue
                if bw <= 0 or bh <= 0 or cx < 0 or cy < 0 or cx > 1 or cy > 1:
                    bad += 1
                    continue
                nm = names[c] if c < len(names) else f"class{c}"
                cls[nm] += 1
                per[nm] += 1
                n += 1
                total += 1
                if bw * w * bh * h < SMALL_AREA:
                    small += 1
        if n == 0:
            empty += 1
        rows.append({"image": str(img.relative_to(root)), "width": w, "height": h,
                     "n_boxes": n, "classes": "|".join(f"{k}:{v}" for k, v in sorted(per.items()))})

    out = {
        "kind": "yolo",
        "images": len(imgs), "labels": len(lbls),
        "paired": len(set(imgs) & set(lbls)),
        "only_image": len(set(imgs) - set(lbls)),
        "only_label": len(set(lbls) - set(imgs)),
        "boxes": total, "classes": dict(cls.most_common()),
        "invalid_boxes": bad,
        "empty_label_images": empty,
        "small_boxes_lt_32x32": small,
        "small_ratio": round(small / total, 4) if total else 0.0,
        "exact_duplicate_images": dup_md5,
        "near_duplicate_pairs_dhash6": near_dup_pairs(hashes),
        "distinct_sizes": len(sizes),
        "top_sizes": [{"w": k[0], "h": k[1], "n": v} for k, v in sizes.most_common(5)],
        "rows": rows,
    }
    if subsets:
        out["subsets"] = {}
        for tag, path in subsets.items():
            if path.is_file():
                lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
                out["subsets"][tag] = len(lines)
                out["subsets"][f"{tag}_examples"] = lines[:3]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rtts")
    ap.add_argument("--dawn")
    ap.add_argument("--ots")
    ap.add_argument("--out-dir", default="reports")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict] = {}

    if args.rtts:
        r = Path(args.rtts)
        report["rtts"] = audit_voc(r / "JPEGImages", r / "Annotations")
    if args.dawn:
        d = Path(args.dawn)
        subs = {t: d / f"subset_{t}.txt"
                for t in ("fog", "rain", "sand", "snow", "other")}
        report["dawn"] = audit_yolo(d, DAWN_CLASSES, subs)
    if args.ots:
        report["ots417"] = audit_yolo(Path(args.ots), OTS_CLASSES)

    # per-image details for the main training set
    for key in ("rtts", "dawn", "ots417"):
        if key in report:
            rows = report[key].pop("rows", [])
            with (out_dir / f"{key}_pairs.csv").open("w", newline="",
                                                     encoding="utf-8-sig") as fh:
                if rows:
                    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)

    (out_dir / "data_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    with (out_dir / "data_audit.csv").open("w", newline="",
                                           encoding="utf-8-sig") as fh:
        fields = ["dataset", "kind", "images", "annotations", "labels", "paired",
                  "only_image", "only_annotation", "only_label", "boxes",
                  "difficult", "degenerate_or_bad_boxes", "invalid_boxes",
                  "boxes_outside_image", "empty_annotation_images",
                  "empty_label_images", "small_boxes_lt_32x32", "small_ratio",
                  "exact_duplicate_images", "near_duplicate_pairs_dhash6"]
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for key, val in report.items():
            w.writerow({"dataset": key, **val})

    # console summary
    print(f"{'dataset':8s} {'imgs':>6s} {'paired':>7s} {'boxes':>7s} "
          f"{'empty':>6s} {'bad':>5s} {'small%':>7s} {'dup':>5s} {'near':>5s}")
    for key, val in report.items():
        print(f"{key:8s} {val['images']:6d} {val['paired']:7d} {val['boxes']:7d} "
              f"{val.get('empty_annotation_images', val.get('empty_label_images', 0)):6d} "
              f"{val.get('degenerate_or_bad_boxes', val.get('invalid_boxes', 0)):5d} "
              f"{val['small_ratio'] * 100:6.1f}% "
              f"{val['exact_duplicate_images']:5d} {val['near_duplicate_pairs_dhash6']:5d}")
    print(f"\nclasses:")
    for key, val in report.items():
        print(f"  {key:8s} {val['classes']}")
        if val.get("subsets"):
            print(f"           subsets: {val['subsets']}")
    print(f"\nwritten: {out_dir}/data_audit.json  {out_dir}/data_audit.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
