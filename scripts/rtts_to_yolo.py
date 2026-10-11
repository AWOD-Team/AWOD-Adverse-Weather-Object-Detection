#!/usr/bin/env python3
"""Convert RTTS (VOC xml, real fog) into a flat YOLO dataset with 2 classes.

Class map (confirmed design):  vehicle = bicycle|car|motorbike|bus|truck,
person = person.  Boxes whose class is not in the map are dropped, boxes
degenerate after clipping are dropped.

Layout produced:
    <dst>/images/<stem>.<ext>   (hard-linked when possible, else copied)
    <dst>/labels/<stem>.txt

Usage:
    python rtts_to_yolo.py --src /data/fog_detect/rtts --dst /data/fog_detect/rtts_ds
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
CLASSES = ["vehicle", "person"]
NAME_TO_ID = {
    "bicycle": 0, "car": 0, "motorbike": 0, "motorcycle": 0,
    "bus": 0, "truck": 0, "person": 1,
}


def link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="RTTS root (JPEGImages/ + Annotations/)")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--min-side", type=float, default=1.0,
                    help="drop boxes thinner than this many pixels (default 1.0)")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    img_dir, ann_dir = src / "JPEGImages", src / "Annotations"
    if not img_dir.is_dir() or not ann_dir.is_dir():
        print(f"[!] expected {img_dir} and {ann_dir}", file=sys.stderr)
        return 1
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)

    kept = Counter()
    dropped = Counter()
    n_img = n_box = 0
    no_xml, no_box = [], 0

    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in IMG_EXT:
            continue
        ann = ann_dir / f"{img.stem}.xml"
        if not ann.is_file():
            no_xml.append(img.name)
            continue
        try:
            root = ET.parse(ann).getroot()
        except ET.ParseError as exc:
            print(f"[!] bad xml {ann.name}: {exc}", file=sys.stderr)
            continue
        size = root.find("size")
        if size is None:
            no_xml.append(img.name)
            continue
        w_img = float(size.find("width").text)
        h_img = float(size.find("height").text)

        lines = []
        for obj in root.findall("object"):
            node = obj.find("name")
            name = (node.text or "").strip().lower() if node is not None else ""
            cid = NAME_TO_ID.get(name)
            if cid is None:
                dropped[name or "<empty>"] += 1
                continue
            bb = obj.find("bndbox")
            try:
                x1 = float(bb.find("xmin").text)
                y1 = float(bb.find("ymin").text)
                x2 = float(bb.find("xmax").text)
                y2 = float(bb.find("ymax").text)
            except (AttributeError, TypeError, ValueError):
                dropped["<bad-bbox>"] += 1
                continue
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)
            if x2 - x1 < args.min_side or y2 - y1 < args.min_side:
                dropped["<tiny>"] += 1
                continue
            cx, cy = ((x1 + x2) / 2) / w_img, ((y1 + y2) / 2) / h_img
            bw, bh = (x2 - x1) / w_img, (y2 - y1) / h_img
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            kept[CLASSES[cid]] += 1

        if not lines:
            no_box += 1
            continue

        link_or_copy(img, dst / "images" / img.name)
        (dst / "labels" / f"{img.stem}.txt").write_text("\n".join(lines) + "\n",
                                                        encoding="utf-8")
        n_img += 1
        n_box += len(lines)

    print(f"images written : {n_img}")
    print(f"boxes written  : {n_box}")
    print(f"class counts   : {dict(kept)}")
    print(f"dropped boxes  : {dict(dropped)}")
    print(f"skipped        : no-xml/无size={len(no_xml)}  empty-after-map={no_box}")
    yaml = dst / "data.yaml"
    yaml.write_text(
        f"path: {dst.resolve()}\n"
        "train: images\n"
        "val: images\n"
        "names:\n" + "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES)) + "\n",
        encoding="utf-8",
    )
    print(f"yaml           : {yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
