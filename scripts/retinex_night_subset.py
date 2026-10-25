#!/usr/bin/env python3
"""Build a session-held-out night subset and Retinex (MSRCR) brightened copies.

Images are grouped into shooting sessions by EXIF DateTime (gap > 30 min).
Within each session a fixed number of images is sampled evenly across the
luminance range, so the brightening test covers both dark and bright frames.
Whole sessions are assigned to train or test, preventing near-duplicate
leakage between the two splits.

Outputs:
  <out>/train/original, <out>/train/retinex
  <out>/test/original,  <out>/test/retinex
  <out>/lists/{train,test,train_retinex,test_retinex}.txt
  <out>/selection.csv

Run with the OpenCV environment, e.g.:
  /home/linux/anaconda3/envs/torch2.4_cuda12.1/bin/python \
      scripts/retinex_night_subset.py
"""
import argparse
import csv
import datetime as dt
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.dehaze.retinex import Retinex  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png"}
SESSION_GAP = dt.timedelta(minutes=30)


def exif_time(path: Path):
    try:
        value = Image.open(path).getexif().get(306)
    except Exception:
        return None
    if not value:
        return None
    try:
        return dt.datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None


def mean_luma(path: Path) -> float:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return float(img.mean()) if img is not None else float("nan")


def group_sessions(rows):
    rows = sorted(rows, key=lambda r: r[1])
    groups = [[rows[0]]]
    for prev, cur in zip(rows, rows[1:]):
        if cur[1] - prev[1] > SESSION_GAP:
            groups.append([])
        groups[-1].append(cur)
    return groups


def pick_session_samples(session, count, min_luma):
    scored = [(mean_luma(f), f, t) for f, t in session]
    scored = [x for x in scored if x[0] >= min_luma]
    if not scored:
        return []
    if len(scored) < count:
        raise ValueError(f"session only has {len(scored)} images above min-luma")
    scored.sort(key=lambda x: x[0])
    idx = np.unique(np.linspace(0, len(scored) - 1, count).round().astype(int))
    picked = [scored[i] for i in idx]
    while len(picked) < count:
        for i in range(len(scored)):
            if scored[i] not in picked:
                picked.append(scored[i])
                break
    return picked[:count]


def retinex_one(args):
    src, dst, quality = args
    img_bgr = cv2.imread(str(src))
    if img_bgr is None:
        raise RuntimeError(f"cannot read {src}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    out_rgb = Retinex(num_workers=1).enhance(img_rgb)
    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    dst.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(dst), out_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"cannot write {dst}")
    return dst, float(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2GRAY).mean())


def write_list(path: Path, files):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{f}\n" for f in files), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="/data/awod_data/Nikon750_1280")
    parser.add_argument("--out", default="/data/awod_data/Nikon750_1280_retinex_subset")
    parser.add_argument("--per-session", type=int, default=25)
    parser.add_argument("--min-luma", type=float, default=3.0,
                        help="drop near-black frames below this mean gray value")
    parser.add_argument("--test-sessions", type=int, default=2,
                        help="number of whole sessions held out for test")
    parser.add_argument("--test-dates", nargs="*", default=None,
                        help="explicit test session dates, e.g. 2020-06-16 2020-06-20")
    parser.add_argument("--quality", type=int, default=92)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    if not src.is_dir():
        raise SystemExit(f"source not found: {src}")

    files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT)
    rows = []
    skipped = []
    for f in files:
        t = exif_time(f)
        if t is None:
            skipped.append(f)
        else:
            rows.append((f, t))
    if not rows:
        raise SystemExit("no images with EXIF DateTime")
    if skipped:
        print(f"[WARN] skipped {len(skipped)} images without EXIF DateTime")

    sessions = group_sessions(rows)
    print(f"{len(rows)} images in {len(sessions)} sessions")

    picked = []
    for session in sessions:
        picked.extend(pick_session_samples(session, args.per_session, args.min_luma))

    date_of = lambda t: t.date().isoformat()
    by_date = {}
    for luma, f, t in picked:
        by_date.setdefault(date_of(t), []).append((luma, f, t))

    if args.test_dates:
        test_dates = set(args.test_dates)
    else:
        ranked = sorted(by_date, key=lambda d: float(np.median([x[0] for x in by_date[d]])))
        if args.test_sessions <= 0 or args.test_sessions >= len(ranked):
            raise SystemExit("--test-sessions must be between 1 and sessions-1")
        test_dates = {ranked[0], ranked[-1]}
        for d in ranked[1:-1]:
            if len(test_dates) >= args.test_sessions:
                break
            test_dates.add(d)
        if len(test_dates) > args.test_sessions:
            test_dates = set(sorted(test_dates)[:args.test_sessions])

    unknown = test_dates - set(by_date)
    if unknown:
        raise SystemExit(f"test dates not in data: {sorted(unknown)}")

    split_of = {}
    for luma, f, t in picked:
        split_of[f] = "test" if date_of(t) in test_dates else "train"

    n_test = sum(1 for s in split_of.values() if s == "test")
    print(f"selected {len(picked)}: train={len(picked)-n_test}, test={n_test}, "
          f"test sessions={sorted(test_dates)}")

    tasks = []
    for luma, f, t in picked:
        split = split_of[f]
        orig = out / split / "original" / f.name
        ret = out / split / "retinex" / f.name
        orig.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, orig)
        tasks.append((f, ret, args.quality))

    ret_means = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for ret, m in pool.map(retinex_one, tasks, chunksize=4):
            ret_means[ret.name] = m
    print("retinex done")

    lists = out / "lists"
    for split in ("train", "test"):
        originals = [str(out / split / "original" / f.name) for _, f, _ in picked
                     if split_of[f] == split]
        retinex = [str(out / split / "retinex" / f.name) for _, f, _ in picked
                   if split_of[f] == split]
        write_list(lists / f"{split}.txt", originals)
        write_list(lists / f"{split}_retinex.txt", retinex)

    with (out / "selection.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "datetime", "session_date", "split",
                         "mean_luma", "retinex_mean_luma", "original_path", "retinex_path"])
        for luma, f, t in picked:
            split = split_of[f]
            writer.writerow([f.name, t.isoformat(sep=" "), date_of(t), split,
                             f"{luma:.2f}", f"{ret_means[f.name]:.2f}",
                             out / split / "original" / f.name,
                             out / split / "retinex" / f.name])

    print(f"output: {out}")


if __name__ == "__main__":
    main()
