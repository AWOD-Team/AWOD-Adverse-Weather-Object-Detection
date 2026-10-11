#!/usr/bin/env python3
"""Dataset tools for AWOD fog detection: audit / remap / folds / summarize.

Examples:
  python dataset_tools.py audit     --root DS --names vehicle person -o audit.csv
  python dataset_tools.py remap     --src DS --dst DS2 --map "0:0,1:0,3:0,2:1"
  python dataset_tools.py folds     --audit audit.csv --root DS --names vehicle person -k 5 --out-dir folds
  python dataset_tools.py folds     --audit audit.csv --root DS --names vehicle person -k 5 --out-dir folds --fog-manifest DS/fog_manifest.csv
  python dataset_tools.py summarize --runs-dir runs -o summary.csv
"""
import argparse, csv, json, math, re, shutil, sys
from collections import defaultdict
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SKIP_DIRS = {"images_fog", "labels_fog"}


def list_images(root):
    root = Path(root)
    return [p for p in sorted(root.rglob("*"))
            if p.suffix.lower() in IMG_EXT and p.is_file() and not (SKIP_DIRS & set(p.parts))]


def label_for(img, root):
    root = Path(root)
    rel = img.relative_to(root)
    parts = list(rel.parts)
    cands = []
    if parts and parts[0] == "images":
        cands.append(root / "labels" / Path(*parts[1:]).with_suffix(".txt"))
    cands.append(img.with_suffix(".txt"))
    cands.append(root / "labels" / Path(*parts[:-1]) / (img.stem + ".txt"))
    for c in cands:
        if c.is_file():
            return c
    return None


def stem_group(name):
    s = re.sub(r"[_\-\.]?\d+$", "", Path(name).stem)
    return s if s else Path(name).stem


def dhash(p, size=8):
    from PIL import Image
    with Image.open(p) as im:
        im = im.convert("L").resize((size + 1, size), Image.LANCZOS)
        px = list(im.getdata())
    bits, k = 0, 0
    for r in range(size):
        for c in range(size):
            if px[r * (size + 1) + c] > px[r * (size + 1) + c + 1]:
                bits |= 1 << k
            k += 1
    return bits


def cluster_phash(images, thresh):
    out, groups = {}, []
    for p in images:
        h = dhash(p)
        for gh, members in groups:
            if bin(gh ^ h).count("1") <= thresh:
                members.append(p)
                out[str(p)] = len(groups) - 1
                break
        else:
            groups.append((h, [p]))
            out[str(p)] = len(groups) - 1
    return {k: "phash%d" % v for k, v in out.items()}


def cluster_time(images, gap):
    pairs = sorted(((p, p.stat().st_mtime) for p in images), key=lambda x: x[1])
    out, gid, prev = {}, 0, None
    for p, t in pairs:
        if prev is not None and t - prev > gap:
            gid += 1
        out[str(p)] = "t%d" % gid
        prev = t
    return out


def parse_remap(s):
    if not s:
        return None
    return {int(a): int(b) for a, b in (p.split(":") for p in s.split(",") if ":" in p)}


def read_labels(lp, remap, wh):
    counts, small = defaultdict(int), 0
    if lp and lp.is_file():
        for line in lp.read_text(errors="ignore").splitlines():
            f = line.split()
            if len(f) < 5:
                continue
            c = int(float(f[0]))
            if remap is not None and c not in remap:
                continue
            if remap is not None:
                c = remap[c]
            counts[c] += 1
            if float(f[3]) * float(f[4]) * wh[0] * wh[1] < 1024:
                small += 1
    return counts, small


def image_size(p):
    try:
        from PIL import Image
        with Image.open(p) as im:
            return im.size
    except Exception:
        return (0, 0)


def cmd_audit(a):
    root = Path(a.root)
    images = list_images(root)
    if not images:
        print("[!] no images under %s" % root, file=sys.stderr)
        return 1
    remap = parse_remap(a.remap)
    ph = cluster_phash(images, a.phash_thresh) if a.group_mode == "phash" else None
    tm = cluster_time(images, a.time_gap) if a.group_mode == "time" else None
    gm = None
    if a.groups_csv:
        gm = {}
        with open(a.groups_csv, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                gm[row["image"]] = row["group_id"]
    rows, per_class, n_small, n_box, gcount = [], defaultdict(int), 0, 0, defaultdict(int)
    for p in images:
        lp = label_for(p, root)
        wh = image_size(p)
        counts, small = read_labels(lp, remap, wh)
        if ph is not None:
            gid = ph[str(p)]
        elif tm is not None:
            gid = tm[str(p)]
        elif gm is not None:
            gid = gm.get(p.name) or gm.get(str(p.relative_to(root))) or stem_group(p.name)
        else:
            gid = stem_group(p.name)
        gcount[gid] += 1
        for c, n in counts.items():
            per_class[c] += n
        n_small += small
        n_box += sum(counts.values())
        rows.append({"image": str(p.relative_to(root)), "group_id": gid,
                     "width": wh[0], "height": wh[1], "n_boxes": sum(counts.values()),
                     "n_small": small, "classes": "|".join("%d:%d" % (c, n) for c, n in sorted(counts.items()))})
    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    names = a.names or ["class%d" % c for c in sorted(per_class)]
    sizes = sorted(gcount.values())
    print("images=%d groups=%d boxes=%d small(<32^2)=%d (%d%%)" % (len(images), len(gcount), n_box, n_small, 100 * n_small / max(n_box, 1)))
    print("images per group: min=%d median=%d max=%d" % (sizes[0], sizes[len(sizes) // 2], sizes[-1]))
    for c in sorted(per_class):
        nm = names[c] if c < len(names) else "class%d" % c
        print("  class %d (%s): %d instances" % (c, nm, per_class[c]))
    print("written: %s" % a.out)
    return 0


def cmd_remap(a):
    src, dst, remap = Path(a.src), Path(a.dst), parse_remap(a.map)
    if not remap:
        print("[!] --map required, e.g. \"0:0,2:0,1:1,3:1\"", file=sys.stderr)
        return 1
    n_img = n_drop = 0
    for p in list_images(src):
        rel = p.relative_to(src)
        parts = list(rel.parts)
        inner = Path(*parts[1:]) if parts and parts[0] == "images" else rel
        lp = label_for(p, src)
        out_img = dst / "images" / inner
        out_lab = dst / "labels" / inner.with_suffix(".txt")
        out_img.parent.mkdir(parents=True, exist_ok=True)
        out_lab.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out_img)
        n_img += 1
        lines = []
        if lp and lp.is_file():
            for line in lp.read_text(errors="ignore").splitlines():
                f = line.split()
                if len(f) < 5:
                    continue
                c = int(float(f[0]))
                if c in remap:
                    lines.append(" ".join([str(remap[c])] + f[1:5]))
                else:
                    n_drop += 1
        out_lab.write_text("\n".join(lines) + ("\n" if lines else ""))
    print("copied %d images (dropped %d boxes outside map) -> %s" % (n_img, n_drop, dst))
    return 0


def cmd_folds(a):
    with open(a.audit, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    root, out_dir, k = Path(a.root), Path(a.out_dir), a.k
    out_dir.mkdir(parents=True, exist_ok=True)
    gcount, gimgs = defaultdict(lambda: defaultdict(int)), defaultdict(list)
    for r in rows:
        gimgs[r["group_id"]].append(r["image"])
        for kv in r["classes"].split("|"):
            if kv:
                c, n = kv.split(":")
                gcount[r["group_id"]][int(c)] += int(n)

    folds, tally = [[] for _ in range(k)], [defaultdict(int) for _ in range(k)]
    for g in sorted(gcount, key=lambda g: -sum(gcount[g].values())):
        i = min(range(k), key=lambda j: sum(tally[j].values()))
        folds[i].append(g)
        for c, n in gcount[g].items():
            tally[i][c] += n

    fog_of_group = defaultdict(list)
    if a.fog_manifest and Path(a.fog_manifest).is_file():
        with open(a.fog_manifest, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                fog_of_group[row["source_group"]].append(row["fog_image"])

    names = a.names or ["class%d" % c for c in sorted(set().union(*[set(t) for t in tally]))]
    summary = {}
    for i in range(k):
        val_groups = set(folds[i])
        train_groups = set(g for j in range(k) if j != i for g in folds[j])
        tr = [str(root / im) for g in train_groups for im in gimgs[g]]
        extra = [str(root / im) for g in train_groups for im in fog_of_group.get(g, [])]
        va = [str(root / im) for g in val_groups for im in gimgs[g]]
        excluded = [im for g in val_groups for im in fog_of_group.get(g, [])]
        (out_dir / ("fold%d_train.txt" % (i + 1))).write_text("\n".join(tr + extra) + "\n", encoding="utf-8")
        (out_dir / ("fold%d_val.txt" % (i + 1))).write_text("\n".join(va) + "\n", encoding="utf-8")
        yml = ["path: %s" % root.resolve(),
               "train: %s" % (out_dir / ("fold%d_train.txt" % (i + 1))).resolve(),
               "val: %s" % (out_dir / ("fold%d_val.txt" % (i + 1))).resolve(),
               "names:"]
        yml += ["  %d: %s" % (c, nm) for c, nm in enumerate(names)]
        (out_dir / ("data_fold%d.yaml" % (i + 1))).write_text("\n".join(yml) + "\n", encoding="utf-8")
        tr_inst = defaultdict(int)
        for j in range(k):
            if j != i:
                for c, n in tally[j].items():
                    tr_inst[c] += n
        summary["fold%d" % (i + 1)] = {
            "train_images": len(tr), "train_fog_images": len(extra), "val_images": len(va),
            "val_groups": len(val_groups), "fog_excluded_for_val": len(excluded),
            "train_instances": {str(c): n for c, n in sorted(tr_inst.items())},
            "val_instances": {str(c): n for c, n in sorted(tally[i].items())},
        }
        print("fold%d: train %d (+fog %d) / val %d | train inst %s | val inst %s | fog excluded for val %d"
              % (i + 1, len(tr), len(extra), len(va), dict(sorted(tr_inst.items())), dict(sorted(tally[i].items())), len(excluded)))
    (out_dir / "fold_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("written to %s" % out_dir)
    return 0


def cmd_summarize(a):
    files = sorted(Path(a.runs_dir).rglob("results.csv"))
    if not files:
        print("[!] no results.csv under %s" % a.runs_dir, file=sys.stderr)
        return 1
    per_key, out_rows = defaultdict(list), []
    print("%-34s %7s %9s %8s" % ("run", "best_ep", "mAP50-95", "mAP50"))
    for fp in files:
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(fp, newline=""))]
        if not rows:
            continue
        k95 = next((k for k in rows[0] if "mAP50-95" in k), None)
        k50 = next((k for k in rows[0] if "mAP50" in k and "95" not in k), None)
        if not k95:
            continue
        best = max(rows, key=lambda r: float(r[k95] or 0))
        run = fp.parent.name
        tag = re.sub(r"_?fold\d+.*$", "", run)
        per_key[tag].append(float(best[k95]))
        print("%-34s %7d %9.4f %8.4f" % (run, int(float(best["epoch"])), float(best[k95]), float(best.get(k50) or 0)))
    print("\n=== mean +- std over folds ===")
    for tag, vals in sorted(per_key.items()):
        m = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals)) if len(vals) > 1 else 0.0
        print("%-24s n=%d  mAP50-95 = %.4f +- %.4f" % (tag, len(vals), m, sd))
        out_rows.append({"tag": tag, "n_folds": len(vals), "mAP50_95_mean": round(m, 4), "mAP50_95_std": round(sd, 4)})
    if a.out and out_rows:
        with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
        print("written: %s" % a.out)
    return 0


def main():
    ap = argparse.ArgumentParser(description="AWOD dataset tools")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("audit")
    p.add_argument("--root", required=True)
    p.add_argument("--names", nargs="*")
    p.add_argument("--remap")
    p.add_argument("--group-mode", choices=["stem", "phash", "time"], default="stem")
    p.add_argument("--groups-csv")
    p.add_argument("--phash-thresh", type=int, default=6)
    p.add_argument("--time-gap", type=float, default=2.0)
    p.add_argument("-o", "--out", default="audit.csv")
    p.set_defaults(fn=cmd_audit)
    p = sub.add_parser("remap")
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--map", required=True)
    p.set_defaults(fn=cmd_remap)
    p = sub.add_parser("folds")
    p.add_argument("--audit", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--names", nargs="*")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fog-manifest")
    p.set_defaults(fn=cmd_folds)
    p = sub.add_parser("summarize")
    p.add_argument("--runs-dir", required=True)
    p.add_argument("-o", "--out")
    p.set_defaults(fn=cmd_summarize)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())