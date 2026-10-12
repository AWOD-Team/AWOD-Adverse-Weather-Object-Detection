#!/usr/bin/env python3
"""Day-2 split builder for the fog week (see 下一周任务计划.md §4.3/§4.4).

Builds, from a flat YOLO dataset (images/ + labels/):

  * source groups, derived from the file name (RTTS names look like
    ``GSGL_Baidu_078`` -> group ``GSGL_Baidu``; a whole group is never split),
  * a grouped k-fold split (default 5), balanced on instance counts,
  * nested small-sample subsets inside each fold's TRAIN pool only:
    600 ⊂ 1200 ⊂ 2300 ⊂ full, balanced on vehicle/person box counts,
  * a leak audit across the produced lists (group / file name / near-duplicate).

Outputs (under --out-dir):

    folds/fold{k}_train.txt, folds/fold{k}_val.txt, folds/data_fold{k}.yaml
    manifests/fold{k}_train_{600,1200,2300,full}.txt
    fold_summary.json
    leak_report.json

Usage:
    python make_folds.py --ds datasets/fog_2cls --k 5 \
        --out-dir . --targets 600 1200 2300 --hash-cache reports/rt_hash.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
SMALL_AREA = 32 * 32
CLASSES = ["vehicle", "person"]


def stem_group(name: str) -> str:
    s = re.sub(r"[_\-\.]?\d+$", "", Path(name).stem)
    return s if s else Path(name).stem


class DSU:
    """Union-find: one split group = one connected component."""

    def __init__(self) -> None:
        self.p: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def assign_groups(recs, dup_pairs=None, dup_clusters=None, max_group=400):
    """Group key = (source prefix) ∪ (near-duplicate merges under a size cap).

    Two rules, in order:

    1. every image sharing a file-name prefix (``GSGL_Baidu_078`` ->
       ``GSGL_Baidu``) is one group — same city/engine means same source pool;
    2. verified near-duplicate pairs are merged cheapest-edge-first, but a merge
       that would push a group past ``max_group`` images is skipped.

    Rule 2 exists because single-linkage chaining is not safe here: dense-fog
    images are nearly textureless, so one bad cluster can chain a dozen large
    prefixes into a 1000+ image group that can no longer be balanced across
    folds.  The cap keeps the dedup useful and the folds usable.
    """
    dsu = DSU()
    size: dict[str, int] = {}
    for r in recs:
        root = dsu.find(r["name"])
        size[root] = size.get(root, 0) + 1

    def union(a: str, b: str, cap: int | None = None) -> bool:
        ra, rb = dsu.find(a), dsu.find(b)
        if ra == rb:
            return False
        if cap is not None and size[ra] + size[rb] > cap:
            return False
        dsu.p[rb] = ra
        size[ra] = size[ra] + size[rb]
        size.pop(rb, None)
        return True

    by_prefix = defaultdict(list)
    for r in recs:
        by_prefix[stem_group(r["name"])].append(r["name"])
    for names in by_prefix.values():
        for n in names[1:]:
            union(names[0], n)

    merged = already = blocked = 0
    if dup_pairs:
        for a, b, _d in sorted(dup_pairs, key=lambda x: x[2]):
            ra, rb = dsu.find(a), dsu.find(b)
            if ra == rb:
                already += 1
            elif union(a, b, cap=max_group):
                merged += 1
            else:
                blocked += 1
    elif dup_clusters:
        by_cluster = defaultdict(list)
        for r in recs:
            c = dup_clusters.get(r["name"])
            if c:
                by_cluster[c].append(r["name"])
        for names in by_cluster.values():
            for n in names[1:]:
                ra, rb = dsu.find(names[0]), dsu.find(n)
                if ra == rb:
                    already += 1
                elif union(names[0], n, cap=max_group):
                    merged += 1
                else:
                    blocked += 1

    gid = {r["name"]: dsu.find(r["name"]) for r in recs}
    groups = defaultdict(list)
    for r in recs:
        groups[gid[r["name"]]].append(r)
    return dict(groups), gid, {"merged_pairs": merged,
                               "pairs_already_same_group": already,
                               "pairs_blocked_by_cap": blocked,
                               "max_group": max_group}


def group_by_gid(recs, gid):
    out = defaultdict(list)
    for r in recs:
        out[gid[r["name"]]].append(r)
    return dict(out)


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


def load_pool(ds: Path):
    """Return per-image records: name, path, boxes per class, small-box count."""
    img_dir, lbl_dir = ds / "images", ds / "labels"
    imgs = [p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in IMG_EXT]
    recs = []
    for img in imgs:
        lp = lbl_dir / f"{img.stem}.txt"
        per = Counter()
        small = 0
        if lp.is_file():
            for line in lp.read_text(errors="ignore").splitlines():
                f = line.split()
                if len(f) < 5:
                    continue
                try:
                    c = int(float(f[0]))
                    w, h = float(f[3]), float(f[4])
                except ValueError:
                    continue
                nm = CLASSES[c] if c < len(CLASSES) else f"class{c}"
                per[nm] += 1
                if w * h * 640 * 640 < SMALL_AREA:
                    small += 1
        recs.append({"name": img.name, "path": img, "group": stem_group(img.name),
                     "boxes": dict(per), "n_box": sum(per.values()), "small": small})
    return recs


def group_of(recs):
    g = defaultdict(list)
    for r in recs:
        g[r["group"]].append(r)
    return g

def pick_ordering(groups, target_ratio, targets, seed, draws=300):
    """Find a group ordering whose prefixes best match the class ratio."""
    gids = sorted(groups)
    best, best_score = None, None
    rng = random.Random(seed)
    for _ in range(draws):
        order = gids[:]
        rng.shuffle(order)
        score = 0.0
        idx = 0
        p_boxes = v_boxes = 0
        for t in targets:
            while idx < len(order) and sum(
                    len(groups[g]) for g in order[:idx]) < t:
                for r in groups[order[idx]]:
                    p_boxes += r["boxes"].get("person", 0)
                    v_boxes += r["boxes"].get("vehicle", 0)
                idx += 1
            tot = p_boxes + v_boxes
            share = p_boxes / tot if tot else 0.0
            score += abs(share - target_ratio)
        if best_score is None or score < best_score:
            best, best_score = order, score
    return best


def prefix_for(groups, order, target):
    idx, n = 0, 0
    while idx < len(order) and n < target:
        n += len(groups[order[idx]])
        idx += 1
    return order[:idx]


def stats_of(recs):
    p = sum(r["boxes"].get("person", 0) for r in recs)
    v = sum(r["boxes"].get("vehicle", 0) for r in recs)
    return {"images": len(recs), "vehicle_boxes": v, "person_boxes": p,
            "small_boxes": sum(r["small"] for r in recs),
            "person_share": round(p / (p + v), 4) if p + v else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ds", required=True, help="flat dataset with images/ labels/")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--targets", type=int, nargs="*", default=[600, 1200, 2300])
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--dup-clusters",
                    help=".dup.json from dup_scan.py; near-duplicate clusters are "
                         "merged into the split groups")
    ap.add_argument("--near-dup-thresh", type=int, default=6)
    ap.add_argument("--max-group", type=int, default=400,
                    help="refuse a near-dup merge that would make a group larger "
                         "than this many images (guards against chaining)")
    args = ap.parse_args()

    ds = Path(args.ds)
    out = Path(args.out_dir)
    folds_dir = out / "folds"
    man_dir = out / "manifests"
    folds_dir.mkdir(parents=True, exist_ok=True)
    man_dir.mkdir(parents=True, exist_ok=True)

    recs = load_pool(ds)
    if not recs:
        print(f"[!] no images under {ds/'images'}", file=sys.stderr)
        return 1

    dup_clusters = None
    dup_pairs = None
    dup_meta = {}
    if args.dup_clusters and Path(args.dup_clusters).is_file():
        payload = json.loads(Path(args.dup_clusters).read_text())
        dup_clusters = payload.get("clusters")
        dup_pairs = payload.get("pairs")
        dup_meta = {k: payload.get(k) for k in
                    ("thresh", "verify_thresh", "n_images",
                     "n_pairs_coarse_candidates", "n_pairs_verified",
                     "n_pairs_within_thresh", "n_clusters",
                     "n_multi_image_clusters", "largest_clusters",
                     "exact_duplicate_groups")}
        print(f"[dup] {args.dup_clusters}: coarse="
              f"{dup_meta.get('n_pairs_coarse_candidates')} "
              f"verified={dup_meta.get('n_pairs_verified')} "
              f"clusters={dup_meta.get('n_multi_image_clusters')} multi-image")
        if not dup_pairs:
            print("[dup] no pair list in json; falling back to whole clusters")

    groups, gid, group_meta = assign_groups(
        recs, dup_pairs=dup_pairs, dup_clusters=dup_clusters,
        max_group=args.max_group)
    total = stats_of(recs)
    sizes = sorted((len(v) for v in groups.values()), reverse=True)
    print(f"pool: images={total['images']} groups={len(groups)} "
          f"(largest={sizes[0]}, median={sizes[len(sizes)//2]}) "
          f"vehicle={total['vehicle_boxes']} person={total['person_boxes']} "
          f"person_share={total['person_share']}")
    print(f"[groups] near-dup merges applied={group_meta['merged_pairs']} "
          f"already-same-group={group_meta['pairs_already_same_group']} "
          f"blocked_by_cap={group_meta['pairs_blocked_by_cap']} "
          f"(cap={group_meta['max_group']} images/group)")

    # ---- grouped k-fold -------------------------------------------------
    # "must link" components: groups joined by any verified near-duplicate pair
    # are assigned as a unit, which makes a cross-fold near-duplicate pair
    # impossible by construction (no iterative repair, no oscillation).
    gu = DSU()
    for g in groups:
        gu.find(g)
    if dup_pairs:
        for a, b, _d in dup_pairs:
            ga, gb = gid.get(a), gid.get(b)
            if ga is not None and gb is not None:
                gu.union(ga, gb)
    comp_groups: dict[str, list[str]] = defaultdict(list)
    for g in groups:
        comp_groups[gu.find(g)].append(g)
    comps = sorted(comp_groups.values(),
                   key=lambda gs: -sum(len(groups[g]) for g in gs))

    folds: list[list[str]] = [[] for _ in range(args.k)]
    n_img = [0] * args.k
    for comp in comps:
        size = sum(len(groups[g]) for g in comp)
        i = min(range(args.k), key=lambda j: n_img[j])
        folds[i].extend(comp)
        n_img[i] += size
    print(f"[folds] {len(comps)} must-link components assigned; "
          f"fold sizes {n_img}")

    target_ratio = total["person_share"]
    summary, manifests = {}, {}
    for i in range(args.k):
        val_groups = set(folds[i])
        train_groups = [g for j in range(args.k) if j != i for g in folds[j]]
        train_recs = [r for g in train_groups for r in groups[g]]
        val_recs = [r for g in val_groups for r in groups[g]]

        (folds_dir / f"fold{i+1}_train.txt").write_text(
            "\n".join(str(r["path"].resolve()) for r in train_recs) + "\n",
            encoding="utf-8")
        (folds_dir / f"fold{i+1}_val.txt").write_text(
            "\n".join(str(r["path"].resolve()) for r in val_recs) + "\n",
            encoding="utf-8")
        (folds_dir / f"data_fold{i+1}.yaml").write_text(
            f"path: {ds.resolve()}\n"
            f"train: {(folds_dir / f'fold{i+1}_train.txt').resolve()}\n"
            f"val: {(folds_dir / f'fold{i+1}_val.txt').resolve()}\n"
            "names:\n" + "\n".join(f"  {c}: {n}" for c, n in enumerate(CLASSES))
            + "\n", encoding="utf-8")

        # ---- nested subsets inside the training pool only
        tr_groups = group_by_gid(train_recs, gid)
        ratio = (sum(r["boxes"].get("person", 0) for r in train_recs)
                 / max(1, sum(r["n_box"] for r in train_recs)))
        order = pick_ordering(tr_groups, ratio, args.targets, args.seed + i)
        fold_manifest = {}
        for t in sorted(set(args.targets)) + [10 ** 9]:
            if t == 10 ** 9:
                sel, tag = order, "full"
            else:
                sel, tag = prefix_for(tr_groups, order, t), str(t)
            sel_recs = [r for g in sel for r in tr_groups[g]]
            (man_dir / f"fold{i+1}_train_{tag}.txt").write_text(
                "\n".join(str(r["path"].resolve()) for r in sel_recs) + "\n",
                encoding="utf-8")
            fold_manifest[tag] = stats_of(sel_recs)
            manifests[f"fold{i+1}_{tag}"] = len(sel_recs)

        summary[f"fold{i+1}"] = {
            "train_images": len(train_recs), "val_images": len(val_recs),
            "train_groups": len(tr_groups), "val_groups": len(val_groups),
            "train_instances": dict(stats_of(train_recs)),
            "val_instances": dict(stats_of(val_recs)),
            "nested_subsets": fold_manifest,
        }
        print(f"fold{i+1}: train {len(train_recs)} ({len(tr_groups)} groups) / "
              f"val {len(val_recs)} ({len(val_groups)} groups) | "
              f"val person_share={stats_of(val_recs)['person_share']}")
        for tag, st in fold_manifest.items():
            print(f"    subset {tag:>5s}: images={st['images']:5d} "
                  f"v={st['vehicle_boxes']:6d} p={st['person_boxes']:6d} "
                  f"share={st['person_share']}")

    # ---- leak audit over the written lists -------------------------------
    # 一个 group 内部既包含"同来源前缀"也包含"近重复簇"，所以 group 不跨折
    # 就同时保证了同源不跨折与近重复不跨折。这里再独立复核一遍。
    leaks = {"group_overlap": [], "name_overlap": [], "cluster_overlap": [],
             "dup_meta": dup_meta, "nested_violations": [], "group_meta": group_meta}
    gid_of = {r["name"]: gid[r["name"]] for r in recs}

    # which fold does each image belong to (an image is in exactly one val list)
    fold_of: dict[str, int] = {}
    for i in range(args.k):
        for p in (folds_dir / f"fold{i+1}_val.txt").read_text().split():
            fold_of[Path(p).name] = i + 1

    # the decisive check: is any verified near-duplicate pair split across folds?
    cross_fold = []
    if dup_pairs:
        for a, b, d in dup_pairs:
            fa, fb = fold_of.get(a), fold_of.get(b)
            if fa and fb and fa != fb:
                cross_fold.append({"a": a, "b": b, "d16": d, "folds": [fa, fb]})
    leaks["near_dup_pairs_cross_fold"] = {
        "n": len(cross_fold),
        "total_verified_pairs": len(dup_pairs),
        "examples": cross_fold[:10],
    }

    for i in range(args.k):
        tr_names = [Path(p).name for p in
                    (folds_dir / f"fold{i+1}_train.txt").read_text().split()]
        va_names = [Path(p).name for p in
                    (folds_dir / f"fold{i+1}_val.txt").read_text().split()]
        gtr = {gid_of[n] for n in tr_names if n in gid_of}
        gva = {gid_of[n] for n in va_names if n in gid_of}
        shared = gtr & gva
        if shared:
            leaks["group_overlap"].append({"fold": i + 1, "n": len(shared),
                                           "examples": sorted(shared)[:5]})
            leaks["cluster_overlap"].append({"fold": i + 1, "n": len(shared),
                                             "examples": sorted(shared)[:5]})
        ntr, nva = set(tr_names), set(va_names)
        if ntr & nva:
            leaks["name_overlap"].append({"fold": i + 1, "n": len(ntr & nva),
                                          "examples": sorted(ntr & nva)[:5]})
        # nested subsets must be inside this fold's train pool
        pool = set(tr_names)
        for tag in sorted(set(args.targets)):
            f = man_dir / f"fold{i+1}_train_{tag}.txt"
            if not f.is_file():
                continue
            sel = [Path(p).name for p in f.read_text().split()]
            if not set(sel) <= pool:
                leaks["nested_violations"].append(
                    {"fold": i + 1, "subset": tag, "outside": len(set(sel) - pool)})

    (out / "fold_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "leak_report.json").write_text(
        json.dumps(leaks, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== leak audit ===")
    print(f"  group/cluster overlap train vs val : {len(leaks['group_overlap'])} folds")
    print(f"  file name overlap train vs val     : {len(leaks['name_overlap'])} folds")
    print(f"  nested subset outside train pool   : {len(leaks['nested_violations'])}")
    cf = leaks["near_dup_pairs_cross_fold"]
    print(f"  verified near-dup pairs split across folds: {cf['n']} "
          f"of {cf['total_verified_pairs']}")
    print(f"  near-dup clusters merged as groups : "
          f"{dup_meta.get('n_multi_image_clusters', 'n/a')}")
    print(f"written: {out}/fold_summary.json  {out}/leak_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
