#!/usr/bin/env python3
"""Near-duplicate scan with a correct multi-index hash join.

The earlier bucket-on-low-16-bits trick silently misses pairs whose differing
bits fall inside the bucket key.  This uses 7 bands (pigeonhole: two 64-bit
hashes at Hamming distance <= 6 must share at least one intact band), so every
pair within the threshold is found.

Outputs a hash cache and a cluster map that the split builder consumes:

    <out>.hash.json     {image_name: {"md5": ..., "dhash": ...}}
    <out>.dup.json      {thresh, n_images, n_pairs, n_clusters,
                         exact_dup_groups: [[names]], clusters: {name: cluster_id}}

Usage:
    python dup_scan.py --ds datasets/fog_2cls --out reports/fog_2cls_dup
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
N_BANDS = 7  # >= thresh + 1
BAND_BITS = 10  # 7 * 10 = 70 >= 64


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


def hashes_of(path: Path) -> dict:
    """64-bit coarse hash (candidate recall) + 256-bit fine hash (verification).

    The coarse 8x8 hash alone is not safe to cluster on: dense-fog images are
    nearly textureless, so many of them hash within distance 6 of each other
    and single-linkage chaining merges them into one giant false cluster.
    The 16x16 hash is dominated by noise for flat images, so it rejects those
    pairs while staying ~0 for genuine duplicates.
    """
    return {
        "dhash": dhash(path, 8),
        "dhash16": dhash(path, 16),
    }


class DSU:
    def __init__(self):
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


def find_pairs(hashes: dict[str, int], thresh: int) -> list[tuple[str, str]]:
    """All pairs with hamming distance <= thresh (exact, via 7-band join)."""
    bands: list[dict[int, list[str]]] = [defaultdict(list) for _ in range(N_BANDS)]
    for name, h in hashes.items():
        for b in range(N_BANDS):
            bands[b][(h >> (b * BAND_BITS)) & ((1 << BAND_BITS) - 1)].append(name)
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for name, h in hashes.items():
        cands: set[str] = set()
        for b in range(N_BANDS):
            cands.update(bands[b][(h >> (b * BAND_BITS)) & ((1 << BAND_BITS) - 1)])
        for other in cands:
            if other == name:
                continue
            key = (name, other) if name < other else (other, name)
            if key in seen:
                continue
            if bin(h ^ hashes[other]).count("1") <= thresh:
                seen.add(key)
                out.append(key)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ds", required=True)
    ap.add_argument("--out", required=True, help="output prefix (no extension)")
    ap.add_argument("--thresh", type=int, default=6,
                    help="coarse 8x8 dHash distance for candidate pairs")
    ap.add_argument("--verify-thresh", type=int, default=24,
                    help="fine 16x16 dHash (256 bit) distance required to accept a pair")
    ap.add_argument("--hash-cache", help="reuse an existing hash json")
    ap.add_argument("--max-cluster", type=int, default=100,
                    help="clusters larger than this are reported as suspicious")
    args = ap.parse_args()

    ds = Path(args.ds)
    imgs = [p for p in sorted((ds / "images").iterdir())
            if p.suffix.lower() in IMG_EXT]
    if not imgs:
        print(f"[!] no images under {ds/'images'}", file=sys.stderr)
        return 1

    cache = Path(args.hash_cache) if args.hash_cache else Path(args.out + ".hash.json")
    hashes: dict[str, dict] = {}
    if cache.is_file():
        hashes = json.loads(cache.read_text())
        hashes = {k: v for k, v in hashes.items() if "dhash16" in v}
        print(f"[hash] loaded {len(hashes)} usable cached entries from {cache}")

    todo = [p for p in imgs if p.name not in hashes]
    if todo:
        print(f"[hash] computing md5 + 8x8/16x16 dHash for {len(todo)} images ...")
        for i, p in enumerate(todo, 1):
            h = hashes_of(p)
            h["md5"] = md5_of(p)
            hashes[p.name] = h
            if i % 500 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(hashes))
        print(f"[hash] cached -> {cache}")

    dh = {n: v["dhash"] for n, v in hashes.items()}
    dh16 = {n: v["dhash16"] for n, v in hashes.items()}
    print(f"[pairs] coarse join of {len(dh)} hashes at hamming <= {args.thresh} ...",
          flush=True)
    candidates = find_pairs(dh, args.thresh)
    print(f"[pairs] coarse candidates: {len(candidates)}; verifying with 16x16 "
          f"dHash <= {args.verify_thresh} ...", flush=True)
    pairs = [(a, b) for a, b in candidates
             if bin(dh16[a] ^ dh16[b]).count("1") <= args.verify_thresh]
    print(f"[pairs] accepted after verification: {len(pairs)} "
          f"(rejected {len(candidates) - len(pairs)})")

    dsu = DSU()
    for n in dh:
        dsu.find(n)
    for a, b in pairs:
        dsu.union(a, b)

    by_root: dict[str, list[str]] = defaultdict(list)
    for n in dh:
        by_root[dsu.find(n)].append(n)
    clusters = {n: dsu.find(n) for n in dh}
    multi = {k: v for k, v in by_root.items() if len(v) > 1}

    md5_groups: dict[str, list[str]] = defaultdict(list)
    for n, v in hashes.items():
        md5_groups[v["md5"]].append(n)
    exact = [sorted(v) for v in md5_groups.values() if len(v) > 1]

    suspicious = {k: sorted(v) for k, v in multi.items()
                  if len(v) > args.max_cluster}

    payload = {
        "dataset": str(ds),
        "thresh": args.thresh,
        "verify_thresh": args.verify_thresh,
        "n_images": len(dh),
        "n_pairs_coarse_candidates": len(candidates),
        "n_pairs_verified": len(pairs),
        "n_pairs_within_thresh": len(pairs),
        "n_clusters": len(by_root),
        "n_multi_image_clusters": len(multi),
        "largest_clusters": sorted((len(v) for v in multi.values()), reverse=True)[:10],
        "suspicious_clusters_gt_max": {k: v[:20] for k, v in suspicious.items()},
        "exact_duplicate_groups": exact,
        "clusters": clusters,
        "pairs": [[a, b, bin(dh16[a] ^ dh16[b]).count("1")] for a, b in pairs],
    }
    out = Path(args.out + ".dup.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False))

    sizes = Counter(len(v) for v in multi.values())
    print(f"images={len(dh)}  verified pairs: {len(pairs)}  "
          f"clusters={len(by_root)}  (multi-image clusters: {len(multi)})")
    print(f"cluster size histogram: {dict(sorted(sizes.items())[:10])}")
    print(f"exact duplicate groups: {len(exact)}")
    if suspicious:
        print(f"[!] {len(suspicious)} clusters larger than {args.max_cluster} "
              f"(likely over-merged): {[(len(v)) for v in suspicious.values()]}")
    print(f"written: {cache}  {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
