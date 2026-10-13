"""Create leakage-aware five-fold YOLO manifests for the clean RTTS+DAWN fog set.

Run on the training server with Python, Pillow, NumPy, SciPy and PyYAML installed.
The source dataset is read-only. Outputs are written to ``folds5_grouped_v1``.
"""

from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from scipy.fft import dctn


ROOT = Path("/data/awod_data/processed/fog_2cls_clean_v1")
OUT = ROOT / "folds5_grouped_v1"
BUILD = ROOT / ".folds5_grouped_v1.building"
SEED = 42
PHASH_DISTANCE = 4
VALIDATION_FRACTION_OF_REMAINDER = 0.125
SEARCH_TRIALS = 200


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def perceptual_hash(path: Path) -> int:
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L").resize((32, 32)), dtype=np.float32)
    coefficients = dctn(pixels, norm="ortho")[:8, :8]
    midpoint = np.median(coefficients[1:, :])
    return sum(int(value > midpoint) << i for i, value in enumerate(coefficients.flat))


METRICS = ("images", "dawn", "foggy", "haze", "mist", "person_images", "person_boxes", "vehicle_boxes")
WEIGHTS = (10.0, 5.0, 2.0, 2.0, 2.0, 3.0, 1.0, 1.0)


def group_vector(rows: list[dict[str, str]]) -> tuple[int, ...]:
    return (
        len(rows),
        sum(row["source"] == "dawn" for row in rows),
        sum(row["weather"] == "foggy" for row in rows),
        sum(row["weather"] == "haze" for row in rows),
        sum(row["weather"] == "mist" for row in rows),
        sum(int(row["person"]) > 0 for row in rows),
        sum(int(row["person"]) for row in rows),
        sum(int(row["vehicle"]) for row in rows),
    )


def assign_groups(
    groups: list[list[int]],
    rows: list[dict[str, str]],
    fractions: tuple[float, ...],
    seed: int,
) -> tuple[dict[int, int], float]:
    """Greedy randomized grouping; keep best count and class balance."""
    vectors = [group_vector([rows[i] for i in group]) for group in groups]
    totals = [sum(vector[j] for vector in vectors) for j in range(len(METRICS))]
    targets = [[total * fraction for total in totals] for fraction in fractions]
    denominators = [[max(1.0, target) for target in target_row] for target_row in targets]
    best_assignment: dict[int, int] | None = None
    best_score = float("inf")

    def score(counts: list[list[int]]) -> float:
        return sum(
            WEIGHTS[j] * (counts[k][j] - targets[k][j]) ** 2 / denominators[k][j]
            for k in range(len(fractions))
            for j in range(len(METRICS))
        )

    for trial in range(SEARCH_TRIALS):
        rng = random.Random(seed + trial)
        order = list(range(len(groups)))
        rng.shuffle(order)
        order.sort(key=lambda i: -len(groups[i]))
        counts = [[0] * len(METRICS) for _ in fractions]
        assignment = {}
        for group_id in order:
            vector = vectors[group_id]
            choices = []
            for fold in range(len(fractions)):
                delta = sum(
                    WEIGHTS[j]
                    * (2 * (counts[fold][j] - targets[fold][j]) * vector[j] + vector[j] ** 2)
                    / denominators[fold][j]
                    for j in range(len(METRICS))
                )
                choices.append((delta, rng.random(), fold))
            fold = min(choices)[2]
            assignment[group_id] = fold
            for j in range(len(METRICS)):
                counts[fold][j] += vector[j]
        final = score(counts)
        if final < best_score:
            best_assignment, best_score = assignment, final
    assert best_assignment is not None
    return best_assignment, best_score


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    assert ROOT.is_dir() and not OUT.exists() and not BUILD.exists(), "Dataset missing or output directory already exists"
    with (ROOT / "manifest.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2681
    for row in rows:
        image = ROOT / "images" / row["merged_name"]
        label = ROOT / "labels" / (Path(row["merged_name"]).stem + ".txt")
        assert image.is_file() and label.is_file(), row["merged_name"]

    union = UnionFind(len(rows))
    video_first = {}
    for i, row in enumerate(rows):
        if row["source"] == "rtts":
            match = re.fullmatch(r"(hv\d+)_\d+\.[^.]+", row["original_name"], re.I)
            if match:
                video = match.group(1).lower()
                if video in video_first:
                    union.union(i, video_first[video])
                else:
                    video_first[video] = i

    hashes = [perceptual_hash(ROOT / "images" / row["merged_name"]) for row in rows]
    similar_pairs = []
    for i, value in enumerate(hashes):
        for j in range(i):
            distance = (value ^ hashes[j]).bit_count()
            if distance <= PHASH_DISTANCE:
                union.union(i, j)
                similar_pairs.append({
                    "left": rows[j]["merged_name"],
                    "right": rows[i]["merged_name"],
                    "distance": distance,
                })

    grouped = defaultdict(list)
    for i in range(len(rows)):
        grouped[union.find(i)].append(i)
    groups = [grouped[key] for key in sorted(grouped)]
    index_to_group = {i: gid for gid, group in enumerate(groups) for i in group}

    outer, outer_score = assign_groups(groups, rows, (0.2,) * 5, SEED)
    image_outer = {i: outer[index_to_group[i]] for i in range(len(rows))}
    splits = []
    summary = []
    for test_fold in range(5):
        remaining = [gid for gid in range(len(groups)) if outer[gid] != test_fold]
        subgroups = [groups[gid] for gid in remaining]
        inner, inner_score = assign_groups(
            subgroups,
            rows,
            (VALIDATION_FRACTION_OF_REMAINDER, 1 - VALIDATION_FRACTION_OF_REMAINDER),
            SEED + 10000 + 1000 * test_fold,
        )
        validation_groups = {remaining[gid] for gid, side in inner.items() if side == 0}
        statuses = {}
        for i, row in enumerate(rows):
            gid = index_to_group[i]
            statuses[i] = "test" if outer[gid] == test_fold else "val" if gid in validation_groups else "train"
        fold_dir = BUILD / f"fold{test_fold + 1}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_stats = {"fold": test_fold + 1, "inner_balance_score": round(inner_score, 4)}
        for split in ("train", "val", "test"):
            selected = [rows[i] for i in range(len(rows)) if statuses[i] == split]
            assert selected
            (fold_dir / f"{split}.txt").write_text(
                "".join(str(ROOT / "images" / row["merged_name"]) + "\n" for row in selected),
                encoding="utf-8",
            )
            fold_stats[split] = {
                "images": len(selected),
                "rtts": sum(row["source"] == "rtts" for row in selected),
                "dawn": sum(row["source"] == "dawn" for row in selected),
                "person_images": sum(int(row["person"]) > 0 for row in selected),
                "vehicle_boxes": sum(int(row["vehicle"]) for row in selected),
                "person_boxes": sum(int(row["person"]) for row in selected),
                "weather": dict(Counter(row["weather"] for row in selected)),
            }
        config = {
            "path": str(ROOT),
            "train": str(OUT / f"fold{test_fold + 1}" / "train.txt"),
            "val": str(OUT / f"fold{test_fold + 1}" / "val.txt"),
            "test": str(OUT / f"fold{test_fold + 1}" / "test.txt"),
            "names": {0: "vehicle", 1: "person"},
        }
        (fold_dir / "data.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        for i, status in statuses.items():
            splits.append({"fold": test_fold + 1, "name": rows[i]["merged_name"], "split": status})
        summary.append(fold_stats)

    memberships = defaultdict(set)
    name_to_group = {row["merged_name"]: index_to_group[i] for i, row in enumerate(rows)}
    for item in splits:
        memberships[(item["fold"], name_to_group[item["name"]])].add(item["split"])
    assert all(len(states) == 1 for states in memberships.values())
    assert all(sum(image_outer[i] == fold for i in range(len(rows))) == summary[fold]["test"]["images"] for fold in range(5))
    assert {row["name"] for row in splits if row["split"] == "test"} == {row["merged_name"] for row in rows}

    write_csv(BUILD / "split_manifest.csv", splits)
    group_manifest = [
        {
            "name": row["merged_name"],
            "source": row["source"],
            "original_name": row["original_name"],
            "weather": row["weather"],
            "group_id": index_to_group[i],
            "test_fold": image_outer[i] + 1,
            "phash": f"{hashes[i]:016x}",
        }
        for i, row in enumerate(rows)
    ]
    write_csv(BUILD / "group_manifest.csv", group_manifest)
    report = {
        "seed": SEED,
        "images": len(rows),
        "groups": len(groups),
        "video_sources": len(video_first),
        "video_images": sum(bool(re.fullmatch(r"hv\d+_\d+\.[^.]+", row["original_name"], re.I)) for row in rows if row["source"] == "rtts"),
        "phash_distance_limit": PHASH_DISTANCE,
        "phash_pairs": similar_pairs,
        "outer_balance_score": round(outer_score, 4),
        "folds": summary,
        "group_leaks": 0,
    }
    (BUILD / "fold_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (BUILD / "README.md").write_text(
        "# Grouped five-fold split\n\n"
        "Fixed seed 42. Each fold has one held-out test part; 12.5% of the remaining images "
        "(approximately) form a group-separated validation set for checkpoint selection. "
        "All other images form the training set. Each image appears in test exactly once.\n\n"
        "RTTS frames with the same hv video source stay in one split. Images with perceptual "
        "hash distance <= 4 stay in one split, including cross-dataset near duplicates. "
        "Image counts, source, weather and person/vehicle counts were balanced with randomized "
        "greedy search. Grouping based on filenames and perceptual hash cannot prove that every "
        "unidentified video sequence is separated.\n\n"
        "Use `foldN/data.yaml` with Ultralytics YOLO. Keep the test split untouched while "
        "selecting epochs or hyperparameters. Use `fold_summary.json` and both CSV manifests "
        "to audit the split.\n",
        encoding="utf-8",
    )
    BUILD.rename(OUT)
    print(json.dumps({"output": str(OUT), "groups": len(groups), "phash_pairs": len(similar_pairs), "folds": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
