# Grouped five-fold split

Fixed seed 42. Each fold has one held-out test part; 12.5% of the remaining images (approximately) form a group-separated validation set for checkpoint selection. All other images form the training set. Each image appears in test exactly once.

RTTS frames with the same hv video source stay in one split. Images with perceptual hash distance <= 4 stay in one split, including cross-dataset near duplicates. Image counts, source, weather and person/vehicle counts were balanced with randomized greedy search. Grouping based on filenames and perceptual hash cannot prove that every unidentified video sequence is separated.

Use `foldN/data.yaml` with Ultralytics YOLO. Keep the test split untouched while selecting epochs or hyperparameters. Use `fold_summary.json` and both CSV manifests to audit the split.
