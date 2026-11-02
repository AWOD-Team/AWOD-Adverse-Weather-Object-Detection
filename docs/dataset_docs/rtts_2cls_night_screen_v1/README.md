# RTTS night/low-light screened v1

Source: `/data/awod_data/processed/rtts_2cls_clean_v1`. Images and YOLO labels are copied without changing pixels or boxes.

Visual screening retained night and dusk/dawn driving/pedestrian scenes with at least three labeled vehicle/person instances. Mean brightness was used only to generate candidates; daytime fog was excluded by visual review. Close-up portraits, posed photos, and nearly empty scenes were excluded. See `selection.csv` for lighting and counts.

This subset is **not split** into train/val/test. Do not train or report validation results with the same images in both sets. When merging with DAWN, create group-aware disjoint splits, especially for `hv` video frames.
