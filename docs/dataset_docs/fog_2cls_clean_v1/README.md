# Fog 2-class clean dataset v1

Merged clean real-fog dataset for YOLO object detection.

- RTTS: 2419 images (all retained clean RTTS)
- DAWN fog-like subset: 262 images (`foggy`, `haze`, `mist` only)
- Total: 2681 images and labels
- Classes: `0 vehicle`, `1 person`
- RTTS boxes: vehicle 24890, person 5267
- DAWN boxes: vehicle 1700, person 103

Files are hard-linked from the two source clean datasets. Do not edit images or labels in place.
No train/val/test split is included yet. Split by source/sequence groups before training to prevent leakage.
`manifest.csv` records provenance and original filenames.
