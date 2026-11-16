# AWOD-Adverse-Weather-Object-Detection
Object detection for vehicles and pedestrians under adverse weather conditions using computer vision.

## DCP dehazing

`core/dehaze/dcp.py` accepts `uint8` RGB images. It follows the dark-channel
prior's atmospheric-light estimate, coarse transmission, and radiance recovery.
Transmission is refined by default with a fast guided filter (an engineering
approximation, **not** the 2009 paper's soft matting). To use the paper's
soft-matting step, select `DarkChannelPrior(refinement="soft_matting")` in Python.
This mode requires SciPy and is capped at 40,000 pixels by default because the
sparse solve is expensive; resize the input or raise `matting_max_pixels` only
if memory permits. `refinement="none"` reproduces the previous coarse-map
behavior. All modes keep the same image dimensions and RGB/uint8 format.

These changes improve numerical safety and edge alignment; they do not by
themselves establish improved YOLO mAP. Compare detection on the *same* fixed
splits and labels before adopting dehazed data for training.

## Retinex-BDCP dehazing

`core/dehaze/retinex_bdcp.py` provides `RetinexBDCP`: multiscale Retinex
illumination, weighted L2-L1 smoothing, bright/dark-channel transmission fusion,
guided refinement, and physically bounded radiance recovery. It uses fixed
parameters and requires only NumPy/OpenCV; it does not train or alter YOLO.
This is a paper-informed engineering combination, not an exact reproduction
of the nighttime paper's nonconvex solver or a demonstrated mAP improvement.

```bash
python -m tools.dehaze_images --input /path/to/images --output /path/to/new_run --save-maps
python -m tools.dehaze_images --input /path/to/images --output /path/to/dcp_run --method dcp
python -m unittest discover -s tests -v
```

Outputs preserve filenames and image geometry under `new_run/images/`, with
parameters and solver diagnostics in `run.json`. Existing results are never
overwritten. Labels and fold membership must remain paired with the originals.
See [公式、论文差异、参数与实验说明](docs/retinex_bdcp.md).
