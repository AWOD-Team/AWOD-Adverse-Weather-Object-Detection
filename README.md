# AWOD-Adverse-Weather-Object-Detection
Object detection for vehicles and pedestrians under adverse weather conditions using computer vision.

## DCP dehazing

`core/dehaze/dcp.py` accepts `uint8` RGB images. It follows the dark-channel
prior's atmospheric-light estimate, coarse transmission, and radiance recovery.
Transmission is refined by default with a fast guided filter (an engineering
approximation, **not** the 2009 paper's soft matting). To use the paper's
soft-matting step, select `DarkChannelPrior(refinement="soft_matting")` or run
`python -m demo.demo_only_dcp --image IMAGE --method dcp --dcp-refinement soft_matting`.
This mode requires SciPy and is capped at 40,000 pixels by default because the
sparse solve is expensive; resize the input or raise `matting_max_pixels` only
if memory permits. `refinement="none"` reproduces the previous coarse-map
behavior. All modes keep the same image dimensions and RGB/uint8 format.

These changes improve numerical safety and edge alignment; they do not by
themselves establish improved YOLO mAP. Compare detection on the *same* fixed
splits and labels before adopting dehazed data for training.
