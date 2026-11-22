# Low-Light Image Enhancement

Multiscale Retinex with Color Restoration (MSRCR) implementation based on:
> Jobson, D. J., Rahman, Z., & Woodell, G. A. (1997). A multiscale retinex for bridging the gap between color images and the human observation of scenes. *IEEE Transactions on Image processing*, 6(7), 965-976.

## Quick Start

### Command Line

```bash
# Use default (moderate) preset
python -m core.lowlight input.jpg output.png

# For extremely dark scenes (recommended for nighttime images)
python -m core.lowlight input.jpg output.png --preset extreme

# Gentle enhancement preserving deep shadows
python -m core.lowlight input.jpg output.png --preset preserve

# Custom parameters
python -m core.lowlight input.jpg output.png --gain 0.7 --shadow-rolloff 10 --contrast-stretch --denoise moderate
```

### Python API

```python
from core.lowlight import LowLightRetinex
from tools.dehaze_images import read_rgb, write_rgb

# Default moderate preset
model = LowLightRetinex()
enhanced = model.enhance(read_rgb("input.jpg"))
write_rgb("output.png", enhanced)

# Extreme preset for very dark scenes
model = LowLightRetinex(
    gain=0.7,                 # Higher gain = stronger enhancement
    shadow_rolloff=12.0,      # Lower rolloff = enhance more dark regions
    contrast_stretch=True,    # Apply adaptive histogram stretching
    denoise='moderate'        # Reduce amplified noise
)
enhanced = model.enhance(image)
```

## Presets

| Preset | Use Case | Parameters |
|--------|----------|------------|
| **extreme** | Very dark scenes with noise (nighttime, tunnels, heavy fog) | `gain=0.7, shadow_rolloff=12.0, contrast_stretch=True, denoise='moderate'` |
| **moderate** | Balanced enhancement with gentle denoising (default) | `gain=0.5, shadow_rolloff=20.0, contrast_stretch=True, denoise='mild'` |
| **preserve** | Gentle enhancement, preserves deep shadows | `gain=0.35, shadow_rolloff=32.0, contrast_stretch=False, denoise=None` |

## Parameters

- **`scales`** (tuple): Gaussian surround scales in pixels. Default: `(15, 80, 250)`
  - Smaller scales capture fine details
  - Larger scales capture global illumination
  
- **`gain`** (float): Display transform gain. Default: `0.35`
  - Higher values = stronger enhancement
  - Range: `0.1` (subtle) to `1.0` (extreme)
  
- **`midpoint`** (float): Display transform midpoint. Default: `128.0`
  - Target brightness level in [0, 255]
  
- **`shadow_rolloff`** (float): Shadow protection threshold. Default: `32.0`
  - Pixels with mean RGB < this value are gradually blended with original
  - Lower values enhance more of the dark regions
  - Set to `0.0` for unblended paper response
  
- **`contrast_stretch`** (bool): Apply adaptive contrast stretching. Default: `False`
  - Recommended for extreme low-light scenes
  - Uses percentile-based histogram stretching (1%, 99%) to improve clarity

- **`denoise`**: Noise reduction strength. Default: `None`
  - `None` or `False`: No denoising
  - `'mild'`: Gentle bilateral filtering (preserves fine details)
  - `'moderate'`: Balanced noise reduction
  - `'strong'`: Aggressive denoising (may soften edges)
  - Essential for extreme low-light enhancement where noise is amplified

## Algorithm Overview

1. **Multiscale Retinex (MSR)**
   ```
   Ri(x,y) = Σ wn · [log Ii(x,y) - log(Fn * Ii(x,y))]
   ```
   where `Fn` are Gaussian surrounds at different scales.

2. **Color Restoration**
   ```
   Ci(x,y) = β · [log(α·Ii) - log(Σ Ii)]
   ```
   with `β=46` and `α=125` from the paper.

3. **Final Output**
   ```
   MSRCRi = Ci · Ri
   ```

4. **Shadow Protection** (engineering addition)
   - Smoothly blends with original in deep shadows using smoothstep interpolation
   - Avoids over-enhancement of pure black regions

5. **Denoising** (optional, recommended for extreme low-light)
   - Bilateral filtering after MSRCR to reduce amplified noise
   - Preserves edges while smoothing flat regions
   - Three strength levels: mild, moderate, strong

6. **Contrast Stretch** (optional post-processing)
   - Per-channel percentile clipping (1%, 99%)
   - Linear stretch to fill [0, 255] range
   - Recovers local contrast compressed by MSRCR

## Examples

```bash
# Compare all presets side-by-side
python tools/compare_lowlight_presets.py input.jpg output_dir/
```

This creates a 2×2 comparison grid and individual enhanced images.

## Testing

```bash
python -m pytest tests/test_lowlight_retinex.py -v
```

Tests verify:
- Correctness against paper equations (pixel-level validation)
- Edge cases (uniform fields, clipped blacks, invalid inputs)
- Shadow protection behavior
- Contrast stretching effectiveness

## Implementation Notes

- **Numerical stability**: Adds `1.0` to signal before taking logarithm to handle `log(0)`
- **Boundary handling**: Uses `cv2.BORDER_REFLECT_101` to avoid edge artifacts
- **Gaussian kernel**: Converts paper's `exp(-r²/c²)` to OpenCV's `exp(-r²/(2σ²))` via `σ = c/√2`
- **Color preservation**: Shadow rolloff uses common RGB mean to avoid coloring neutral shadows

## Citation

If you use this implementation in research, please cite the original paper:

```bibtex
@article{jobson1997multiscale,
  title={A multiscale retinex for bridging the gap between color images and the human observation of scenes},
  author={Jobson, Daniel J and Rahman, Zia-ur and Woodell, Glenn A},
  journal={IEEE Transactions on Image processing},
  volume={6},
  number={7},
  pages={965--976},
  year={1997},
  publisher={IEEE}
}
```
