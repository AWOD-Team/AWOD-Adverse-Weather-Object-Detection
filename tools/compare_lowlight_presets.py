"""Compare low-light enhancement presets on a test image.

Usage: python tools/compare_lowlight_presets.py input.jpg output_dir/
Creates side-by-side comparison with original, moderate, and extreme presets.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from core.lowlight import LowLightRetinex
from tools.dehaze_images import read_rgb, write_rgb


def create_comparison(image: np.ndarray) -> np.ndarray:
    """Create a 2x2 grid: original, preserve, moderate, extreme."""
    h, w = image.shape[:2]

    # Compute all variants
    original = image
    preserve = LowLightRetinex(
        gain=0.35, shadow_rolloff=32.0, contrast_stretch=False, denoise=None
    ).enhance(image)
    moderate = LowLightRetinex(
        gain=0.5, shadow_rolloff=20.0, contrast_stretch=True, denoise="mild"
    ).enhance(image)
    extreme = LowLightRetinex(
        gain=0.7, shadow_rolloff=12.0, contrast_stretch=True, denoise="moderate"
    ).enhance(image)

    # Add labels
    def add_label(img, text):
        labeled = img.copy()
        cv2.putText(labeled, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 255), 2, cv2.LINE_AA)
        return labeled

    top_row = np.hstack([
        add_label(original, "Original"),
        add_label(preserve, "Preserve"),
    ])
    bottom_row = np.hstack([
        add_label(moderate, "Moderate (default)"),
        add_label(extreme, "Extreme"),
    ])

    return np.vstack([top_row, bottom_row])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input low-light image")
    parser.add_argument("output_dir", type=Path, help="Output directory")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    image = read_rgb(args.input)
    comparison = create_comparison(image)

    output_path = args.output_dir / f"{args.input.stem}_comparison.png"
    write_rgb(output_path, comparison)
    print(f"Comparison saved to: {output_path}")

    # Also save individual versions
    for name, params in [
        ("preserve", {"gain": 0.35, "shadow_rolloff": 32.0, "contrast_stretch": False, "denoise": None}),
        ("moderate", {"gain": 0.5, "shadow_rolloff": 20.0, "contrast_stretch": True, "denoise": "mild"}),
        ("extreme", {"gain": 0.7, "shadow_rolloff": 12.0, "contrast_stretch": True, "denoise": "moderate"}),
    ]:
        enhanced = LowLightRetinex(**params).enhance(image)
        individual_path = args.output_dir / f"{args.input.stem}_{name}.png"
        write_rgb(individual_path, enhanced)
        print(f"  {name}: {individual_path}")


if __name__ == "__main__":
    main()
