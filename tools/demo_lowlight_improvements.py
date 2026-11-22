"""Demonstrate the impact of denoising and adjusted thresholds on low-light enhancement.

This script creates a visual comparison showing how the improved parameters
(moderate denoise, adjusted shadow_rolloff) address the "foggy" appearance
that can occur with extreme low-light enhancement.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from core.lowlight import LowLightRetinex
from tools.dehaze_images import read_rgb, write_rgb


def create_before_after_comparison(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compare old aggressive preset vs new balanced extreme preset."""

    # Old aggressive approach (high gain, low rolloff, no denoise)
    old_aggressive = LowLightRetinex(
        gain=0.8,
        shadow_rolloff=8.0,
        contrast_stretch=True,
        denoise=None
    ).enhance(image)

    # New balanced extreme (moderate gain, balanced rolloff, with denoise)
    new_balanced = LowLightRetinex(
        gain=0.7,
        shadow_rolloff=12.0,
        contrast_stretch=True,
        denoise='moderate'
    ).enhance(image)

    return image, old_aggressive, new_balanced


def add_text_overlay(img: np.ndarray, text: str, position: str = 'top') -> np.ndarray:
    """Add descriptive text to an image."""
    labeled = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    color = (0, 255, 255)  # Cyan, visible on both dark and light

    # Calculate text size and position
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    h, w = img.shape[:2]

    if position == 'top':
        x = (w - text_width) // 2
        y = 40
    else:  # bottom
        x = (w - text_width) // 2
        y = h - 20

    # Add semi-transparent background for readability
    overlay = labeled.copy()
    cv2.rectangle(overlay, (x - 10, y - text_height - 10),
                  (x + text_width + 10, y + baseline + 5), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, labeled, 0.4, 0, labeled)

    # Draw text
    cv2.putText(labeled, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return labeled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Input extremely dark image')
    parser.add_argument('output', type=Path, help='Output comparison image')
    args = parser.parse_args()

    print(f"Processing {args.input}...")
    image = read_rgb(args.input)

    original, old_result, new_result = create_before_after_comparison(image)

    # Create labeled versions
    original_labeled = add_text_overlay(original, "Original (very dark)")
    old_labeled = add_text_overlay(old_result, "Old: High noise, foggy")
    new_labeled = add_text_overlay(new_result, "New: Denoised, balanced")

    # Stack vertically for comparison
    comparison = np.vstack([
        original_labeled,
        old_labeled,
        new_labeled
    ])

    write_rgb(args.output, comparison)
    print(f"Comparison saved to: {args.output}")
    print("\nKey improvements:")
    print("  1. Moderate denoising reduces noise amplification")
    print("  2. Adjusted shadow_rolloff (8.0 → 12.0) prevents over-enhancement")
    print("  3. Reduced gain (0.8 → 0.7) balances brightness and quality")
    print("  4. Result: clearer image without foggy appearance")


if __name__ == '__main__':
    main()
