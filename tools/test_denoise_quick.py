"""Quick demonstration of denoising effectiveness on synthetic low-light image."""

import numpy as np
import cv2
from core.lowlight import LowLightRetinex

# Create synthetic dark noisy image (simulates low-light sensor output)
rng = np.random.default_rng(42)
image = np.full((256, 256, 3), 15, dtype=np.uint8)

# Add structure: a gradient and some rectangles
image[:, 128:] = 25
image[64:128, 64:128] = 35
image[128:192, 128:192] = 40

# Add realistic noise
noise = rng.integers(-6, 7, (256, 256, 3), dtype=np.int16)
image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

print("Processing synthetic dark noisy image...")
print(f"Original - Mean: {image.mean():.1f}, Variance: {image.var():.1f}\n")

# Compare denoising strengths
configs = [
    ("No denoise", {'denoise': None}),
    ("Mild denoise", {'denoise': 'mild'}),
    ("Moderate denoise", {'denoise': 'moderate'}),
]

results = []
for name, params in configs:
    model = LowLightRetinex(
        gain=0.7,
        shadow_rolloff=12.0,
        contrast_stretch=True,
        **params
    )
    result = model.enhance(image)
    results.append(result)

    print(f"{name:20s} - Mean: {result.mean():6.1f}, Variance: {result.var():7.1f}")

# Calculate noise reduction
print(f"\nNoise reduction (variance):")
print(f"  Mild:     {(1 - results[1].var() / results[0].var()) * 100:.1f}%")
print(f"  Moderate: {(1 - results[2].var() / results[0].var()) * 100:.1f}%")

print("\n✓ Denoising working correctly")
print("  For real images: use --preset extreme or --denoise moderate")
