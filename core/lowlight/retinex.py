"""Multiscale Retinex with color restoration for low-light RGB images.

Reference: Jobson, Rahman & Woodell (1997), sections III–V and table II.
DOI: 10.1109/83.597272.
"""

import cv2
import numpy as np


class LowLightRetinex:
    """Equal-weight MSRCR. Scales are Gaussian space constants in pixels.

    The display transform is shared across channels and images. Its gain and
    midpoint are engineering defaults for natural logarithms and uint8 output.
    shadow_rolloff is the mean RGB level above which enhancement is fully
    applied. Set it to zero for the unblended paper response.

    contrast_stretch: If True, apply adaptive histogram stretching after MSRCR
    to improve output clarity. Recommended for extreme low-light scenes.

    denoise: Noise reduction strength. Options:
        None or False: No denoising
        'mild': Gentle bilateral filtering (preserves fine detail)
        'moderate': Balanced noise reduction
        'strong': Aggressive denoising (may soften edges)
    """

    def __init__(self, scales=(15, 80, 250), *, gain=0.35, midpoint=128.0,
                 shadow_rolloff=32.0, contrast_stretch=False, denoise=None):
        self.scales = tuple(float(scale) for scale in scales)
        if not self.scales or not all(np.isfinite(s) and s > 0 for s in self.scales):
            raise ValueError("scales must contain finite positive space constants")
        if not np.isfinite(gain) or gain <= 0:
            raise ValueError("gain must be finite and positive")
        if not np.isfinite(midpoint) or not 0 <= midpoint <= 255:
            raise ValueError("midpoint must be between 0 and 255")
        if not np.isfinite(shadow_rolloff) or not 0 <= shadow_rolloff <= 255:
            raise ValueError("shadow_rolloff must be between 0 and 255")
        if denoise not in (None, False, 'mild', 'moderate', 'strong'):
            raise ValueError("denoise must be None, False, 'mild', 'moderate', or 'strong'")
        self.gain = float(gain)
        self.midpoint = float(midpoint)
        self.shadow_rolloff = float(shadow_rolloff)
        self.contrast_stretch = bool(contrast_stretch)
        self.denoise = denoise

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """Enhance an H×W×3 uint8 RGB image, preserving its pixel geometry."""
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                or image.ndim != 3 or image.shape[2] != 3 or image.size == 0):
            raise ValueError("expected a nonempty H×W×3 uint8 RGB image")
        # A spatially uniform field has no reflectance detail to enhance.
        if np.all(image == image[0, 0]):
            return image.copy()

        signal = image.astype(np.float32) + 1.0  # one digital count for log(0)
        log_signal = np.log(signal)
        response = np.zeros_like(signal)
        for scale in self.scales:
            # Paper: exp(-r²/c²). OpenCV: exp(-r²/(2 sigma²)).
            surround = cv2.GaussianBlur(signal, (0, 0), scale / np.sqrt(2.0),
                                        borderType=cv2.BORDER_REFLECT_101)
            response += log_signal - np.log(surround)
        response /= len(self.scales)

        chromaticity = signal / signal.sum(axis=2, keepdims=True)
        color_restoration = 46.0 * np.log(125.0 * chromaticity)
        display = self.midpoint + self.gain * color_restoration * response
        if self.shadow_rolloff > 0:
            # A common RGB weight avoids coloring nearly neutral deep shadows.
            level = image.mean(axis=2, dtype=np.float32, keepdims=True)
            x = np.minimum(level / self.shadow_rolloff, 1.0)
            weight = x * x * (3.0 - 2.0 * x)
            display = image + weight * (display - image)

        result = np.rint(np.clip(display, 0, 255)).astype(np.uint8)

        if self.denoise:
            result = self._apply_denoising(result)

        if self.contrast_stretch:
            result = self._adaptive_contrast_stretch(result)

        return result

    def _apply_denoising(self, image: np.ndarray) -> np.ndarray:
        """Apply bilateral filtering to reduce noise while preserving edges.

        Bilateral filter smooths similar-colored regions while keeping sharp
        transitions intact, making it ideal for denoising enhanced low-light
        images where noise is amplified but edge information is valuable.
        """
        if self.denoise == 'mild':
            # Gentle: preserve fine details, remove obvious noise
            return cv2.bilateralFilter(image, d=5, sigmaColor=50, sigmaSpace=50)
        elif self.denoise == 'moderate':
            # Balanced: good noise reduction with acceptable edge preservation
            return cv2.bilateralFilter(image, d=7, sigmaColor=75, sigmaSpace=75)
        elif self.denoise == 'strong':
            # Aggressive: maximum noise removal, may soften edges
            return cv2.bilateralFilter(image, d=9, sigmaColor=100, sigmaSpace=100)
        return image

    def _adaptive_contrast_stretch(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive histogram stretching to improve contrast clarity.

        Uses per-channel percentile clipping to avoid outlier saturation,
        then stretches the histogram to fill [0, 255]. This recovers local
        contrast that MSRCR may compress into the mid-gray range.
        """
        stretched = np.zeros_like(image)
        for c in range(3):
            channel = image[:, :, c]
            # Clip extreme outliers using percentiles (1%, 99% more aggressive)
            p_low, p_high = np.percentile(channel, (1, 99))
            if p_high > p_low:
                # Linear stretch to [0, 255]
                stretched_ch = (channel.astype(np.float32) - p_low) * (255.0 / (p_high - p_low))
                stretched[:, :, c] = np.clip(stretched_ch, 0, 255).astype(np.uint8)
            else:
                stretched[:, :, c] = channel
        return stretched
