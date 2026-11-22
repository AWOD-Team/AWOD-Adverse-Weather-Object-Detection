import unittest

import numpy as np

from core.lowlight import LowLightRetinex


class LowLightRetinexTests(unittest.TestCase):
    def test_center_pixel_matches_paper_equations(self):
        # The center is far from the boundary for both Gaussian scales.
        image = np.random.default_rng(4).integers(5, 60, (61, 61, 3), dtype=np.uint8)
        signal = image.astype(float) + 1
        pixel = signal[30, 30]
        y, x = np.mgrid[-30:31, -30:31]
        responses = []
        for scale in (2, 4):
            kernel = np.exp(-(x*x + y*y) / scale**2)
            surround = (signal * kernel[:, :, None]).sum(axis=(0, 1)) / kernel.sum()
            responses.append(np.log(pixel / surround))
        expected = 128 + 0.35 * 46 * np.log(125 * pixel / pixel.sum()) * np.mean(responses, axis=0)
        actual = LowLightRetinex(scales=(2, 4), shadow_rolloff=0).enhance(image)
        np.testing.assert_allclose(actual[30, 30], np.clip(expected, 0, 255), atol=1)

    def test_dark_detail_and_clipped_black(self):
        image = np.full((48, 64, 3), 10, np.uint8)
        image[14:34, 23:41] = 24
        image[:3] = 0
        original = image.copy()
        result = LowLightRetinex().enhance(image)
        self.assertEqual(result.shape, image.shape)
        self.assertEqual(result.dtype, image.dtype)
        self.assertGreater(int(result[24, 32, 0]) - int(result[24, 5, 0]), 14)
        self.assertGreater(np.median(result), np.median(image))
        np.testing.assert_array_equal(result[:3], 0)
        np.testing.assert_array_equal(result[..., 0], result[..., 2])
        np.testing.assert_array_equal(image, original)

    def test_deep_shadow_transitions_smoothly_from_black(self):
        ramp = np.repeat(np.arange(64, dtype=np.uint8), 3).reshape(1, 64, 3)
        protected = LowLightRetinex().enhance(ramp)
        unblended = LowLightRetinex(shadow_rolloff=0).enhance(ramp)
        self.assertLessEqual(int(protected[0, 2, 0]), 4)
        np.testing.assert_array_equal(protected[:, 32:], unblended[:, 32:])

    def test_flat_fields_and_invalid_inputs(self):
        model = LowLightRetinex()
        for color in ((0, 0, 0), (0, 8, 13), (255, 255, 255)):
            image = np.full((3, 4, 3), color, dtype=np.uint8)
            np.testing.assert_array_equal(model.enhance(image), image)
        for image in (np.zeros((4, 4)), np.zeros((0, 1, 3), np.uint8),
                      np.zeros((3, 3, 3), np.float32)):
            with self.assertRaises(ValueError):
                model.enhance(image)

    def test_contrast_stretch_improves_dynamic_range(self):
        """Contrast stretching should expand compressed mid-gray output."""
        # Create a very dark image with subtle structure
        image = np.random.default_rng(42).integers(5, 25, (64, 64, 3), dtype=np.uint8)

        without_stretch = LowLightRetinex(gain=0.5, contrast_stretch=False).enhance(image)
        with_stretch = LowLightRetinex(gain=0.5, contrast_stretch=True).enhance(image)

        # Contrast stretch should increase the range between min and max
        range_without = without_stretch.max() - without_stretch.min()
        range_with = with_stretch.max() - with_stretch.min()
        self.assertGreater(range_with, range_without * 1.2)

    def test_extreme_preset_parameters(self):
        """Extreme preset should produce stronger enhancement than default."""
        image = np.random.default_rng(7).integers(8, 35, (48, 64, 3), dtype=np.uint8)

        default = LowLightRetinex().enhance(image)
        extreme = LowLightRetinex(gain=0.7, shadow_rolloff=12.0, contrast_stretch=True).enhance(image)

        # Extreme preset should produce brighter output
        self.assertGreater(extreme.mean(), default.mean())
        # And better utilize the full dynamic range
        self.assertGreater(extreme.std(), default.std())

    def test_denoising_reduces_variance(self):
        """Bilateral filtering should reduce noise variance."""
        # Create noisy dark image
        rng = np.random.default_rng(99)
        image = np.full((64, 64, 3), 20, dtype=np.uint8)
        noise = rng.integers(-8, 9, (64, 64, 3), dtype=np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        without_denoise = LowLightRetinex(gain=0.5, denoise=None).enhance(image)
        with_mild = LowLightRetinex(gain=0.5, denoise='mild').enhance(image)
        with_moderate = LowLightRetinex(gain=0.5, denoise='moderate').enhance(image)

        # Denoising should reduce variance (noise amplification)
        var_none = without_denoise.var()
        var_mild = with_mild.var()
        var_mod = with_moderate.var()

        self.assertLess(var_mild, var_none * 0.9)
        self.assertLess(var_mod, var_mild * 0.9)

        # All outputs should still enhance the original
        self.assertGreater(with_moderate.mean(), image.mean() * 3)
