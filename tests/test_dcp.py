"""Focused regression tests for the DCP algorithm (no YOLO dependency)."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "core" / "dehaze" / "dcp.py"
SPEC = importlib.util.spec_from_file_location("awod_dcp", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
DarkChannelPrior = MODULE.DarkChannelPrior


class DCPTests(unittest.TestCase):
    def test_black_image_is_unchanged(self):
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        result = DarkChannelPrior().dehaze(image)
        np.testing.assert_array_equal(result, image)
        self.assertIsNot(result, image)

    def test_zero_atmospheric_channel_is_finite(self):
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        image[:, :, 0] = 180
        image[5:15, 5:15, 1] = 80
        result = DarkChannelPrior().dehaze(image)
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.shape, image.shape)

    def test_refinement_changes_coarse_map_and_keeps_bounds(self):
        image = np.full((40, 40, 3), 180, dtype=np.uint8)
        image[:, :20] = [30, 45, 65]
        image_float = image.astype(np.float32) / 255.0
        dehazer = DarkChannelPrior(win_size=15)
        coarse = dehazer._transmission(
            image_float,
            dehazer._atmospheric_light(image_float, dehazer._dark_channel(image_float)),
        )
        refined = dehazer._guided_refine(image_float, coarse)
        self.assertEqual(refined.shape, coarse.shape)
        self.assertTrue(np.isfinite(refined).all())
        self.assertGreater(np.max(np.abs(refined - coarse)), 0.01)

    def test_soft_matting_on_flat_image(self):
        try:
            import scipy  # noqa: F401
        except ImportError:
            self.skipTest("scipy not installed")
        image = np.full((8, 8, 3), 100, dtype=np.uint8)
        result = DarkChannelPrior(refinement="soft_matting").dehaze(image)
        np.testing.assert_allclose(result, image, atol=1)

    def test_soft_matting_handles_nonuniform_image_and_size_limit(self):
        try:
            import scipy  # noqa: F401
        except ImportError:
            self.skipTest("scipy not installed")
        image = np.random.default_rng(42).integers(
            25, 230, (12, 12, 3), dtype=np.uint8)
        result = DarkChannelPrior(refinement="soft_matting").dehaze(image)
        self.assertEqual(result.shape, image.shape)
        self.assertEqual(result.dtype, np.uint8)
        with self.assertRaisesRegex(ValueError, "at most"):
            DarkChannelPrior(refinement="soft_matting",
                             matting_max_pixels=100).dehaze(image)

    def test_invalid_input_and_parameters(self):
        with self.assertRaises(ValueError):
            DarkChannelPrior(win_size=0)
        with self.assertRaises(ValueError):
            DarkChannelPrior(refinement="unknown")
        with self.assertRaises(ValueError):
            DarkChannelPrior().dehaze(np.zeros((4, 4), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
