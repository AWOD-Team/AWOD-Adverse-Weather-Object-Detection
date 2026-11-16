"""Formula, numerical and batch-I/O regression checks; no YOLO weights needed."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from core.dehaze import RetinexBDCP
from core.dehaze.retinex_bdcp import _adjoint, _smooth_illumination, _sobel
from tools.dehaze_images import read_rgb, write_rgb


class RetinexBDCPTests(unittest.TestCase):
    def test_constant_images_preserve_radiance_including_zero_channels(self):
        model = RetinexBDCP()
        for color in ([0, 0, 0], [255, 255, 255], [0, 140, 230], [100, 100, 100]):
            with self.subTest(color=color):
                image = np.full((8, 10, 3), color, dtype=np.uint8)
                np.testing.assert_allclose(model.dehaze(image), image, atol=1)

    def test_small_images_and_resolution_changes_keep_geometry(self):
        model = RetinexBDCP(light_max_side=16)
        for shape in ((1, 1, 3), (1, 20, 3), (20, 1, 3), (31, 45, 3)):
            with self.subTest(shape=shape):
                image = np.random.default_rng(42).integers(0, 256, shape, dtype=np.uint8)
                original = image.copy()
                result, maps = model.dehaze_with_details(image)
                self.assertEqual(result.shape, shape)
                self.assertEqual(result.dtype, np.uint8)
                np.testing.assert_array_equal(original, image)
                for value in maps.values():
                    if isinstance(value, np.ndarray):
                        self.assertTrue(np.isfinite(value).all())
                        self.assertGreaterEqual(float(value.min()), 0)
                        self.assertLessEqual(float(value.max()), 1)
                self.assertGreaterEqual(float(maps["transmission"].min()), model.dcp.t0 - 1e-6)

    def test_recovery_inverts_spatial_scattering_model(self):
        rng = np.random.default_rng(3)
        clear = rng.uniform(0, 1, (12, 17, 3)).astype(np.float32)
        light = rng.uniform(0.2, 1, clear.shape).astype(np.float32)
        t = rng.uniform(0.1, 1, clear.shape[:2]).astype(np.float32)
        observed = clear * t[..., None] + light * (1 - t[..., None])
        np.testing.assert_allclose(RetinexBDCP._recover(observed, light, t), clear, atol=1e-6)

    def test_priors_recover_known_transmission_at_black_and_white_objects(self):
        model = RetinexBDCP(win_size=1, omega=1)
        light = np.full((2, 2, 3), 0.6, np.float32)
        t = 0.4
        dark, _, _, _ = model._transmissions(light * (1 - t), light)
        _, bright, _, _ = model._transmissions(t + light * (1 - t), light)
        np.testing.assert_allclose(dark, t, atol=1e-6)
        np.testing.assert_allclose(bright, t, atol=1e-6)

    def test_radiance_bound_prevents_clipping_with_spatial_colored_light(self):
        rng = np.random.default_rng(17)
        image = rng.uniform(0, 1, (10, 20, 3)).astype(np.float32)
        light = rng.uniform(0, 1, image.shape).astype(np.float32)
        light[:, :2] = 0
        light[:, -2:] = 1
        bound = RetinexBDCP._radiance_bound(image, light)
        restored = RetinexBDCP._recover(image, light, np.maximum(bound, 0.1))
        self.assertGreaterEqual(float(restored.min()), -1e-6)
        self.assertLessEqual(float(restored.max()), 1 + 1e-6)
        # A smaller t must violate at least one channel when the bound is active.
        invalid = RetinexBDCP._recover(image, light, np.maximum(bound * 0.9, 1e-6))
        violates = (invalid.min(axis=2) < -1e-6) | (invalid.max(axis=2) > 1 + 1e-6)
        self.assertTrue(violates[bound > 0.2].all())

    def test_end_to_end_recovery_is_in_range_before_final_clipping(self):
        image = np.random.default_rng(19).integers(0, 256, (48, 64, 3), dtype=np.uint8)
        model = RetinexBDCP()
        _, details = model.dehaze_with_details(image)
        raw = model._recover(image.astype(np.float32) / 255,
                             details["illumination"], details["transmission"])
        self.assertGreaterEqual(float(raw.min()), -1e-6)
        self.assertLessEqual(float(raw.max()), 1 + 1e-6)

    def test_sobel_adjoint_identity(self):
        rng = np.random.default_rng(5)
        x, gx, gy = (rng.normal(size=(13, 17, 3)) for _ in range(3))
        dx, dy = _sobel(x)
        self.assertAlmostEqual(float(np.sum(dx * gx + dy * gy)),
                               float(np.sum(x * _adjoint(gx, gy))), places=10)

    def test_smoothing_identity_without_regularization(self):
        initial = np.random.default_rng(1).uniform(0.1, 0.9, (13, 19, 3)).astype(np.float32)
        result, info = _smooth_illumination(initial, 0, 0, 100, 1e-7)
        np.testing.assert_allclose(result, initial, atol=1e-6)
        self.assertTrue(info["converged"])

    def test_convex_smoothing_reduces_objective(self):
        rng = np.random.default_rng(6)
        initial = rng.uniform(0.3, 0.7, (24, 32, 3)).astype(np.float32)
        result, _ = _smooth_illumination(initial, 0.2, 0.02, 200, 1e-6)
        gx0, gy0 = _sobel(initial)
        wx = 1 / (1 + 10 * np.abs(gx0).max(axis=2, keepdims=True))
        wy = 1 / (1 + 10 * np.abs(gy0).max(axis=2, keepdims=True))

        def energy(value):
            gx, gy = _sobel(value)
            return (0.5 * np.sum((value - initial) ** 2)
                    + 0.2 * np.sum((wx * gx) ** 2 + (wy * gy) ** 2)
                    + 0.02 * np.sum(np.abs(wx * gx) + np.abs(wy * gy)))

        self.assertLess(energy(result), energy(initial))

    def test_global_light_ablation_and_determinism(self):
        model = RetinexBDCP(local_weight=0)
        image = np.random.default_rng(8).integers(20, 230, (18, 25, 3), dtype=np.uint8)
        output, maps = model.dehaze_with_details(image)
        np.testing.assert_allclose(maps["illumination"],
                                   np.broadcast_to(maps["atmospheric_light"], image.shape))
        np.testing.assert_array_equal(output, model.dehaze(image))
        json.dumps(model.parameters())

    def test_invalid_input_and_parameters(self):
        for kwargs in ({"omega": float("nan")}, {"bright_gamma": 0}, {"light_iterations": 0},
                       {"scales": []}, {"scales": [float("inf")]}, {"local_weight": 1.1},
                       {"light_l1": -1}, {"win_size": 2}, {"light_max_side": 2.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RetinexBDCP(**kwargs)
        for image in (None, np.zeros((4, 4), np.uint8), np.zeros((4, 4, 4), np.uint8),
                      np.zeros((0, 4, 3), np.uint8), np.zeros((4, 4, 3), np.float32)):
            with self.assertRaises(ValueError):
                RetinexBDCP().dehaze(image)


class BatchDehazeTests(unittest.TestCase):
    def test_unicode_paths_mirrored_layout_maps_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "输入"
            original = source / "val" / "雾图.png"
            image = np.full((12, 16, 3), [45, 100, 210], np.uint8)
            write_rgb(original, image)
            destination = root / "result"
            command = [sys.executable, "-m", "tools.dehaze_images", "--input", str(source),
                       "--output", str(destination), "--save-maps"]
            run = subprocess.run(command, capture_output=True)
            self.assertEqual(run.returncode, 0, run.stderr.decode(errors="replace"))
            np.testing.assert_array_equal(read_rgb(original), image)
            np.testing.assert_array_equal(read_rgb(destination / "images/val/雾图.png"), image)
            with np.load(destination / "maps/val/雾图.png.npz") as maps:
                self.assertEqual(maps["transmission"].shape, (12, 16))
            report = json.loads((destination / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["processed"], 1)
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)

    def test_import_does_not_load_detector_dependencies(self):
        run = subprocess.run([sys.executable, "-c",
                              "import sys; from core import RetinexBDCP; "
                              "assert 'torch' not in sys.modules; "
                              "assert 'ultralytics' not in sys.modules"], capture_output=True)
        self.assertEqual(run.returncode, 0, run.stderr.decode(errors="replace"))


if __name__ == "__main__":
    unittest.main()
