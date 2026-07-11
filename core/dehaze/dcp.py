import cv2
import numpy as np


class DarkChannelPrior:
    def __init__(self, omega: float = 0.95, win_size: int = 15, t0: float = 0.1):
        self.omega = omega
        self.win_size = win_size
        self.t0 = t0

    def _dark_channel(self, img: np.ndarray) -> np.ndarray:
        min_rgb = img.min(axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.win_size, self.win_size))
        dark = cv2.erode(min_rgb, kernel)
        return dark

    def _atmospheric_light(self, img: np.ndarray, dark: np.ndarray) -> np.ndarray:
        h, w = dark.shape
        num_pixels = h * w
        top_ratio = 0.001
        top_n = max(int(num_pixels * top_ratio), 1)
        flat_dark = dark.ravel()
        indices = np.argpartition(flat_dark, -top_n)[-top_n:]
        flat_img = img.reshape(-1, 3)
        candidates = flat_img[indices]
        brightest_index = indices[np.argmax(candidates.sum(axis=1))]
        return flat_img[brightest_index].astype(np.float32, copy=True)

    def _transmission(self, img: np.ndarray, A: np.ndarray) -> np.ndarray:
        norm = img.astype(np.float32) / A.reshape(1, 1, 3)
        return 1.0 - self.omega * self._dark_channel(norm)

    def dehaze(self, img: np.ndarray) -> np.ndarray:
        img_float = img.astype(np.float32) / 255.0
        A = self._atmospheric_light(img_float, self._dark_channel(img_float).astype(np.float32))
        t = self._transmission(img_float, A)
        t = np.maximum(t, self.t0)
        J = np.zeros_like(img_float)
        for c in range(3):
            J[:, :, c] = (img_float[:, :, c] - A[c]) / t + A[c]
        J = np.clip(J, 0.0, 1.0)
        return (J * 255.0).astype(np.uint8)
