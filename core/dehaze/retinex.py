import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor


def _fast_gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """大 sigma 高斯模糊加速: 降采样 → 模糊 → 升采样.

    sigma=80 快 ~9x, sigma=200 快 ~60x. 光照分量是低频信号, 降采样几乎无损.
    """
    if sigma <= 30:
        return cv2.GaussianBlur(img, (0, 0), sigma)

    factor = int(sigma / 25)
    h, w = img.shape[:2]
    small_h, small_w = max(h // factor, 1), max(w // factor, 1)
    small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    small_blur = cv2.GaussianBlur(small, (0, 0), sigma / factor)
    return cv2.resize(small_blur, (w, h), interpolation=cv2.INTER_LINEAR)


def _single_scale_retinex(img: np.ndarray, sigma: float) -> np.ndarray:
    blur = _fast_gaussian_blur(img, sigma)
    retinex = np.log10(img + 1.0) - np.log10(blur + 1.0)
    return retinex


def _color_restoration(img: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    img_sum = np.sum(img, axis=2, keepdims=True) + 1.0
    cr = beta * (np.log10(alpha * img + 1.0) - np.log10(img_sum + 1.0))
    return cr


def _simple_color_balance(img: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    h, w, c = img.shape
    pixels = h * w
    flat = img.reshape(pixels, c)
    low_val = np.percentile(flat, low_pct, axis=0)
    high_val = np.percentile(flat, high_pct, axis=0)
    for i in range(c):
        channel = img[:, :, i]
        img[:, :, i] = np.clip((channel - low_val[i]) / (high_val[i] - low_val[i] + 1e-6), 0.0, 1.0)
    return img


class Retinex:
    def __init__(self, scales: list = None, alpha: float = 125.0, beta: float = 46.0,
                 low_pct: float = 1.0, high_pct: float = 99.0,
                 num_workers: int = 1):
        self.scales = scales or [15, 80, 200]
        self.alpha = alpha
        self.beta = beta
        self.low_pct = low_pct
        self.high_pct = high_pct
        self.num_workers = num_workers

    def enhance(self, img: np.ndarray) -> np.ndarray:
        img_float = img.astype(np.float32)
        n_scales = len(self.scales)

        # 多尺度 Retinex: 各尺度独立, 可并行 (OpenCV 释放 GIL)
        if self.num_workers > 1 and n_scales > 1:
            msr = np.zeros_like(img_float)
            with ThreadPoolExecutor(max_workers=min(self.num_workers, n_scales)) as pool:
                futures = {pool.submit(_single_scale_retinex, img_float, s): s
                          for s in self.scales}
                for f in futures:
                    msr += f.result()
            msr /= n_scales
        else:
            msr = np.zeros_like(img_float)
            for sigma in self.scales:
                msr += _single_scale_retinex(img_float, sigma)
            msr /= n_scales

        cr = _color_restoration(img_float, self.alpha, self.beta)
        msrcr = msr * cr
        msrcr = (msrcr - msrcr.min()) / (msrcr.max() - msrcr.min() + 1e-6)
        balanced = _simple_color_balance(msrcr, self.low_pct, self.high_pct)
        return (balanced * 255.0).astype(np.uint8)
