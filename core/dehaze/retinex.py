import cv2
import numpy as np


def _single_scale_retinex(img: np.ndarray, sigma: float) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
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
                 low_pct: float = 1.0, high_pct: float = 99.0):
        self.scales = scales or [15, 80, 200]
        self.alpha = alpha
        self.beta = beta
        self.low_pct = low_pct
        self.high_pct = high_pct

    def enhance(self, img: np.ndarray) -> np.ndarray:
        img_float = img.astype(np.float32)
        msr = np.zeros_like(img_float)
        n_scales = len(self.scales)
        for sigma in self.scales:
            msr += _single_scale_retinex(img_float, sigma)
        msr /= n_scales
        cr = _color_restoration(img_float, self.alpha, self.beta)
        msrcr = msr * cr
        msrcr = (msrcr - msrcr.min()) / (msrcr.max() - msrcr.min() + 1e-6)
        balanced = _simple_color_balance(msrcr, self.low_pct, self.high_pct)
        return (balanced * 255.0).astype(np.uint8)
