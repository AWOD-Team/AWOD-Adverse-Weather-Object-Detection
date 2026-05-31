import cv2
import numpy as np

from core.dehaze.dcp import DarkChannelPrior
from core.dehaze.retinex import Retinex


class HybridDehazer:
    def __init__(self, omega: float = 0.95, win_size: int = 15, t0: float = 0.1,
                 retinex_scales: list = None, alpha: float = 125.0, beta: float = 46.0,
                 fusion_weight: float = 0.5):
        self.dcp = DarkChannelPrior(omega=omega, win_size=win_size, t0=t0)
        self.retinex = Retinex(scales=retinex_scales, alpha=alpha, beta=beta)
        self.fusion_weight = fusion_weight

    def dehaze(self, img: np.ndarray) -> np.ndarray:
        dcp_result = self.dcp.dehaze(img).astype(np.float32)
        retinex_result = self.retinex.enhance(img).astype(np.float32)
        fused = self.fusion_weight * dcp_result + (1.0 - self.fusion_weight) * retinex_result
        fused = np.clip(fused, 0.0, 255.0)
        return fused.astype(np.uint8)
