import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from core.dehaze.dcp import DarkChannelPrior
from core.dehaze.retinex import Retinex


class HybridDehazer:
    def __init__(self, omega: float = 0.95, win_size: int = 15, t0: float = 0.1,
                 retinex_scales: list = None, alpha: float = 125.0, beta: float = 46.0,
                 fusion_weight: float = 0.5, num_workers: int = 2):
        self.dcp = DarkChannelPrior(omega=omega, win_size=win_size, t0=t0)
        self.retinex = Retinex(scales=retinex_scales, alpha=alpha, beta=beta,
                               num_workers=num_workers)
        self.fusion_weight = fusion_weight
        self.num_workers = num_workers

    def dehaze(self, img: np.ndarray) -> np.ndarray:
        if self.num_workers <= 1:
            # 单线程：顺序执行
            dcp_result = self.dcp.dehaze(img).astype(np.float32)
            retinex_result = self.retinex.enhance(img).astype(np.float32)
        else:
            # 多线程：DCP 与 Retinex 并行（OpenCV 底层释放 GIL，线程有效）
            with ThreadPoolExecutor(max_workers=2) as pool:
                future_dcp = pool.submit(self._dcp_wrapper, img)
                future_retinex = pool.submit(self._retinex_wrapper, img)
                dcp_result = future_dcp.result()
                retinex_result = future_retinex.result()

        fused = self.fusion_weight * dcp_result + (1.0 - self.fusion_weight) * retinex_result
        fused = np.clip(fused, 0.0, 255.0)
        return fused.astype(np.uint8)

    def _dcp_wrapper(self, img: np.ndarray) -> np.ndarray:
        return self.dcp.dehaze(img).astype(np.float32)

    def _retinex_wrapper(self, img: np.ndarray) -> np.ndarray:
        return self.retinex.enhance(img).astype(np.float32)
