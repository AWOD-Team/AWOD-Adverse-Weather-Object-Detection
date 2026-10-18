"""Dark channel prior dehazing for RGB uint8 images.

The optional soft-matting mode follows He, Sun & Tang (CVPR 2009). The
default guided filter is a faster edge-aware approximation, not that paper's
soft-matting method.
"""

import cv2
import numpy as np


class DarkChannelPrior:
    def __init__(self, omega: float = 0.95, win_size: int = 15,
                 t0: float = 0.1, refinement: str = "guided",
                 guided_radius: int = 20, guided_eps: float = 1e-3,
                 matting_lambda: float = 1e-4,
                 matting_max_pixels: int = 40000):
        if not 0 < omega <= 1:
            raise ValueError("omega must be in (0, 1]")
        if win_size < 1 or win_size % 2 != 1:
            raise ValueError("win_size must be a positive odd integer")
        if not 0 < t0 <= 1:
            raise ValueError("t0 must be in (0, 1]")
        if refinement not in ("none", "guided", "soft_matting"):
            raise ValueError("refinement must be none, guided, or soft_matting")
        if guided_radius < 1 or guided_eps <= 0:
            raise ValueError("guided_radius and guided_eps must be positive")
        if matting_lambda <= 0 or matting_max_pixels < 1:
            raise ValueError("matting_lambda and matting_max_pixels must be positive")
        self.omega = omega
        self.win_size = win_size
        self.t0 = t0
        self.refinement = refinement
        self.guided_radius = guided_radius
        self.guided_eps = guided_eps
        self.matting_lambda = matting_lambda
        self.matting_max_pixels = matting_max_pixels

    def _dark_channel(self, img: np.ndarray) -> np.ndarray:
        min_rgb = img.min(axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                           (self.win_size, self.win_size))
        return cv2.erode(min_rgb, kernel)

    def _atmospheric_light(self, img: np.ndarray,
                           dark: np.ndarray) -> np.ndarray:
        # Paper section 4.4: brightest original-image pixel among the top
        # 0.1% of pixels in the dark channel.
        top_n = max(int(dark.size * 0.001), 1)
        indices = np.argpartition(dark.ravel(), -top_n)[-top_n:]
        flat_img = img.reshape(-1, 3)
        brightest = indices[np.argmax(flat_img[indices].sum(axis=1))]
        return flat_img[brightest].astype(np.float32, copy=True)

    def _transmission(self, img: np.ndarray, A: np.ndarray) -> np.ndarray:
        # Prevent zero atmospheric-light channels from producing NaN/Inf.
        safe_A = np.maximum(A, 1.0 / 255.0)
        norm = img.astype(np.float32) / safe_A.reshape(1, 1, 3)
        return 1.0 - self.omega * self._dark_channel(norm)

    def _guided_refine(self, img: np.ndarray,
                       coarse: np.ndarray) -> np.ndarray:
        """Fast grayscale guided filter; an edge-aware soft-matting substitute."""
        guide = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        k = (2 * self.guided_radius + 1,) * 2

        def mean(x):
            return cv2.boxFilter(x, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT)

        mean_i, mean_p = mean(guide), mean(coarse)
        var_i = mean(guide * guide) - mean_i * mean_i
        cov_ip = mean(guide * coarse) - mean_i * mean_p
        a = cov_ip / (np.maximum(var_i, 0.0) + self.guided_eps)
        b = mean_p - a * mean_i
        return mean(a) * guide + mean(b)

    def _soft_matting_refine(self, img: np.ndarray,
                             coarse: np.ndarray) -> np.ndarray:
        """Paper section 4.2: solve (L + lambda I)t = lambda*t_coarse."""
        try:
            from scipy import sparse
            from scipy.sparse.linalg import cg
        except ImportError as exc:
            raise ImportError("soft_matting requires scipy (pip install scipy)") from exc

        h, w = coarse.shape
        n = h * w
        if n > self.matting_max_pixels:
            raise ValueError(
                f"soft_matting supports at most {self.matting_max_pixels} pixels; "
                "resize the image or use refinement='guided'"
            )
        if h < 3 or w < 3:
            return coarse

        # Levin et al.'s closed-form matting Laplacian, 3x3 RGB windows.
        ids = np.arange(n, dtype=np.int32).reshape(h, w)
        rows, cols, values = [], [], []
        eye = np.eye(9, dtype=np.float64)
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                window_ids = ids[y - 1:y + 2, x - 1:x + 2].ravel()
                rgb = img[y - 1:y + 2, x - 1:x + 2].reshape(9, 3).astype(np.float64)
                centered = rgb - rgb.mean(axis=0)
                covariance = centered.T @ centered / 9.0
                covariance += (1e-7 / 9.0) * np.eye(3)
                affinity = (1.0 + centered @ np.linalg.solve(
                    covariance, centered.T)) / 9.0
                local = eye - affinity
                rows.append(np.repeat(window_ids, 9))
                cols.append(np.tile(window_ids, 9))
                values.append(local.ravel())

        laplacian = sparse.coo_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n, n),
        ).tocsr()
        system = laplacian + self.matting_lambda * sparse.eye(n, format="csr")
        rhs = self.matting_lambda * coarse.astype(np.float64).ravel()
        preconditioner = sparse.diags(1.0 / system.diagonal())
        refined, info = cg(system, rhs, x0=coarse.ravel().astype(np.float64),
                           M=preconditioner, maxiter=500)
        if info != 0:
            raise RuntimeError(f"soft matting conjugate-gradient solve failed (info={info})")
        return refined.reshape(h, w).astype(np.float32)

    def dehaze(self, img: np.ndarray) -> np.ndarray:
        if not isinstance(img, np.ndarray) or img.dtype != np.uint8 or \
                img.ndim != 3 or img.shape[2] != 3 or img.size == 0:
            raise ValueError("img must be a non-empty HxWx3 uint8 RGB array")

        img_float = img.astype(np.float32) / 255.0
        dark = self._dark_channel(img_float)
        A = self._atmospheric_light(img_float, dark)
        if not np.any(A):
            return img.copy()  # No recoverable radiance in a black image.

        coarse = self._transmission(img_float, A)
        if self.refinement == "guided":
            t = self._guided_refine(img_float, coarse)
        elif self.refinement == "soft_matting":
            t = self._soft_matting_refine(img_float, coarse)
        else:
            t = coarse
        t = np.clip(t, self.t0, 1.0)

        # Paper equation 16; floor the transmission only at recovery.
        radiance = (img_float - A) / t[:, :, None] + A
        return np.round(np.clip(radiance, 0.0, 1.0) * 255.0).astype(np.uint8)
