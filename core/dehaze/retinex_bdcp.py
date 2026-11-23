"""RGB dehazing: RetinexBDCP, dark channel prior, and legacy Retinex fusion."""


from concurrent.futures import ThreadPoolExecutor

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
                           dark: np.ndarray, *, reject_local_highlights=False) -> np.ndarray:
        # Paper section 4.4: brightest original-image pixel among the top
        # 0.1% of pixels in the dark channel.
        top_n = max(int(dark.size * 0.001), 1)
        indices = np.argpartition(dark.ravel(), -top_n)[-top_n:]
        flat_img = img.reshape(-1, 3)
        brightest = indices[np.argmax(flat_img[indices].sum(axis=1))]
        light = flat_img[brightest].astype(np.float32, copy=True)
        if not reject_local_highlights or light.max() < 0.97:
            return light

        # Distinguish an almost clipped local light from a broadly bright sky.
        window = max(self.win_size, round(min(dark.shape) * 0.1)) | 1
        peak = img.max(axis=2)
        surround = cv2.blur(peak, (window, window))
        highlights = (peak >= 0.97) & (peak - surround > 0.10)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window, window))
        excluded = cv2.dilate(highlights.astype(np.uint8), kernel)
        if not excluded.flat[brightest] or excluded.all():
            return light

        # Prefer extended haze patches outside light cores and their halos.
        eligible = excluded == 0
        broad_dark = cv2.erode(dark, kernel)
        cutoff = np.percentile(broad_dark[eligible], 99.5)
        return np.median(img[eligible & (broad_dark >= cutoff)], axis=0)

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


def _sobel(image):
    """Normalized periodic Sobel derivatives; the adjoints are -Sx and -Sy."""
    dx = (np.roll(image, -1, axis=1) - np.roll(image, 1, axis=1)) / 2
    dy = (np.roll(image, -1, axis=0) - np.roll(image, 1, axis=0)) / 2
    gx = (np.roll(dx, -1, axis=0) + 2 * dx + np.roll(dx, 1, axis=0)) / 4
    gy = (np.roll(dy, -1, axis=1) + 2 * dy + np.roll(dy, 1, axis=1)) / 4
    return gx, gy


def _adjoint(gx, gy):
    return -_sobel(gx)[0] - _sobel(gy)[1]


def _srgb_to_linear(values):
    return np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(values):
    return np.where(values <= 0.0031308, 12.92 * values, 1.055 * values ** (1 / 2.4) - 0.055)


def _smooth_illumination(initial, l2, l1, iterations, tolerance):
    """ADMM for .5||L-L0||² + l2*sum||W*SjL||² + l1*sum||W*SjL||1.

    p=1 makes the soft-threshold update exact. Reflect padding reduces the
    periodic FFT boundary effect. A bounded iteration count is deliberate;
    convergence diagnostics are returned rather than silently claimed.
    """
    pad = 8
    reference = np.pad(initial.astype(np.float64),
                       ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    light = reference.copy()
    gx, gy = _sobel(reference)
    # Paper does not specify W numerically. Share edge weights across RGB.
    wx = 1 / (1 + 10 * np.max(np.abs(gx), axis=2, keepdims=True))
    wy = 1 / (1 + 10 * np.max(np.abs(gy), axis=2, keepdims=True))
    vx, vy, ux, uy = gx.copy(), gy.copy(), gx.copy(), gy.copy()
    yx, yy, zx, zy = (np.zeros_like(reference) for _ in range(4))
    h, w = reference.shape[:2]
    fy = 2 * np.pi * np.fft.fftfreq(h)[:, None]
    fx = 2 * np.pi * np.fft.rfftfreq(w)[None, :]
    sx2 = (np.sin(fx) * (1 + np.cos(fy)) / 2) ** 2
    sy2 = (np.sin(fy) * (1 + np.cos(fx)) / 2) ** 2
    # Both constraints have ADMM penalty mu=1, data fidelity lambda=1.
    denominator = (1 + 2 * (sx2 + sy2))[..., None]

    def rms(value):
        return float(np.sqrt(np.mean(value * value)))

    converged = False
    for step in range(1, iterations + 1):
        rhs = reference + _adjoint(vx + yx + ux + zx, vy + yy + uy + zy)
        light = np.fft.irfft2(np.fft.rfft2(rhs, axes=(0, 1)) / denominator,
                             s=(h, w), axes=(0, 1))
        gx, gy = _sobel(light)
        old_x, old_y = vx + ux, vy + uy
        vx, vy = (gx - yx) / (1 + 2 * l2 * wx ** 2), (gy - yy) / (1 + 2 * l2 * wy ** 2)
        qx, qy = gx - zx, gy - zy
        ux = np.sign(qx) * np.maximum(np.abs(qx) - l1 * wx, 0)
        uy = np.sign(qy) * np.maximum(np.abs(qy) - l1 * wy, 0)
        yx, yy = yx + vx - gx, yy + vy - gy
        zx, zy = zx + ux - gx, zy + uy - gy
        primal = max(rms(vx - gx), rms(vy - gy), rms(ux - gx), rms(uy - gy))
        dual = rms(_adjoint(vx + ux - old_x, vy + uy - old_y))
        if max(primal, dual) <= tolerance:
            converged = True
            break
    info = {"iterations": step, "converged": converged,
            "primal_rms": primal, "dual_rms": dual}
    return np.clip(light[pad:-pad, pad:-pad], 0, 1).astype(np.float32), info


class RetinexBDCP:
    """Balanced DCP-based dehazer with noise suppression and edge preservation.

    Key design choices:
    - Stays in sRGB space (no gamma conversion) to preserve natural colors
    - Moderate t0=0.3 balances dehazing vs. noise amplification in fog regions
    - Large guided filter radius (40) for strong edge-aware smoothing
    - Light regularization suppresses blocky DCP artifacts without over-blurring

    Default mode (local_weight=0) uses pure DCP with global atmospheric light.
    Set local_weight > 0 to blend with Retinex local illumination.
    """

    def __init__(self, omega=0.95, win_size=15, t0=0.3,
                 scales=(15.0, 80.0, 200.0), local_weight=0.0,
                 light_max_side=256, light_l2=0.1, light_l1=0.01,
                 light_iterations=40, light_tolerance=1e-4,
                 bright_gamma=2.0, guided_radius=40, guided_eps=1e-2, *,
                 mode="standard"):
        if mode not in ("standard", "daylight"):
            raise ValueError("mode must be standard or daylight")
        self.mode = mode
        # Simplified defaults: closer to pure DCP behavior
        if local_weight is None:
            local_weight = 0.0  # Default to global atmospheric light (pure DCP)
        if t0 is None:
            t0 = 0.1
        numbers = (omega, t0, local_weight, light_l2, light_l1,
                   light_tolerance, bright_gamma, guided_eps)
        if not all(isinstance(v, (int, float, np.number)) and np.isfinite(v) for v in numbers):
            raise ValueError("parameters must be finite numbers")
        for name, value in (("win_size", win_size), ("guided_radius", guided_radius),
                            ("light_max_side", light_max_side),
                            ("light_iterations", light_iterations)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        scales = tuple(float(s) for s in scales)
        if not scales or not all(np.isfinite(s) and s > 0 for s in scales):
            raise ValueError("scales must contain finite positive Gaussian sigmas")
        if not 0 <= local_weight <= 1:
            raise ValueError("local_weight must be in [0, 1]")
        if min(light_l2, light_l1) < 0 or min(light_tolerance, bright_gamma) <= 0:
            raise ValueError("regularization must be nonnegative; tolerance and gamma positive")
        self.dcp = DarkChannelPrior(omega=omega, win_size=win_size, t0=t0,
                                    guided_radius=guided_radius, guided_eps=guided_eps)
        self.scales = scales
        self.local_weight = float(local_weight)
        self.light_max_side = int(light_max_side)
        self.light_l2, self.light_l1 = float(light_l2), float(light_l1)
        self.light_iterations, self.light_tolerance = int(light_iterations), float(light_tolerance)
        self.bright_gamma = float(bright_gamma)

    def parameters(self):
        """JSON-serializable configuration for reproducible preprocessing."""
        algorithm = "balanced_dcp_v3" if self.local_weight == 0 else "dcp_retinex_balanced_v3"
        return {"algorithm": algorithm, "reject_local_highlights": True, "omega": self.dcp.omega,
                "win_size": self.dcp.win_size, "t0": self.dcp.t0,
                "scales": list(self.scales), "local_weight": self.local_weight,
                "light_max_side": self.light_max_side, "light_l2": self.light_l2,
                "light_l1": self.light_l1, "light_iterations": self.light_iterations,
                "light_tolerance": self.light_tolerance, "bright_gamma": self.bright_gamma,
                "guided_radius": self.dcp.guided_radius, "guided_eps": self.dcp.guided_eps}

    def _illumination(self, image):
        h, w = image.shape[:2]
        scale = min(1.0, self.light_max_side / max(h, w))
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        small = cv2.resize(image, size, interpolation=cv2.INTER_AREA) if scale < 1 else image
        eps = 1.0 / 255.0
        log_surround = np.zeros_like(small)
        for sigma in self.scales:
            surround = _fast_gaussian_blur(small + eps, sigma * scale)
            log_surround += np.log(np.maximum(surround, eps)) / len(self.scales)
        # MSR = log(I+eps) - mean(log(G_sigma*(I+eps))).
        initial = np.clip(np.exp(log_surround) - eps, 0, 1)
        light, info = _smooth_illumination(initial, self.light_l2, self.light_l1,
                                            self.light_iterations, self.light_tolerance)
        if scale < 1:
            light = cv2.resize(light, (w, h), interpolation=cv2.INTER_LINEAR)
        info["working_size"] = list(size)
        return light, info

    def _transmissions(self, image, light):
        eps = 1.0 / 255.0
        dark = np.clip(1 - self.dcp._dark_channel(image / np.maximum(light, eps)), 0, 1)

        # For standard mode (most fog cases), skip bright channel computation
        # to avoid introducing extra complexity and noise
        if self.mode == "daylight":
            # Bright fog and sky are not reliable cues for choosing the bright prior.
            zero = np.zeros_like(dark)
            return dark, zero, zero, 1 - self.dcp.omega + self.dcp.omega * dark

        # Simplified: only use dark channel for fog (pure DCP approach)
        zero = np.zeros_like(dark)
        coarse = 1 - self.dcp.omega + self.dcp.omega * dark
        return dark, zero, zero, coarse

    @staticmethod
    def _radiance_bound(image, light):
        """Smallest t for which 0 <= (I-L)/t+L <= 1 in every RGB channel.

        This follows directly from the scattering equation, not from a new
        statistical prior. Exact zero denominators impose no constraint on
        their respective side of the range.
        """
        lower = np.divide(light - image, light, out=np.zeros_like(image), where=light > 0)
        upper = np.divide(image - light, 1 - light, out=np.zeros_like(image), where=light < 1)
        return np.clip(np.maximum(lower, upper).max(axis=2), 0, 1)

    @staticmethod
    def _recover(image, light, transmission):
        # The SAME bounded transmission is used in both terms (DCP eq. 16).
        return (image - light) / transmission[..., None] + light

    def dehaze_with_details(self, img):
        """Return (RGB uint8 image, maps/solver diagnostics); no input mutation."""
        if not isinstance(img, np.ndarray) or img.dtype != np.uint8 or \
                img.ndim != 3 or img.shape[2] != 3 or img.size == 0:
            raise ValueError("img must be a non-empty HxWx3 uint8 RGB array")
        image = img.astype(np.float32) / 255.0

        # When local_weight=0, skip illumination solver entirely (pure DCP mode)
        if self.local_weight > 0:
            local, solver = self._illumination(image)
        else:
            local = np.zeros_like(image)
            solver = {"iterations": 0, "converged": True, "working_size": [0, 0]}

        atmosphere = self.dcp._atmospheric_light(
            image, self.dcp._dark_channel(image), reject_local_highlights=True)
        light = np.clip((1 - self.local_weight) * atmosphere + self.local_weight * local, 0, 1)

        # Remove sRGB ↔ linear conversion to avoid color distortion
        # Keep everything in sRGB space (standard DCP behavior)

        dark, bright, weight, coarse = self._transmissions(image, light)
        refined = self.dcp._guided_refine(image, coarse)
        bound = self._radiance_bound(image, light)
        # Retain omega's small residual-haze margin above the physical bound.
        safe_bound = 1 - self.dcp.omega + self.dcp.omega * bound
        transmission = np.clip(np.maximum(refined, safe_bound), self.dcp.t0, 1)
        radiance = np.clip(self._recover(image, light, transmission), 0, 1)

        output = np.rint(radiance * 255).astype(np.uint8)
        details = {"local_illumination": local, "atmospheric_light": atmosphere,
                   "illumination": light, "t_dark": dark, "t_bright": bright,
                   "bright_weight": weight, "t_coarse": coarse, "t_bound": bound,
                   "transmission": transmission, "solver": solver}
        return output, details

    def dehaze(self, img):
        return self.dehaze_with_details(img)[0]


class HybridDehazer:
    def __init__(self, omega: float = 0.95, win_size: int = 15, t0: float = 0.1,
                 retinex_scales: list = None, alpha: float = 125.0, beta: float = 46.0,
                 fusion_weight: float = 0.5, num_workers: int = 2,
                 dcp_refinement: str = "guided"):
        self.dcp = DarkChannelPrior(omega=omega, win_size=win_size, t0=t0,
                                    refinement=dcp_refinement)
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
