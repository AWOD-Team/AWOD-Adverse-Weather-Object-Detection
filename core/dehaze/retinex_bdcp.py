"""Fixed-parameter Retinex / bright-dark-channel dehazing for RGB uint8.

Combines He et al. (2009), Jobson et al. (1997), and Liu et al. (2026).
This is an engineering combination, not an exact reproduction of Liu's
nonconvex L2-Lp solver. See docs/retinex_bdcp.md for equations and deviations.
"""

import cv2
import numpy as np

from .dcp import DarkChannelPrior
from .retinex import _fast_gaussian_blur


def _sobel(image):
    """Normalized periodic Sobel derivatives; the adjoints are -Sx and -Sy."""
    dx = (np.roll(image, -1, axis=1) - np.roll(image, 1, axis=1)) / 2
    dy = (np.roll(image, -1, axis=0) - np.roll(image, 1, axis=0)) / 2
    gx = (np.roll(dx, -1, axis=0) + 2 * dx + np.roll(dx, 1, axis=0)) / 4
    gy = (np.roll(dy, -1, axis=1) + 2 * dy + np.roll(dy, 1, axis=1)) / 4
    return gx, gy


def _adjoint(gx, gy):
    return -_sobel(gx)[0] - _sobel(gy)[1]


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
    """Geometry-preserving dehazer; requires only NumPy and OpenCV.

    local_weight=0 uses DCP's global atmospheric light; 1 uses the Retinex
    illumination map. The default mix is an untrained engineering parameter.
    All defaults need validation on the existing detection validation split.
    """

    def __init__(self, omega=0.95, win_size=15, t0=0.1,
                 scales=(15.0, 80.0, 200.0), local_weight=0.5,
                 light_max_side=256, light_l2=0.2, light_l1=0.02,
                 light_iterations=80, light_tolerance=1e-4,
                 bright_gamma=2.0, guided_radius=20, guided_eps=1e-3):
        numbers = (omega, t0, local_weight, light_l2, light_l1,
                   light_tolerance, bright_gamma, guided_eps)
        if not all(np.isfinite(v) for v in numbers):
            raise ValueError("parameters must be finite")
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
        return {"algorithm": "retinex_bdcp_v1", "omega": self.dcp.omega,
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
        dark = 1 - self.dcp._dark_channel(image / np.maximum(light, eps))
        ratio = (image - light) / np.maximum(1 - light, eps)
        kernel = np.ones((self.dcp.win_size, self.dcp.win_size), dtype=np.uint8)
        bright = cv2.dilate(ratio.max(axis=2), kernel)
        # Each prior is only a local hypothesis; invalid negative estimates
        # must not cancel a valid estimate from the other branch.
        dark, bright = np.clip(dark, 0, 1), np.clip(bright, 0, 1)
        weight = np.max(image, axis=2) ** self.bright_gamma
        fused = (1 - weight) * dark + weight * bright
        # He et al.'s residual-haze factor, applied consistently to both priors.
        coarse = 1 - self.dcp.omega + self.dcp.omega * fused
        return dark, bright, weight, coarse

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
        local, solver = self._illumination(image)
        atmosphere = self.dcp._atmospheric_light(image, self.dcp._dark_channel(image))
        light = np.clip((1 - self.local_weight) * atmosphere + self.local_weight * local, 0, 1)
        dark, bright, weight, coarse = self._transmissions(image, light)
        refined = self.dcp._guided_refine(image, coarse)
        bound = self._radiance_bound(image, light)
        # Retain omega's small residual-haze margin above the physical bound.
        safe_bound = 1 - self.dcp.omega + self.dcp.omega * bound
        transmission = np.clip(np.maximum(refined, safe_bound), self.dcp.t0, 1)
        radiance = self._recover(image, light, transmission)
        output = np.rint(np.clip(radiance, 0, 1) * 255).astype(np.uint8)
        details = {"local_illumination": local, "atmospheric_light": atmosphere,
                   "illumination": light, "t_dark": dark, "t_bright": bright,
                   "bright_weight": weight, "t_coarse": coarse, "t_bound": bound,
                   "transmission": transmission, "solver": solver}
        return output, details

    def dehaze(self, img):
        return self.dehaze_with_details(img)[0]
