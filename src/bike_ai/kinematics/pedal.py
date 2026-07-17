"""Pedal-cycle analysis from the athlete's smoothed keypoints (Step 1).

Side-view assumption: each foot orbits the bottom bracket on the crank circle,
which projects to an ellipse in the image. We fit that ellipse, convert each
foot position to a crank angle, derive cadence, and enforce the physical
left/right antiphase (~180 deg) to fix Sapiens' occasional L/R swaps.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..tracking.keypoints import (
    LEFT_ANKLE,
    LEFT_BIG_TOE,
    RIGHT_ANKLE,
    RIGHT_BIG_TOE,
)
from .smoothing import smooth_series


@dataclass
class PedalCircle:
    """Fitted pedal ellipse (projection of the crank circle)."""

    cx: float
    cy: float
    a: float  # semi-axis 1 (px)
    b: float  # semi-axis 2 (px)
    theta: float  # rotation of axis a (rad)
    rmse_norm: float  # fit residual / mean radius
    n_points: int

    def phase(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        dx, dy = x - self.cx, y - self.cy
        c, s = np.cos(-self.theta), np.sin(-self.theta)
        xr = c * dx - s * dy
        yr = s * dx + c * dy
        return np.arctan2(yr / self.b, xr / self.a)


@dataclass
class PedalAnalysis:
    num_frames: int
    fps: float
    circle: PedalCircle
    phase_left: np.ndarray = field(repr=False)
    phase_right: np.ndarray = field(repr=False)
    cadence_rpm: np.ndarray = field(repr=False)
    phase_diff_deg: np.ndarray = field(repr=False)
    swapped: np.ndarray = field(repr=False)
    direction: int = 1
    median_cadence_rpm: float = 0.0
    antiphase_error_deg: float = 0.0
    ankle_sep_px_median: float = 0.0
    lr_separability: float = 0.0  # median ankle sep / ellipse diameter
    foot_xy: dict[str, np.ndarray] = field(default_factory=dict, repr=False)


def fit_ellipse(x: np.ndarray, y: np.ndarray) -> tuple[float, ...]:
    """Halir-Flusser (1998) numerically-stable direct ellipse fit.

    Returns conic coefficients (a, b, c, d, e, f) for
    a x^2 + b xy + c y^2 + d x + e y + f = 0.
    """
    x = x.astype(float)
    y = y.astype(float)
    D1 = np.column_stack([x * x, x * y, y * y])
    D2 = np.column_stack([x, y, np.ones_like(x)])
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.solve(S3, S2.T)
    M = S1 + S2 @ T
    C_inv = np.array([[0.0, 0.0, 0.5], [0.0, -1.0, 0.0], [0.5, 0.0, 0.0]])
    M = C_inv @ M
    eigval, eigvec = np.linalg.eig(M)
    cond = 4 * eigvec[0] * eigvec[2] - eigvec[1] ** 2
    a1 = eigvec[:, np.nonzero(cond > 0)[0]]
    if a1.size == 0:
        a1 = eigvec[:, :1]
    a1 = np.real(a1[:, 0])
    a2 = T @ a1
    return tuple(np.concatenate([a1, a2]).tolist())


def conic_to_geometric(coeffs: tuple[float, ...]) -> tuple[float, float, float, float, float]:
    """Conic (a,b,c,d,e,f) -> (cx, cy, semi_a, semi_b, theta)."""
    a, b, c, d, e, f = coeffs
    b2 = b / 2.0
    d2 = d / 2.0
    e2 = e / 2.0
    den = b2 * b2 - a * c
    cx = (c * d2 - b2 * e2) / den
    cy = (a * e2 - b2 * d2) / den
    num = 2 * (a * e2 * e2 + c * d2 * d2 + f * b2 * b2 - 2 * b2 * d2 * e2 - a * c * f)
    root = np.sqrt((a - c) ** 2 + 4 * b2 * b2)
    axis1 = np.sqrt(abs(num / (den * ((a + c) + root))))
    axis2 = np.sqrt(abs(num / (den * ((a + c) - root))))
    if abs(b2) < 1e-12:
        theta = 0.0 if a < c else np.pi / 2
    else:
        theta = 0.5 * np.arctan2(2 * b2, a - c)
    sa, sb = max(axis1, axis2), min(axis1, axis2)
    if axis1 < axis2:
        theta += np.pi / 2
    return float(cx), float(cy), float(sa), float(sb), float(theta)


def fit_pedal_circle(xs: np.ndarray, ys: np.ndarray) -> PedalCircle:
    """Fit the pedal ellipse to a cloud of foot points (NaNs ignored)."""
    m = np.isfinite(xs) & np.isfinite(ys)
    x, y = xs[m], ys[m]
    coeffs = fit_ellipse(x, y)
    cx, cy, sa, sb, theta = conic_to_geometric(coeffs)
    circle = PedalCircle(cx, cy, sa, sb, theta, rmse_norm=0.0, n_points=int(m.sum()))
    ph = circle.phase(x, y)
    r_pred = (sa * sb) / np.sqrt(
        (sb * np.cos(ph)) ** 2 + (sa * np.sin(ph)) ** 2
    )
    r_act = np.hypot(x - cx, y - cy)
    circle.rmse_norm = float(np.sqrt(np.mean((r_act - r_pred) ** 2)) / np.mean(r_act))
    return circle


def _stack_xy(frames: list[dict], idx: int, kpt_thr: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(frames)
    xs = np.full(n, np.nan)
    ys = np.full(n, np.nan)
    for i, f in enumerate(frames):
        if not f["instances"]:
            continue
        ins = f["instances"][0]
        if ins["keypoint_scores"][idx] > kpt_thr:
            xs[i] = ins["keypoints"][idx][0]
            ys[i] = ins["keypoints"][idx][1]
    return xs, ys


def _resolve_lr_swaps(
    phL: np.ndarray, phR: np.ndarray, direction: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedily choose identity vs L/R swap per frame to keep each leg's phase
    advancing smoothly (constant sign ``direction``). Returns corrected
    (phL, phR, swapped_flags)."""
    n = len(phL)
    outL = phL.copy()
    outR = phR.copy()
    swapped = np.zeros(n, dtype=bool)
    step = direction * 2 * np.pi / 20.0  # ~expected per-frame advance guess
    predL = predR = None
    for i in range(n):
        l, r = phL[i], phR[i]
        if not (np.isfinite(l) and np.isfinite(r)):
            if np.isfinite(l):
                predL = l
            if np.isfinite(r):
                predR = r
            continue
        if predL is None or predR is None:
            predL, predR = l, r
            continue

        def ang_err(a, b):
            return abs(np.angle(np.exp(1j * (a - b))))

        cost_id = ang_err(l, predL + step) + ang_err(r, predR + step)
        cost_sw = ang_err(r, predL + step) + ang_err(l, predR + step)
        if cost_sw < cost_id:
            outL[i], outR[i] = r, l
            swapped[i] = True
        predL, predR = outL[i], outR[i]
    return outL, outR, swapped


def analyze_pedaling(
    data: dict,
    *,
    fps: float = 10.0,
    kpt_thr: float = 0.3,
    smooth_method: str = "one_euro",
    use_toe: bool = False,
) -> PedalAnalysis:
    """Full Step-1 pedal analysis on an athlete predictions dict."""
    frames = data["frames"]
    n = len(frames)
    la = LEFT_BIG_TOE if use_toe else LEFT_ANKLE
    ra = RIGHT_BIG_TOE if use_toe else RIGHT_ANKLE

    lx, ly = _stack_xy(frames, la, kpt_thr)
    rx, ry = _stack_xy(frames, ra, kpt_thr)
    sm = dict(method=smooth_method, freq=fps, max_gap=5)
    lxs, lys = smooth_series(lx, **sm), smooth_series(ly, **sm)
    rxs, rys = smooth_series(rx, **sm), smooth_series(ry, **sm)

    all_x = np.concatenate([lxs, rxs])
    all_y = np.concatenate([lys, rys])
    circle = fit_pedal_circle(all_x, all_y)

    phL = circle.phase(lxs, lys)
    phR = circle.phase(rxs, rys)

    both = np.isfinite(phL) & np.isfinite(phR)
    ref = phL if np.isfinite(phL).sum() >= np.isfinite(phR).sum() else phR
    dphi = np.diff(np.unwrap(ref[np.isfinite(ref)]))
    direction = 1 if np.median(dphi) >= 0 else -1

    phL, phR, swapped = _resolve_lr_swaps(phL, phR, direction)

    def cadence_from(phase: np.ndarray) -> np.ndarray:
        out = np.full(n, np.nan)
        valid = np.isfinite(phase)
        if valid.sum() < 3:
            return out
        idx = np.where(valid)[0]
        unw = np.unwrap(phase[idx])
        v = np.gradient(unw, idx) * fps  # rad/s
        out[idx] = np.abs(v) / (2 * np.pi) * 60.0
        return out

    cadL = cadence_from(phL)
    cadR = cadence_from(phR)
    stack = np.vstack([cadL, cadR])
    empty = np.all(~np.isfinite(stack), axis=0)
    cadence = np.full(n, np.nan)
    cadence[~empty] = np.nanmean(stack[:, ~empty], axis=0)

    ankle_sep = np.hypot(lxs - rxs, lys - rys)
    sep_med = float(np.nanmedian(ankle_sep))
    lr_sep = sep_med / (2.0 * max(circle.a, circle.b))

    diff = np.full(n, np.nan)
    diff[both] = np.degrees(
        np.angle(np.exp(1j * (phL[both] - phR[both])))
    )
    antiphase_err = float(np.nanmedian(np.abs(np.abs(diff) - 180.0)))

    return PedalAnalysis(
        num_frames=n,
        fps=fps,
        circle=circle,
        phase_left=phL,
        phase_right=phR,
        cadence_rpm=cadence,
        phase_diff_deg=diff,
        swapped=swapped,
        direction=direction,
        median_cadence_rpm=float(np.nanmedian(cadence)),
        antiphase_error_deg=antiphase_err,
        ankle_sep_px_median=sep_med,
        lr_separability=lr_sep,
        foot_xy={"lx": lxs, "ly": lys, "rx": rxs, "ry": rys},
    )
