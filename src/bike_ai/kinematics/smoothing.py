"""Temporal smoothing utilities for keypoint trajectories.

- ``interpolate_gaps``: linearly bridge short missing runs (keeps long gaps,
  e.g. setup segments, as NaN so they are not fabricated).
- ``one_euro``: the 1-Euro filter (Casiez et al. 2012), low-latency jitter
  removal with a velocity-adaptive cutoff.
- ``smooth_series``: convenience wrapper (gap-fill -> one-euro or Savitzky-Golay)
  operating on a 1D signal with NaNs.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def interpolate_gaps(y: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly interpolate NaN runs no longer than ``max_gap`` samples."""
    y = y.astype(float).copy()
    n = len(y)
    valid = np.isfinite(y)
    if valid.sum() < 2:
        return y
    idx = np.arange(n)
    filled = y.copy()
    # candidate fill for every position, then only keep short gaps
    interp = np.interp(idx, idx[valid], y[valid])
    i = 0
    while i < n:
        if not np.isfinite(y[i]):
            j = i
            while j < n and not np.isfinite(y[j]):
                j += 1
            gap_len = j - i
            has_left = i > 0 and np.isfinite(y[i - 1])
            has_right = j < n and np.isfinite(y[j])
            if gap_len <= max_gap and has_left and has_right:
                filled[i:j] = interp[i:j]
            i = j
        else:
            i += 1
    return filled


def _alpha(cutoff: float, freq: float) -> float:
    tau = 1.0 / (2.0 * np.pi * cutoff)
    te = 1.0 / freq
    return 1.0 / (1.0 + tau / te)


def one_euro(
    y: np.ndarray,
    freq: float = 10.0,
    min_cutoff: float = 1.0,
    beta: float = 0.3,
    d_cutoff: float = 1.0,
) -> np.ndarray:
    """1-Euro filter on a 1D signal (NaNs are passed through untouched).

    ``min_cutoff`` sets baseline smoothing; ``beta`` raises the cutoff with
    speed to reduce lag on fast motion (e.g. the bottom of the pedal stroke).
    """
    y = y.astype(float)
    out = np.full_like(y, np.nan)
    x_prev = None
    dx_prev = 0.0
    for i, x in enumerate(y):
        if not np.isfinite(x):
            x_prev = None
            dx_prev = 0.0
            continue
        if x_prev is None:
            out[i] = x
            x_prev = x
            continue
        dx = (x - x_prev) * freq
        a_d = _alpha(d_cutoff, freq)
        dx_hat = a_d * dx + (1 - a_d) * dx_prev
        cutoff = min_cutoff + beta * abs(dx_hat)
        a = _alpha(cutoff, freq)
        x_hat = a * x + (1 - a) * x_prev
        out[i] = x_hat
        x_prev = x_hat
        dx_prev = dx_hat
    return out


def _savgol_nan(y: np.ndarray, window: int, poly: int) -> np.ndarray:
    """Savitzky-Golay applied to each contiguous finite run."""
    y = y.astype(float)
    out = y.copy()
    n = len(y)
    i = 0
    while i < n:
        if np.isfinite(y[i]):
            j = i
            while j < n and np.isfinite(y[j]):
                j += 1
            seg = y[i:j]
            if len(seg) >= window:
                w = window if window % 2 == 1 else window + 1
                out[i:j] = savgol_filter(seg, w, poly)
            i = j
        else:
            i += 1
    return out


def smooth_series(
    y: np.ndarray,
    *,
    method: str = "one_euro",
    freq: float = 10.0,
    max_gap: int = 5,
    min_cutoff: float = 1.0,
    beta: float = 0.3,
    savgol_window: int = 11,
    savgol_poly: int = 2,
) -> np.ndarray:
    """Gap-fill short holes then smooth. ``method`` in {one_euro, savgol}."""
    filled = interpolate_gaps(y, max_gap)
    if method == "savgol":
        return _savgol_nan(filled, savgol_window, savgol_poly)
    return one_euro(filled, freq=freq, min_cutoff=min_cutoff, beta=beta)
