"""Kinematic analysis of the rider under physical constraints (Step 1+).

Currently covers pedaling: temporal smoothing of keypoint trajectories, pedal
circle (ellipse in image) fitting, crank-angle / cadence extraction, and
left/right phase-consistency correction.
"""

from __future__ import annotations

from .pedal import (
    PedalAnalysis,
    PedalCircle,
    analyze_pedaling,
    fit_pedal_circle,
)
from .smoothing import interpolate_gaps, one_euro, smooth_series

__all__ = [
    "PedalAnalysis",
    "PedalCircle",
    "analyze_pedaling",
    "fit_pedal_circle",
    "interpolate_gaps",
    "one_euro",
    "smooth_series",
]
