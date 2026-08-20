"""Known bicycle rigid geometry used as scale + contact constraints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BicycleGeometry:
    """Units: meters. Defaults are typical track-bike ballparks."""

    crank_length_m: float = 0.170
    # Bottom bracket in a bike/world frame; fitted or measured.
    bb_xyz: np.ndarray | None = None

    def ankle_circle_residuals(
        self,
        left_ankle: np.ndarray,
        right_ankle: np.ndarray,
        bb: np.ndarray,
        plane_normal: np.ndarray | None = None,
    ) -> np.ndarray:
        """Both ankles should lie on a sphere/circle of radius crank_length around BB.

        If plane_normal is given (unit), also penalize out-of-plane deviation
        (crank mostly rotates in the sagittal plane of the bike).
        """
        r = self.crank_length_m
        dL = np.linalg.norm(left_ankle - bb) - r
        dR = np.linalg.norm(right_ankle - bb) - r
        out = [dL, dR]
        if plane_normal is not None:
            n = plane_normal / (np.linalg.norm(plane_normal) + 1e-9)
            out.append(float(np.dot(left_ankle - bb, n)))
            out.append(float(np.dot(right_ankle - bb, n)))
        return np.asarray(out, dtype=np.float64)
