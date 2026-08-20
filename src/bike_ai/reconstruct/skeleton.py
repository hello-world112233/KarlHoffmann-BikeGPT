"""Rigid-ish human skeleton CAD model (COCO-17).

This is the "framework" discussed in planning: bone lengths are parameters
(scale to 160cm / 180cm athletes), joint angles/positions are fitted to video.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bike_ai.tracking.keypoints import COCO17_INDEX, COCO17_NAMES

# Undirected bones for length constraints (parent, child) in COCO-17 indices.
COCO17_BONES: list[tuple[int, int]] = [
    (5, 6),  # shoulders
    (5, 7),
    (7, 9),  # left arm
    (6, 8),
    (8, 10),  # right arm
    (5, 11),
    (6, 12),  # torso sides
    (11, 12),  # hips
    (11, 13),
    (13, 15),  # left leg
    (12, 14),
    (14, 16),  # right leg
    (0, 5),
    (0, 6),  # head-ish to shoulders
]


# Mean bone lengths in meters for a ~1.75m adult (rough anthropometric priors).
DEFAULT_BONE_LENGTHS_M: dict[tuple[int, int], float] = {
    (5, 6): 0.35,
    (5, 7): 0.28,
    (7, 9): 0.25,
    (6, 8): 0.28,
    (8, 10): 0.25,
    (5, 11): 0.48,
    (6, 12): 0.48,
    (11, 12): 0.28,
    (11, 13): 0.42,
    (13, 15): 0.40,
    (12, 14): 0.42,
    (14, 16): 0.40,
    (0, 5): 0.22,
    (0, 6): 0.22,
}


@dataclass
class SkeletonModel:
    """CAD skeleton: named joints + bone length priors."""

    bone_lengths_m: dict[tuple[int, int], float]
    height_scale: float = 1.0  # 1.0 ≈ 1.75m reference

    @classmethod
    def from_height_m(cls, height_m: float = 1.75) -> SkeletonModel:
        scale = height_m / 1.75
        lengths = {b: L * scale for b, L in DEFAULT_BONE_LENGTHS_M.items()}
        return cls(bone_lengths_m=lengths, height_scale=scale)

    def bone_length_residuals(self, joints_xyz: np.ndarray) -> np.ndarray:
        """joints_xyz: (17, 3) → residual per bone (meters)."""
        res = []
        for (i, j), L in self.bone_lengths_m.items():
            d = np.linalg.norm(joints_xyz[i] - joints_xyz[j])
            res.append(d - L)
        return np.asarray(res, dtype=np.float64)


def angle_3points(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Interior angle at b formed by a-b-c, in degrees."""
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1) + 1e-9
    n2 = np.linalg.norm(v2) + 1e-9
    cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def joint_name(i: int) -> str:
    return COCO17_NAMES[i]
