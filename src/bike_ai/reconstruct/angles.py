"""Biomechanical joint angles from 3D COCO-17 joints."""

from __future__ import annotations

from typing import Any

import numpy as np

from bike_ai.reconstruct.skeleton import angle_3points


def compute_joint_angles(joints_xyz: np.ndarray) -> dict[str, float]:
    """Return named angles in degrees. joints_xyz shape (17, 3)."""
    j = np.asarray(joints_xyz, dtype=np.float64)
    out: dict[str, float] = {}
    # Knees
    out["left_knee_deg"] = angle_3points(j[11], j[13], j[15])
    out["right_knee_deg"] = angle_3points(j[12], j[14], j[16])
    # Hips (shoulder-hip-knee)
    out["left_hip_deg"] = angle_3points(j[5], j[11], j[13])
    out["right_hip_deg"] = angle_3points(j[6], j[12], j[14])
    # Elbows
    out["left_elbow_deg"] = angle_3points(j[5], j[7], j[9])
    out["right_elbow_deg"] = angle_3points(j[6], j[8], j[10])
    # Trunk lean: angle between torso (midhip->midshoulder) and vertical
    mid_sh = 0.5 * (j[5] + j[6])
    mid_hip = 0.5 * (j[11] + j[12])
    torso = mid_sh - mid_hip
    vertical = np.array([0.0, 0.0, 1.0])
    n = np.linalg.norm(torso) + 1e-9
    cos = float(np.clip(np.dot(torso / n, vertical), -1.0, 1.0))
    out["trunk_lean_from_vertical_deg"] = float(np.degrees(np.arccos(cos)))
    return out


def angles_for_sequence(joints_seq: np.ndarray) -> list[dict[str, float]]:
    """joints_seq: (T, 17, 3)"""
    return [compute_joint_angles(joints_seq[t]) for t in range(len(joints_seq))]


def summarize_angles(frames: list[dict[str, float]]) -> dict[str, Any]:
    if not frames:
        return {}
    keys = frames[0].keys()
    summary = {}
    for k in keys:
        vals = np.array([f[k] for f in frames if np.isfinite(f[k])], dtype=np.float64)
        if len(vals) == 0:
            continue
        summary[k] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }
    return summary
