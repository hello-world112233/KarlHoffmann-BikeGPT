"""Multi-view skeleton fitting: CAD bones + optional bicycle constraints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from bike_ai.reconstruct.bicycle import BicycleGeometry
from bike_ai.reconstruct.cameras import Camera, triangulate_joints
from bike_ai.reconstruct.skeleton import SkeletonModel


@dataclass
class FitResult:
    joints_xyz: np.ndarray  # (17, 3)
    reprojection_rmse_px: float
    bone_rmse_m: float
    success: bool
    message: str

    def to_dict(self) -> dict:
        return {
            "joints_xyz": self.joints_xyz.tolist(),
            "reprojection_rmse_px": self.reprojection_rmse_px,
            "bone_rmse_m": self.bone_rmse_m,
            "success": self.success,
            "message": self.message,
        }


def _pack(joints: np.ndarray, bb: np.ndarray | None) -> np.ndarray:
    if bb is None:
        return joints.reshape(-1)
    return np.concatenate([joints.reshape(-1), bb.reshape(-1)])


def _unpack(x: np.ndarray, with_bb: bool) -> tuple[np.ndarray, np.ndarray | None]:
    joints = x[: 17 * 3].reshape(17, 3)
    bb = x[17 * 3 : 17 * 3 + 3] if with_bb else None
    return joints, bb


def fit_multiview_skeleton(
    cameras: list[Camera],
    joints_2d: list[np.ndarray],
    scores: list[np.ndarray] | None = None,
    skeleton: SkeletonModel | None = None,
    bicycle: BicycleGeometry | None = None,
    bone_weight: float = 50.0,
    bike_weight: float = 30.0,
    score_thr: float = 0.3,
) -> FitResult:
    """Fit one frame of 3D COCO-17 joints to multi-view 2D observations.

    Pipeline:
    1. DLT triangulation init (needs camera matrices)
    2. Nonlinear refine: reprojection + bone lengths + optional crank constraints
    """
    skeleton = skeleton or SkeletonModel.from_height_m(1.75)
    init = triangulate_joints(cameras, joints_2d, scores=scores, score_thr=score_thr)

    # Fill NaNs with hip midpoint / reasonable defaults
    valid = np.isfinite(init).all(axis=1)
    if valid.sum() < 3:
        return FitResult(
            joints_xyz=init,
            reprojection_rmse_px=float("inf"),
            bone_rmse_m=float("inf"),
            success=False,
            message="Too few triangulated joints; check calibration / sync / athlete selection.",
        )
    center = np.nanmean(init[valid], axis=0)
    for j in range(17):
        if not np.isfinite(init[j]).all():
            init[j] = center

    use_bb = bicycle is not None
    if use_bb:
        # BB init: midpoint of ankles, then move toward hips
        ankles = 0.5 * (init[15] + init[16])
        hips = 0.5 * (init[11] + init[12])
        bb0 = ankles + 0.15 * (hips - ankles)
    else:
        bb0 = None

    x0 = _pack(init, bb0)

    def residuals(x: np.ndarray) -> np.ndarray:
        joints, bb = _unpack(x, use_bb)
        res = []
        # Reprojection
        for ci, cam in enumerate(cameras):
            proj = cam.project(joints)
            obs = joints_2d[ci]
            w = np.ones(17)
            if scores is not None:
                w = np.asarray(scores[ci], dtype=np.float64)
                w = np.where(w >= score_thr, w, 0.0)
            diff = (proj - obs) * w[:, None]
            # Ignore non-finite obs
            mask = np.isfinite(obs).all(axis=1)
            diff = diff[mask]
            res.append(diff.reshape(-1))
        # Bone lengths
        bone_r = skeleton.bone_length_residuals(joints) * bone_weight
        res.append(bone_r)
        # Bicycle
        if use_bb and bb is not None:
            bike_r = bicycle.ankle_circle_residuals(joints[15], joints[16], bb) * bike_weight
            res.append(bike_r)
        return np.concatenate(res)

    out = least_squares(residuals, x0, method="lm", max_nfev=200)
    joints, bb = _unpack(out.x, use_bb)

    # Metrics
    errs = []
    for ci, cam in enumerate(cameras):
        proj = cam.project(joints)
        obs = joints_2d[ci]
        mask = np.isfinite(obs).all(axis=1)
        if scores is not None:
            mask &= np.asarray(scores[ci]) >= score_thr
        if mask.any():
            errs.append(np.linalg.norm(proj[mask] - obs[mask], axis=1))
    reproj = float(np.sqrt(np.mean(np.concatenate(errs) ** 2))) if errs else float("inf")
    bone_rmse = float(np.sqrt(np.mean(skeleton.bone_length_residuals(joints) ** 2)))

    if use_bb and bb is not None and bicycle is not None:
        bicycle.bb_xyz = bb

    return FitResult(
        joints_xyz=joints,
        reprojection_rmse_px=reproj,
        bone_rmse_m=bone_rmse,
        success=bool(out.success),
        message=str(out.message),
    )


def fit_sequence(
    cameras: list[Camera],
    seq_2d: list[np.ndarray],
    seq_scores: list[np.ndarray] | None = None,
    skeleton: SkeletonModel | None = None,
    bicycle: BicycleGeometry | None = None,
) -> list[FitResult]:
    """Fit each frame independently.

    seq_2d: list over cameras of arrays (T, 17, 2)
    """
    t = seq_2d[0].shape[0]
    results = []
    for ti in range(t):
        joints = [s[ti] for s in seq_2d]
        scores = [s[ti] for s in seq_scores] if seq_scores is not None else None
        results.append(
            fit_multiview_skeleton(
                cameras, joints, scores=scores, skeleton=skeleton, bicycle=bicycle
            )
        )
    return results


def save_fit_sequence(results: list[FitResult], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "num_frames": len(results),
        "frames": [r.to_dict() for r in results],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
