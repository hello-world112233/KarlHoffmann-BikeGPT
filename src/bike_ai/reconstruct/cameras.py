"""Camera helpers and multi-view triangulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Camera:
    """Pinhole camera: x = K [R|t] X_hom."""

    name: str
    K: np.ndarray  # (3,3)
    R: np.ndarray  # (3,3) world-to-camera
    t: np.ndarray  # (3,)

    @property
    def P(self) -> np.ndarray:
        Rt = np.concatenate([self.R, self.t.reshape(3, 1)], axis=1)
        return self.K @ Rt

    def project(self, xyz: np.ndarray) -> np.ndarray:
        """xyz (..., 3) → (..., 2) pixels."""
        X = np.asarray(xyz, dtype=np.float64)
        flat = X.reshape(-1, 3)
        ones = np.ones((flat.shape[0], 1))
        hom = np.concatenate([flat, ones], axis=1)  # (N,4)
        p = (self.P @ hom.T).T
        p = p[:, :2] / np.clip(p[:, 2:3], 1e-8, None)
        return p.reshape(X.shape[:-1] + (2,))


def triangulate_point(cameras: list[Camera], observations: list[np.ndarray]) -> np.ndarray:
    """DLT triangulation for one point seen in >=2 views. observations are (2,) each."""
    if len(cameras) < 2:
        raise ValueError("need >=2 cameras")
    A = []
    for cam, uv in zip(cameras, observations):
        P = cam.P
        u, v = float(uv[0]), float(uv[1])
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    A = np.asarray(A, dtype=np.float64)
    _, _, vt = np.linalg.svd(A)
    X = vt[-1]
    X = X[:3] / (X[3] + 1e-12)
    return X


def triangulate_joints(
    cameras: list[Camera],
    joints_2d: list[np.ndarray],
    scores: list[np.ndarray] | None = None,
    score_thr: float = 0.3,
) -> np.ndarray:
    """Triangulate COCO-17 joints.

    joints_2d: list of (17,2) per camera.
    Returns (17,3); joints with <2 good views become NaN.
    """
    k = joints_2d[0].shape[0]
    out = np.full((k, 3), np.nan, dtype=np.float64)
    for j in range(k):
        cams = []
        obs = []
        for ci, cam in enumerate(cameras):
            uv = joints_2d[ci][j]
            if not np.isfinite(uv).all():
                continue
            if scores is not None and scores[ci][j] < score_thr:
                continue
            cams.append(cam)
            obs.append(uv)
        if len(cams) >= 2:
            out[j] = triangulate_point(cams, obs)
    return out


def camera_from_dict(d: dict) -> Camera:
    return Camera(
        name=str(d["name"]),
        K=np.asarray(d["K"], dtype=np.float64),
        R=np.asarray(d["R"], dtype=np.float64),
        t=np.asarray(d["t"], dtype=np.float64),
    )
