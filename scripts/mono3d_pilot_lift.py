#!/usr/bin/env python3
"""Monocular 2D→3D possibility-test lift (no camera calibration).

Takes Sapiens2 predictions (308-kpt), picks the athlete instance per frame,
lifts COCO-17 body joints with a simple bone-length / foreshortening model,
writes JSON + rotating skeleton MP4.

This is intentionally approximate — for feasibility only, not metric accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bike_ai.tracking.keypoints import (  # noqa: E402
    LEFT_ANKLE,
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NOSE,
    RIGHT_ANKLE,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    SKELETON_LINKS,
)

# Parent→child kinematic chains for depth solve (COCO-17 indices)
CHAINS: list[tuple[int, int, float]] = [
    # (parent, child, prior_length_m)
    (LEFT_HIP, LEFT_SHOULDER, 0.48),
    (RIGHT_HIP, RIGHT_SHOULDER, 0.48),
    (LEFT_SHOULDER, RIGHT_SHOULDER, 0.36),
    (LEFT_HIP, RIGHT_HIP, 0.28),
    (LEFT_SHOULDER, LEFT_ELBOW, 0.28),
    (LEFT_ELBOW, LEFT_WRIST, 0.25),
    (RIGHT_SHOULDER, RIGHT_ELBOW, 0.28),
    (RIGHT_ELBOW, RIGHT_WRIST, 0.25),
    (LEFT_HIP, LEFT_KNEE, 0.42),
    (LEFT_KNEE, LEFT_ANKLE, 0.40),
    (RIGHT_HIP, RIGHT_KNEE, 0.42),
    (RIGHT_KNEE, RIGHT_ANKLE, 0.40),
    (LEFT_SHOULDER, NOSE, 0.30),
    (RIGHT_SHOULDER, NOSE, 0.30),
]

BODY17 = [
    NOSE,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_ELBOW,
    RIGHT_ELBOW,
    LEFT_WRIST,
    RIGHT_WRIST,
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_KNEE,
    RIGHT_KNEE,
    LEFT_ANKLE,
    RIGHT_ANKLE,
]


def pick_athlete(instances: list[dict], image_size: list[int]) -> dict | None:
    """Pick central, fairly large person (rider on rollers)."""
    if not instances:
        return None
    w, h = image_size
    cx0, cy0 = w * 0.5, h * 0.55
    best, best_score = None, -1e9
    for inst in instances:
        bb = inst["bbox"]  # x1,y1,x2,y2 or xywh? check
        if len(bb) == 4:
            x1, y1, x2, y2 = bb
            if x2 < x1:  # xywh
                x2, y2 = x1 + x2, y1 + y2
        else:
            continue
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        area = bw * bh / (w * h)
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        dist = np.hypot((cx - cx0) / w, (cy - cy0) / h)
        # prefer large + central
        score = area * 3.0 - dist * 1.5
        kps = np.asarray(inst["keypoints"], dtype=float)
        sc = np.asarray(inst.get("keypoint_scores") or inst.get("scores") or [1.0] * len(kps))
        if kps.shape[0] >= 17:
            leg = float(np.mean(sc[[LEFT_ANKLE, RIGHT_ANKLE, LEFT_KNEE, RIGHT_KNEE]]))
            score += 0.3 * leg
        if score > best_score:
            best_score, best = score, inst
    return best


def lift_frame(kpts_xy: np.ndarray, scores: np.ndarray, image_size: list[int]) -> np.ndarray:
    """Return (17,3) root-relative joints in meters (approx)."""
    w, h = image_size
    xy = kpts_xy[:17].astype(np.float64).copy()
    sc = scores[:17].astype(np.float64).copy()

    # Mid-hip origin in image
    if sc[LEFT_HIP] > 0.2 and sc[RIGHT_HIP] > 0.2:
        hip = 0.5 * (xy[LEFT_HIP] + xy[RIGHT_HIP])
    else:
        hip = np.nanmean(xy[sc > 0.2], axis=0)

    # Pixel → metric scale from shoulder / hip width
    scales = []
    if sc[LEFT_SHOULDER] > 0.2 and sc[RIGHT_SHOULDER] > 0.2:
        scales.append(0.36 / (np.linalg.norm(xy[LEFT_SHOULDER] - xy[RIGHT_SHOULDER]) + 1e-6))
    if sc[LEFT_HIP] > 0.2 and sc[RIGHT_HIP] > 0.2:
        scales.append(0.28 / (np.linalg.norm(xy[LEFT_HIP] - xy[RIGHT_HIP]) + 1e-6))
    scale = float(np.median(scales)) if scales else 1.7 / h

    # Camera-plane coords (X right, Y up)
    xyz = np.zeros((17, 3), dtype=np.float64)
    xyz[:, 0] = (xy[:, 0] - hip[0]) * scale
    xyz[:, 1] = -(xy[:, 1] - hip[1]) * scale  # flip image y
    xyz[:, 2] = 0.0

    # Foreshortening depth: for each bone, if 2D length < prior, put child behind/in front
    # Prefer cycling-facing: camera side view-ish → knees oscillate in Z
    for parent, child, L in CHAINS:
        if sc[parent] < 0.15 or sc[child] < 0.15:
            continue
        d_xy = np.linalg.norm(xyz[child, :2] - xyz[parent, :2])
        d_xy = min(d_xy, L * 0.999)
        dz = float(np.sqrt(max(L * L - d_xy * d_xy, 0.0)))
        # Sign: ankles/knees go "into" scene relative to hip (positive Z = away from camera)
        # Use image vertical: lower joints slightly toward camera when pedaling toward cam
        # Heuristic: if child is below parent in image, alternate with left/right for pedaling
        sign = 1.0
        if child in (LEFT_KNEE, LEFT_ANKLE, LEFT_WRIST, LEFT_ELBOW):
            sign = 1.0
        elif child in (RIGHT_KNEE, RIGHT_ANKLE, RIGHT_WRIST, RIGHT_ELBOW):
            sign = -1.0
        # Prefer continuity: keep same hemisphere as parent's z
        xyz[child, 2] = xyz[parent, 2] + sign * dz

    # Re-root at mid-hip
    root = 0.5 * (xyz[LEFT_HIP] + xyz[RIGHT_HIP])
    xyz -= root
    # Soft bone-length projection (one pass)
    for parent, child, L in CHAINS:
        if sc[parent] < 0.15 or sc[child] < 0.15:
            continue
        v = xyz[child] - xyz[parent]
        n = np.linalg.norm(v) + 1e-9
        xyz[child] = xyz[parent] + v * (L / n)

    root = 0.5 * (xyz[LEFT_HIP] + xyz[RIGHT_HIP])
    xyz -= root
    # mask low-score joints as nan for viz
    xyz[sc < 0.15] = np.nan
    return xyz


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--fps", type=float, default=10.0)
    args = ap.parse_args()

    data = json.loads(args.predictions.read_text())
    image_size = data["image_size"]  # [w,h] or [h,w]?
    # sapiens usually [w,h] — check
    if isinstance(image_size, dict):
        image_size = [image_size["width"], image_size["height"]]
    # heuristic: first value larger for 4k landscape
    if image_size[0] < image_size[1]:
        image_size = [image_size[1], image_size[0]]

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    sequence = []
    for fi, frame in enumerate(data["frames"]):
        inst = pick_athlete(frame.get("instances") or [], image_size)
        if inst is None:
            sequence.append({"frame": fi, "image_name": frame.get("image_name"), "joints_xyz": None})
            continue
        kps = np.asarray(inst["keypoints"], dtype=float)
        sc = np.asarray(inst.get("keypoint_scores") or [1.0] * len(kps), dtype=float)
        xyz = lift_frame(kps, sc, image_size)
        sequence.append(
            {
                "frame": fi,
                "image_name": frame.get("image_name"),
                "bbox": inst.get("bbox"),
                "joints_xyz": xyz.tolist(),
                "joint_scores": sc[:17].tolist(),
            }
        )

    n_ok = sum(1 for s in sequence if s["joints_xyz"] is not None)
    report = {
        "method": "monocular_bone_foreshortening_pilot",
        "note": "No camera calibration. Scale/depth approximate. Feasibility only.",
        "fps": args.fps,
        "image_size": image_size,
        "n_frames": len(sequence),
        "n_with_athlete": n_ok,
        "coverage": n_ok / max(1, len(sequence)),
    }
    (out / "mono3d_joints.json").write_text(
        json.dumps({"report": report, "frames": sequence}, indent=2), encoding="utf-8"
    )
    (out / "mono3d_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    # Animation
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    links = [lk for lk in SKELETON_LINKS if lk[0] < 17 and lk[1] < 17]

    def draw(i: int):
        ax.cla()
        ax.set_title(f"T014 camA mono3D pilot  frame {i}/{len(sequence)-1}")
        ax.set_xlim(-0.8, 0.8)
        ax.set_ylim(-0.8, 0.8)
        ax.set_zlim(-0.8, 0.8)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z (approx depth)")
        # rotate view over time
        ax.view_init(elev=15, azim=30 + i * 2)
        s = sequence[i]
        if s["joints_xyz"] is None:
            return
        j = np.asarray(s["joints_xyz"], dtype=float)
        for a, b in links:
            if np.any(np.isnan(j[a])) or np.any(np.isnan(j[b])):
                continue
            ax.plot([j[a, 0], j[b, 0]], [j[a, 1], j[b, 1]], [j[a, 2], j[b, 2]], "b-", lw=2)
        ok = ~np.isnan(j[:, 0])
        ax.scatter(j[ok, 0], j[ok, 1], j[ok, 2], c="r", s=20)

    anim = FuncAnimation(fig, draw, frames=len(sequence), interval=1000 / args.fps)
    mp4 = out / "mono3d_skeleton.mp4"
    try:
        writer = FFMpegWriter(fps=args.fps, bitrate=1800)
        anim.save(str(mp4), writer=writer)
        print("wrote", mp4)
    except Exception as e:
        print("mp4 failed", e)
        # fallback gif via pillow if available
        try:
            anim.save(str(out / "mono3d_skeleton.gif"), writer="pillow", fps=args.fps)
            print("wrote gif")
        except Exception as e2:
            print("gif failed", e2)

    # Also dump a few static angle views
    for i in [0, len(sequence) // 2, len(sequence) - 1]:
        draw(i)
        fig.savefig(out / f"mono3d_frame_{i:03d}.png", dpi=120)
    plt.close(fig)
    print("done", out)


if __name__ == "__main__":
    main()
