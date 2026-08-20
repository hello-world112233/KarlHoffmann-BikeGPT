#!/usr/bin/env python3
"""Multi-view 3D reconstruction from calibrated cameras + athlete COCO-17 trajectories.

Inputs
------
- cameras.yaml : list of cameras with K, R, t
- one athlete JSON per camera (from scripts/select_athlete.py)
- optional sync.json to map frame indices by time offset

Outputs
-------
- joints3d.json : per-frame 3D joints
- angles.json   : per-frame biomechanical angles + summary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from bike_ai.reconstruct.angles import angles_for_sequence, summarize_angles
from bike_ai.reconstruct.bicycle import BicycleGeometry
from bike_ai.reconstruct.cameras import camera_from_dict
from bike_ai.reconstruct.fit import fit_sequence, save_fit_sequence
from bike_ai.reconstruct.skeleton import SkeletonModel


def load_athlete(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    frames = data["frames"]
    t = len(frames)
    k = int(data.get("num_keypoints", 17))
    xy = np.full((t, k, 2), np.nan, dtype=np.float64)
    sc = np.zeros((t, k), dtype=np.float64)
    for i, fr in enumerate(frames):
        xy[i] = np.asarray(fr["keypoints"], dtype=np.float64)
        sc[i] = np.asarray(fr["keypoint_scores"], dtype=np.float64)
    return xy, sc


def apply_sync_offsets(
    sequences: dict[str, np.ndarray],
    offsets_sec: dict[str, float],
    fps: float,
    reference: str,
) -> dict[str, np.ndarray]:
    """Crop sequences so index 0 is a shared timeline."""
    # Convert offsets to frame shifts relative to global start = max(offset)
    start = max(offsets_sec[n] for n in sequences)
    shifts = {n: int(round((start - offsets_sec[n]) * fps)) for n in sequences}
    # After skipping `shifts[n]` frames in each cam, lengths differ; take min
    lengths = {n: sequences[n].shape[0] - shifts[n] for n in sequences}
    t = min(lengths.values())
    if t <= 0:
        raise RuntimeError("sync offsets leave no overlapping frames")
    out = {}
    for n, arr in sequences.items():
        s = shifts[n]
        out[n] = arr[s : s + t]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cameras", type=Path, required=True, help="cameras.yaml")
    p.add_argument(
        "--athlete",
        action="append",
        required=True,
        help="name=path to athlete JSON, e.g. --athlete side=side_athlete.json",
    )
    p.add_argument("--sync", type=Path, default=None, help="Optional sync.json from sync_cameras.py")
    p.add_argument("--fps", type=float, default=30.0, help="Pose fps for sync frame mapping")
    p.add_argument("--height-m", type=float, default=1.75)
    p.add_argument("--crank-m", type=float, default=0.170)
    p.add_argument("--no-bike-constraint", action="store_true")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    cam_cfg = yaml.safe_load(args.cameras.read_text(encoding="utf-8"))
    cameras = [camera_from_dict(c) for c in cam_cfg["cameras"]]
    cam_names = [c.name for c in cameras]

    athlete_paths: dict[str, Path] = {}
    for item in args.athlete:
        name, path = item.split("=", 1)
        athlete_paths[name] = Path(path)

    missing = [n for n in cam_names if n not in athlete_paths]
    if missing:
        raise SystemExit(f"Missing athlete JSON for cameras: {missing}")

    xy = {n: load_athlete(athlete_paths[n])[0] for n in cam_names}
    sc = {n: load_athlete(athlete_paths[n])[1] for n in cam_names}

    if args.sync is not None:
        sync = json.loads(args.sync.read_text(encoding="utf-8"))
        xy = apply_sync_offsets(xy, sync["offsets_sec"], args.fps, sync["reference"])
        sc = apply_sync_offsets(sc, sync["offsets_sec"], args.fps, sync["reference"])

    seq_2d = [xy[n] for n in cam_names]
    seq_sc = [sc[n] for n in cam_names]

    skeleton = SkeletonModel.from_height_m(args.height_m)
    bicycle = None if args.no_bike_constraint else BicycleGeometry(crank_length_m=args.crank_m)

    results = fit_sequence(cameras, seq_2d, seq_scores=seq_sc, skeleton=skeleton, bicycle=bicycle)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    save_fit_sequence(results, out_dir / "joints3d.json")

    joints_seq = np.stack([r.joints_xyz for r in results], axis=0)
    ang = angles_for_sequence(joints_seq)
    payload = {"frames": ang, "summary": summarize_angles(ang)}
    (out_dir / "angles.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    ok = sum(1 for r in results if r.success)
    mean_rmse = float(np.mean([r.reprojection_rmse_px for r in results if np.isfinite(r.reprojection_rmse_px)]))
    print(f"frames={len(results)} success={ok} mean_reproj_rmse_px={mean_rmse:.2f}")
    print(f"wrote {out_dir / 'joints3d.json'} and {out_dir / 'angles.json'}")


if __name__ == "__main__":
    main()
