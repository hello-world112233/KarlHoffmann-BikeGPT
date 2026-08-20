#!/usr/bin/env python3
"""Calibrate three DJI cameras from T011 (clap-synced) checkerboard views.

Writes:
  bike-project/cameras.yaml
  bike-project/diagnostics/calibration_report.json
  diagnostics/calib_vis/*.jpg

Square size unknown → use 1.0 (relative units). Scale later if physical mm known.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path("/root/autodl-tmp/bike_projects/bike-project")
VIDEOS = {
    "camera_a": ROOT / "trials/T011/camera_a/original.mp4",
    "camera_b": ROOT / "trials/T011/camera_b/original.mp4",
    "camera_c": ROOT / "trials/T011/camera_c/original.mp4",
}
# Same clap event: A@8.87, B@10.69, C@4.70 → local = T_a + offset
# offset_b = +1.82 means B_local = A_local + 1.82
CLAP_OFFSET = {"camera_a": 0.0, "camera_b": 1.82, "camera_c": -4.17}
FPS = 120000 / 1001
SIZES = [(9, 6), (8, 6), (7, 5), (6, 4), (10, 7), (8, 5), (5, 4), (11, 8), (9, 7), (6, 5), (7, 6), (4, 3)]
SQUARE = 1.0  # relative
DIAG = ROOT / "diagnostics" / "t011_calibration"
VIS = DIAG / "calib_vis"
CACHE = DIAG / "frames_sync"


def ff_frame(path: Path, t: float, out: Path, scale: int = 1280) -> np.ndarray | None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists() or out.stat().st_size < 1000:
        # extract scaled for speed
        vf = f"scale={scale}:-1"
        r = subprocess.run(
            [
                "ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", str(path),
                "-frames:v", "1", "-vf", vf, "-q:v", "2", str(out),
            ],
            capture_output=True,
        )
        if r.returncode != 0 or not out.exists():
            return None
    img = cv2.imread(str(out))
    return img


def detect_board(gray: np.ndarray):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    best = None
    for cols, rows in SIZES:
        ok, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
        if not ok:
            continue
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), term)
        pts = corners.reshape(-1, 2)
        area = float((pts[:, 0].max() - pts[:, 0].min()) * (pts[:, 1].max() - pts[:, 1].min()))
        frac = area / (gray.shape[0] * gray.shape[1])
        if best is None or frac > best["frac"]:
            best = {"size": (cols, rows), "corners": corners, "frac": frac}
    return best


def obj_points(cols: int, rows: int, square: float) -> np.ndarray:
    obj = np.zeros((cols * rows, 3), np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    return obj


def rotate_rodrigues(rvec):
    R, _ = cv2.Rodrigues(rvec)
    return R


def main():
    VIS.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    # Sample A-timeline times where board was seen (~5–40s and mid)
    times_a = list(np.arange(5.0, 45.0, 0.5)) + list(np.arange(45.0, 90.0, 2.0))
    print(f"Sampling {len(times_a)} timestamps on camera_a timeline...")

    # per-cam detections: list of {t_a, size, corners_fullres_scaled, img_shape_scaled, scale_factor}
    dets: dict[str, list] = {c: [] for c in VIDEOS}
    size_votes: dict[tuple, int] = {}

    for t_a in times_a:
        for cam, path in VIDEOS.items():
            t_local = t_a + CLAP_OFFSET[cam]
            if t_local < 0.2:
                continue
            # duration check skip later
            out = CACHE / f"{cam}_ta{t_a:.2f}.jpg"
            img = ff_frame(path, t_local, out, scale=1280)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hit = detect_board(gray)
            if not hit:
                continue
            size_votes[hit["size"]] = size_votes.get(hit["size"], 0) + 1
            # scale corners to full 3840x2160 (OpenCV may return Nx1x2 or Nx2)
            sx, sy = 3840 / img.shape[1], 2160 / img.shape[0]
            corners = np.asarray(hit["corners"], dtype=np.float32)
            pts = corners.reshape(-1, 2).copy()
            pts[:, 0] *= sx
            pts[:, 1] *= sy
            corners_full = pts.reshape(-1, 1, 2)
            dets[cam].append(
                {
                    "t_a": float(t_a),
                    "t_local": float(t_local),
                    "size": hit["size"],
                    "corners": corners_full,
                    "frac": hit["frac"],
                    "preview": str(out),
                }
            )
            # vis
            vis = img.copy()
            cv2.drawChessboardCorners(vis, hit["size"], hit["corners"], True)
            cv2.imwrite(str(VIS / f"{cam}_ta{t_a:.2f}_det.jpg"), vis)
            print(f"  HIT {cam} t_a={t_a:.1f} size={hit['size']} frac={hit['frac']:.3f}")

    print("detections:", {c: len(dets[c]) for c in dets})
    print("size votes:", size_votes)
    if not size_votes:
        # fallback: also try T012 short calib
        print("No boards on T011 sample — trying denser + T012...")
        raise SystemExit("No checkerboard detections. Abort.")

    board_size = max(size_votes, key=size_votes.get)
    print("chosen board_size (inner corners)", board_size)
    cols, rows = board_size
    objp = obj_points(cols, rows, SQUARE)

    # Filter dets to chosen size
    for c in dets:
        dets[c] = [d for d in dets[c] if d["size"] == board_size]
        print(f"  {c}: {len(dets[c])} with size {board_size}")

    # Intrinsics per camera
    img_size = (3840, 2160)
    Ks, Ds, rms_in = {}, {}, {}
    for cam in VIDEOS:
        objpoints, imgpoints = [], []
        for d in dets[cam]:
            objpoints.append(objp)
            imgpoints.append(d["corners"].astype(np.float32))
        if len(objpoints) < 3:
            print(f"WARN {cam}: only {len(objpoints)} views for intrinsics")
        if len(objpoints) < 1:
            raise SystemExit(f"No detections for {cam}")
        rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, img_size, None, None
        )
        Ks[cam], Ds[cam], rms_in[cam] = K, D, float(rms)
        print(f"intrinsics {cam}: rms={rms:.3f}px fx={K[0,0]:.1f}")

    # Pairwise stereo at synchronized times (match t_a within 0.26s)
    def match_pairs(ca, cb, tol=0.26):
        pairs = []
        for da in dets[ca]:
            for db in dets[cb]:
                if abs(da["t_a"] - db["t_a"]) <= tol:
                    pairs.append((da, db))
                    break
        return pairs

    # Use camera_a as world origin: R=I, t=0
    # For each other cam, accumulate stereo poses vs A, take median
    extrinsics = {
        "camera_a": {
            "R": np.eye(3),
            "t": np.zeros(3),
        }
    }
    pair_reports = {}
    for other in ("camera_b", "camera_c"):
        pairs = match_pairs("camera_a", other)
        print(f"stereo pairs A-{other}: {len(pairs)}")
        Rs, ts, rms_list = [], [], []
        for da, db in pairs:
            ok, E, R, t, mask = False, None, None, None, None
            # essential from undistorted points? use stereoCalibrate with fixed K
            flags = cv2.CALIB_FIX_INTRINSIC
            rms, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
                [objp],
                [da["corners"].astype(np.float32)],
                [db["corners"].astype(np.float32)],
                Ks["camera_a"],
                Ds["camera_a"],
                Ks[other],
                Ds[other],
                img_size,
                flags=flags,
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
            )
            Rs.append(R)
            ts.append(T.reshape(3))
            rms_list.append(float(rms))
        if not Rs:
            # fallback: PnP each view to board, relative = Ra^{-1} Rb etc at nearest times
            print(f"  no stereoCalibrate pairs for {other}, trying PnP chain...")
            for da in dets["camera_a"]:
                # nearest other
                db = min(dets[other], key=lambda d: abs(d["t_a"] - da["t_a"]), default=None)
                if db is None or abs(db["t_a"] - da["t_a"]) > 0.5:
                    continue
                ok1, r1, t1 = cv2.solvePnP(objp, da["corners"], Ks["camera_a"], Ds["camera_a"])
                ok2, r2, t2 = cv2.solvePnP(objp, db["corners"], Ks[other], Ds[other])
                if not (ok1 and ok2):
                    continue
                R1 = rotate_rodrigues(r1)
                R2 = rotate_rodrigues(r2)
                # board in cam: Xc = R Xw + t
                # cam_other from cam_a: X_a = R1 Xw + t1; X_b = R2 Xw + t2
                # X_b = R2 R1.T (X_a - t1) + t2 = (R2 R1.T) X_a + (t2 - R2 R1.T t1)
                R_ba = R2 @ R1.T
                t_ba = t2.reshape(3) - R_ba @ t1.reshape(3)
                Rs.append(R_ba)
                ts.append(t_ba)
                rms_list.append(float("nan"))
        if not Rs:
            raise SystemExit(f"Cannot estimate extrinsics for {other}")

        # Robust average of rotations (chordal) / translations
        R_stack = np.stack(Rs, 0)
        # pick median by translation norm closeness to median
        t_stack = np.stack(ts, 0)
        t_med = np.median(t_stack, axis=0)
        idx = int(np.argmin(np.linalg.norm(t_stack - t_med, axis=1)))
        # Or use the stereo with lowest rms if available
        if np.any(np.isfinite(rms_list)):
            idx = int(np.nanargmin(rms_list))
        R_use, t_use = Rs[idx], ts[idx]
        # Normalize scale: set ||t|| of B relative to A using mean stereo baseline of AB
        extrinsics[other] = {"R": R_use, "t": t_use}
        pair_reports[f"camera_a-{other}"] = {
            "n_pairs": len(Rs),
            "rms_list": [float(x) if x == x else None for x in rms_list],
            "chosen_index": idx,
            "t_norm": float(np.linalg.norm(t_use)),
        }
        print(f"  {other}: ||t||={np.linalg.norm(t_use):.3f} (relative units)")

    # Optional: refine C via B if AC weak — skip for now

    # Scale: set camera_b translation norm to 1.0 for readability (relative)
    t_b = extrinsics["camera_b"]["t"]
    scale = 1.0 / (np.linalg.norm(t_b) + 1e-9)
    for cam in extrinsics:
        extrinsics[cam]["t"] = extrinsics[cam]["t"] * scale

    # Connectivity
    connected = all(len(dets[c]) > 0 for c in VIDEOS) and "camera_b" in extrinsics and "camera_c" in extrinsics
    # edges with simultaneous views
    edges = {
        "AB": len(match_pairs("camera_a", "camera_b")),
        "AC": len(match_pairs("camera_a", "camera_c")),
        "BC": len(match_pairs("camera_b", "camera_c")),
    }

    cameras_yaml = {
        "meta": {
            "source_trial": "T011",
            "method": "clap_sync + OpenCV calibrateCamera/stereoCalibrate/PnP",
            "board_inner_corners": [cols, rows],
            "square_size": SQUARE,
            "square_size_unit": "relative (unknown physical mm; do not treat as meters)",
            "image_size": [3840, 2160],
            "sync": {
                "reference": "camera_a",
                "clap_offset_sec": CLAP_OFFSET,
                "note": "local_time = t_on_camera_a + offset",
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
            "warning": (
                "Feasibility calibration. Square size unknown → scale is relative. "
                "Reprojection RMS reported in report. Validate before delivery."
            ),
        },
        "cameras": {},
    }
    for cam in ("camera_a", "camera_b", "camera_c"):
        R = extrinsics[cam]["R"]
        t = extrinsics[cam]["t"]
        # world-from-camera or camera-from-world? Use OpenCV convention:
        # X_cam = R_cw @ X_world + t_cw  (here world = camera_a's frame at identity)
        # For camera_a, R=I,t=0 means world = cam_a
        # For others, R,t map world(cam_a) points into that camera
        cameras_yaml["cameras"][cam] = {
            "image_size": [3840, 2160],
            "K": Ks[cam].tolist(),
            "dist_coeffs": Ds[cam].reshape(-1).tolist(),
            "R_world_to_cam": R.tolist(),
            "t_world_to_cam": t.reshape(-1).tolist(),
            "intrinsics_rms_px": rms_in[cam],
            "n_board_views": len(dets[cam]),
        }

    out_yaml = ROOT / "cameras.yaml"
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(cameras_yaml, f, sort_keys=False, allow_unicode=True)

    report = {
        "stage": 3,
        "source_trial": "T011",
        "board_size_votes": {str(k): v for k, v in size_votes.items()},
        "chosen_board_inner_corners": [cols, rows],
        "detections_per_camera": {c: len(dets[c]) for c in dets},
        "intrinsics_rms_px": rms_in,
        "pair_reports": pair_reports,
        "simultaneous_edge_counts": edges,
        "connected": bool(connected and (edges["AB"] + edges["AC"] + edges["BC"] >= 2 or (edges["AB"] and edges["AC"]))),
        "cameras_yaml": str(out_yaml),
        "scale_note": "Translations scaled so ||t_camera_b||=1 (relative units)",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    # also copy report to diagnostics/
    (DIAG / "calibration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (ROOT / "diagnostics" / "calibration_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print("WROTE", out_yaml)


if __name__ == "__main__":
    main()
