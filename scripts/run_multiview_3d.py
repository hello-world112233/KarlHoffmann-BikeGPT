#!/usr/bin/env python3
"""End-to-end multiview 3D session runner (handover entrypoint).

Expected session layout
-----------------------
Camera names are read from ``cameras.yaml``. They may be neutral IDs such as
``A``, ``B`` and ``C``; no semantic front/side/rear label is required.

The runner accepts both a flat layout::

  session_dir/A.mp4
  session_dir/B.mp4
  session_dir/C.mp4

and BikeTrialMatcher's upload layout::

  session_dir/camera_a/original.mp4
  session_dir/camera_b/original.mp4
  session_dir/camera_c/original.mp4

Camera geometry comes exclusively from calibrated K/R/t in ``cameras.yaml``.

Steps
-----
1) Audio-sync the three videos
2) Select athlete from each Sapiens2 predictions.json
3) Fit multi-view CAD skeleton (+ bicycle crank constraint)
4) Write joints3d.json + angles.json under session_dir/output/
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


VIDEO_SUFFIXES = (".mp4", ".mov", ".m4v")


def _first_video(directory: Path) -> Path | None:
    if not directory.is_dir():
        return None
    files = sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES),
        key=lambda path: path.name.lower(),
    )
    preferred = [path for path in files if path.stem.lower() == "original"]
    return preferred[0] if preferred else (files[0] if files else None)


def discover_camera_videos(session: Path, camera_names: list[str]) -> dict[str, Path]:
    """Find videos for arbitrary calibrated camera IDs without decoding media."""
    videos: dict[str, Path] = {}
    for name in camera_names:
        spellings = dict.fromkeys((name, name.lower(), name.upper()))
        direct = next(
            (
                session / f"{spelling}{suffix}"
                for spelling in spellings
                for suffix in VIDEO_SUFFIXES
                if (session / f"{spelling}{suffix}").is_file()
            ),
            None,
        )
        if direct is not None:
            videos[name] = direct
            continue
        for folder in (f"camera_{name.lower()}", f"camera-{name.lower()}", name, name.lower()):
            nested = _first_video(session / folder)
            if nested is not None:
                videos[name] = nested
                break
    return videos


def find_pose_json(session: Path, camera_name: str) -> Path:
    candidates = [
        session / "pose" / camera_name / "sapiens2_predictions.json",
        session / "pose" / camera_name.lower() / "sapiens2_predictions.json",
        session / "pose" / f"{camera_name}_sapiens2_predictions.json",
        session / "pose" / f"{camera_name.lower()}_sapiens2_predictions.json",
    ]
    return next((path for path in candidates if path.exists()), candidates[0])


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("session_dir", type=Path)
    p.add_argument("--cameras", type=Path, default=None, help="Default: session_dir/cameras.yaml")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--height-m", type=float, default=1.75)
    p.add_argument("--crank-m", type=float, default=0.170)
    p.add_argument("--reference", default=None, help="Camera ID used as sync reference (default: first calibrated camera)")
    args = p.parse_args()

    session = args.session_dir
    cameras = args.cameras or (session / "cameras.yaml")
    if not cameras.exists():
        raise SystemExit(f"Missing {cameras}; copy configs/cameras_example.yaml and fill calibration.")

    camera_data = yaml.safe_load(cameras.read_text(encoding="utf-8")) or {}
    camera_names = [str(item["name"]) for item in camera_data.get("cameras", [])]
    if len(camera_names) < 2:
        raise SystemExit("cameras.yaml must contain at least two calibrated cameras")
    videos = discover_camera_videos(session, camera_names)
    if len(videos) < 2:
        expected = ", ".join(camera_names)
        raise SystemExit(
            f"Need videos for at least two calibrated cameras ({expected}); "
            "accepted layouts include A.mp4 or camera_a/original.mp4"
        )

    out = session / "output"
    out.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    sync_json = out / "sync.json"
    sync_cmd = [
        py,
        "scripts/sync_cameras.py",
        "--out",
        str(sync_json),
        "--reference",
        args.reference if args.reference in videos else next(iter(videos)),
    ]
    for name, path in videos.items():
        sync_cmd += ["--video", f"{name}={path}"]
    run(sync_cmd)

    athlete_args = []
    for name in videos:
        pred = find_pose_json(session, name)
        if not pred.exists():
            raise SystemExit(
                f"Missing Sapiens2 predictions for {name}: expected {pred}. "
                "Run Sapiens2 baseline on AutoDL first, then place JSON here."
            )
        athlete_json = out / f"{name}_athlete.json"
        run([py, "scripts/select_athlete.py", str(pred), "--out", str(athlete_json)])
        athlete_args += ["--athlete", f"{name}={athlete_json}"]

    recon_cmd = [
        py,
        "scripts/reconstruct_3d.py",
        "--cameras",
        str(cameras),
        "--sync",
        str(sync_json),
        "--fps",
        str(args.fps),
        "--height-m",
        str(args.height_m),
        "--crank-m",
        str(args.crank_m),
        "--out-dir",
        str(out / "recon"),
        *athlete_args,
    ]
    run(recon_cmd)
    print(json.dumps({"status": "ok", "output": str(out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
