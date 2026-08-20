#!/usr/bin/env python3
"""Validate and run all ready BikeTrialMatcher trials from a project manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


def selected_complete_trials(manifest: dict, camera_names: list[str]) -> list[dict]:
    """Return selected trials that contain remote files for every calibrated camera ID."""
    ready: list[dict] = []
    for trial in manifest.get("trials", []):
        if not trial.get("selected_for_upload"):
            continue
        streams = trial.get("streams", {})
        if all(streams.get(name) and streams[name].get("remote_path") for name in camera_names):
            ready.append(trial)
    return ready


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--trial-id", action="append", default=[], help="Optional T001 filter; repeatable")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--height-m", type=float, default=1.75)
    parser.add_argument("--crank-m", type=float, default=0.170)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    calibration = yaml.safe_load(args.cameras.read_text(encoding="utf-8")) or {}
    camera_names = [str(item["name"]) for item in calibration.get("cameras", [])]
    if len(camera_names) < 2:
        raise SystemExit("Calibration must contain at least two named cameras")

    trials = selected_complete_trials(manifest, camera_names)
    requested = set(args.trial_id)
    if requested:
        trials = [trial for trial in trials if trial.get("trial_id") in requested]
    if not trials:
        raise SystemExit("No selected, fully uploaded trial matches the calibrated camera IDs")

    missing_files: list[str] = []
    for trial in trials:
        for name in camera_names:
            path = Path(trial["streams"][name]["remote_path"])
            if not path.is_file():
                missing_files.append(f"{trial['trial_id']} Camera {name}: {path}")
    if missing_files:
        raise SystemExit("Manifest paths are not ready:\n" + "\n".join(missing_files))

    summary = {
        "status": "ready" if args.validate_only else "running",
        "camera_ids": camera_names,
        "trials": [trial["trial_id"] for trial in trials],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.validate_only:
        return

    runner = Path(__file__).with_name("run_multiview_3d.py")
    project_root = args.manifest.parent
    for trial in trials:
        trial_id = trial["trial_id"]
        session = project_root / "trials" / trial_id
        command = [
            sys.executable,
            str(runner),
            str(session),
            "--cameras",
            str(args.cameras),
            "--fps",
            str(args.fps),
            "--height-m",
            str(args.height_m),
            "--crank-m",
            str(args.crank_m),
        ]
        print("+", " ".join(command), flush=True)
        subprocess.check_call(command, cwd=runner.parent.parent)


if __name__ == "__main__":
    main()
