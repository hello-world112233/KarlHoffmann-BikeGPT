#!/usr/bin/env python3
"""Generate the constrained T014 ROTA demo and print before/after QA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROTA = ROOT / "apps" / "rota"
sys.path.insert(0, str(ROTA))

from bike_geometry import (  # noqa: E402
    build_optimized_analysis,
    load_body_calibration,
    load_or_create_calibration,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--rota-dir", type=Path, default=ROTA)
    args = parser.parse_args()

    demo = args.rota_dir / "data" / "demo"
    motionbert_path = demo / "joints.json"
    athlete_path = demo / "athlete_2d.json"
    calibration_path = demo / "bike_calibration.json"
    body_calibration_path = demo / "body_calibration.json"
    output_path = demo / "joints_constrained.json"

    missing = [path for path in (motionbert_path, athlete_path) if not path.exists()]
    if missing:
        raise SystemExit("missing required demo inputs: " + ", ".join(map(str, missing)))

    motionbert = json.loads(motionbert_path.read_text(encoding="utf-8"))
    athlete = json.loads(athlete_path.read_text(encoding="utf-8"))
    calibration = load_or_create_calibration(calibration_path)
    body_calibration = load_body_calibration(
        body_calibration_path, n_frames=len(athlete.get("frames") or [])
    )
    optimized, report = build_optimized_analysis(
        motionbert,
        athlete,
        calibration,
        body_calibration,
        fps=10.0,
        steps=args.steps,
    )
    output_path.write_text(json.dumps(optimized, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
