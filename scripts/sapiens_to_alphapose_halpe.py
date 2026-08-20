#!/usr/bin/env python3
"""Convert locked-athlete Sapiens2 predictions → AlphaPose Halpe-26 JSON for MotionBERT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def coco17_to_halpe26(xy17: list, sc17: list) -> list[float]:
    """Build Halpe-26 (x,y,conf)*26 from COCO-17 body points (Sapiens first 17)."""
    pts = []
    for i in range(17):
        x, y = xy17[i]
        c = float(sc17[i]) if i < len(sc17) else 0.0
        pts.append([float(x), float(y), c])

    def mid(i, j):
        return [
            0.5 * (pts[i][0] + pts[j][0]),
            0.5 * (pts[i][1] + pts[j][1]),
            0.5 * (pts[i][2] + pts[j][2]),
        ]

    neck = mid(5, 6)
    hip = mid(11, 12)
    head = [
        pts[0][0] + 0.25 * (pts[0][0] - neck[0]),
        pts[0][1] + 0.25 * (pts[0][1] - neck[1]),
        pts[0][2],
    ]
    # 17 Head, 18 Neck, 19 Hip, 20-25 foot proxies ≈ ankles
    extra = [
        head,
        neck,
        hip,
        pts[15],
        pts[16],
        pts[15],
        pts[16],
        pts[15],
        pts[16],
    ]
    flat: list[float] = []
    for p in pts + extra:
        flat.extend(p)
    return flat


def convert(athlete_json: Path, out_json: Path) -> dict:
    data = json.loads(athlete_json.read_text(encoding="utf-8"))
    rows = []
    for i, fr in enumerate(data.get("frames") or []):
        insts = fr.get("instances") or []
        if not insts:
            # pad empty with zeros so temporal length stays aligned
            kps = [0.0] * (26 * 3)
        else:
            inst = insts[0]
            xy = (inst.get("keypoints") or [])[:17]
            sc = (inst.get("keypoint_scores") or [])[:17]
            if len(xy) < 17:
                kps = [0.0] * (26 * 3)
            else:
                kps = coco17_to_halpe26(xy, sc)
        rows.append(
            {
                "image_id": fr.get("image_name") or f"frame_{i:06d}.jpg",
                "idx": 0,
                "keypoints": kps,
                "box": insts[0].get("bbox") if insts else [0, 0, 0, 0],
            }
        )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows), encoding="utf-8")
    return {"n_frames": len(rows), "out": str(out_json)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--athlete-json",
        type=Path,
        default=Path(
            "/root/autodl-tmp/bike_projects/bike-project/diagnostics/"
            "t014_mono3d_pilot/athlete/athlete_predictions.json"
        ),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/root/autodl-tmp/bike_projects/bike-project/diagnostics/"
            "t014_mono3d_pilot/motionbert/alphapose_halpe.json"
        ),
    )
    args = ap.parse_args()
    info = convert(args.athlete_json, args.out)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
