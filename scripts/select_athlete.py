"""Select the athlete track from multi-person pose predictions (Step 0).

Reads a Sapiens2-style ``predictions.json``, links detections into tracks,
picks the single athlete track (persistent + central + pedaling) and writes:
  - ``<out>/athlete_predictions.json``  : one instance per frame (or none)
  - ``<out>/athlete_selection_report.json`` : per-track scores + choice

Usage:
  python scripts/select_athlete.py \
      --predictions .../sapiens2_0.4b_predictions.json \
      --out-dir .../athlete \
      [--fps 10]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bike_ai.tracking import select_athlete  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Select athlete track (Step 0).")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--min-coverage", type=float, default=0.3)
    parser.add_argument("--select-radius-frac", type=float, default=0.12)
    parser.add_argument("--template-thr", type=float, default=0.06)
    parser.add_argument("--hip-tol-frac", type=float, default=0.06)
    parser.add_argument("--no-anchor", action="store_true", help="use best track instead of anchor reselection")
    parser.add_argument("--max-center-dist-frac", type=float, default=0.08)
    parser.add_argument("--max-gap", type=int, default=8)
    args = parser.parse_args()

    data = json.loads(args.predictions.read_text())
    sel, chosen_track, tracks = select_athlete(
        data,
        fps=args.fps,
        min_coverage=args.min_coverage,
        use_anchor=not args.no_anchor,
        select_radius_frac=args.select_radius_frac,
        template_thr=args.template_thr,
        hip_tol_frac=args.hip_tol_frac,
        max_center_dist_frac=args.max_center_dist_frac,
        max_gap=args.max_gap,
    )

    out_frames = []
    for fi, frame in enumerate(data["frames"]):
        det = chosen_track.detections.get(fi)
        instances = []
        if det is not None:
            instances = [
                {
                    "bbox": det.bbox.tolist(),
                    "keypoints": det.keypoints.tolist(),
                    "keypoint_scores": det.scores.tolist(),
                    "track_id": sel.track_id,
                }
            ]
        out_frames.append(
            {"image_name": frame.get("image_name", f"frame_{fi:06d}.jpg"), "instances": instances}
        )

    out_data = {
        "video": data.get("video"),
        "image_size": data["image_size"],
        "num_keypoints": data.get("num_keypoints"),
        "kpt_thr_used": data.get("kpt_thr_used"),
        "source_predictions": str(args.predictions),
        "athlete_track_id": sel.track_id,
        "selection_method": sel.method,
        "anchor_norm": sel.anchor_norm,
        "cadence_hz": sel.cadence_hz,
        "riding_segment": sel.riding_segment,
        "frames": out_frames,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "athlete_predictions.json").write_text(json.dumps(out_data))

    report = {
        "athlete_track_id": sel.track_id,
        "selection_method": sel.method,
        "anchor_norm": sel.anchor_norm,
        "cadence_hz": sel.cadence_hz,
        "num_frames_total": sel.num_frames_total,
        "num_frames_present": sel.num_frames_present,
        "coverage": sel.num_frames_present / sel.num_frames_total,
        "riding_segment": sel.riding_segment,
        "num_tracks": len(tracks),
        "top_tracks": [asdict(s) for s in sel.scores[:10]],
    }
    (args.out_dir / "athlete_selection_report.json").write_text(json.dumps(report, indent=2))

    print(f"tracks built: {len(tracks)}")
    print(
        f"method={sel.method} anchor={tuple(round(x,3) for x in sel.anchor_norm)} "
        f"cadence={sel.cadence_hz:.2f}Hz ({sel.cadence_hz*60:.0f} rpm)"
    )
    print(
        f"athlete: present {sel.num_frames_present}/{sel.num_frames_total} "
        f"({100 * sel.num_frames_present / sel.num_frames_total:.1f}%), "
        f"riding segment {sel.riding_segment}"
    )
    print("\ntop candidate tracks (by total score):")
    hdr = f"{'id':>4} {'total':>6} {'cover':>6} {'centerNorm':>16} {'cstd':>6} {'conf':>5} {'pedal':>6} {'pRatio':>6} {'pHz':>5} {'pAmp':>5}"
    print(hdr)
    for s in sel.scores[:8]:
        cx, cy = s.center_norm
        print(
            f"{s.track_id:>4} {s.total:>6.3f} {s.coverage:>6.3f} "
            f"({cx:>5.2f},{cy:>5.2f})     {s.center_std_norm:>6.3f} {s.mean_conf:>5.2f} "
            f"{s.pedal_score:>6.3f} {s.pedal_ratio:>6.3f} {s.pedal_freq_hz:>5.2f} {s.pedal_amp_norm:>5.2f}"
        )
    print(f"\nwrote: {args.out_dir/'athlete_predictions.json'}")
    print(f"wrote: {args.out_dir/'athlete_selection_report.json'}")


if __name__ == "__main__":
    main()
