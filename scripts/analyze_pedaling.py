"""Step-1 pedal-cycle analysis + visualization.

Reads athlete predictions (from Step 0), fits the pedal ellipse, extracts
crank angle / cadence, enforces left/right antiphase, and writes:
  - ``<out>/pedaling_analysis.json``     : per-frame phase, cadence, phase diff
  - ``<out>/pedaling_summary.png``       : cadence + phase-diff + trajectory plots
  - ``<out>/overlay/qa_frame_*.jpg``     : sampled pedal-circle overlays (optional)

Usage:
  python scripts/analyze_pedaling.py \
      --athlete .../athlete/athlete_predictions.json \
      --out-dir .../pedaling \
      [--frames-dir .../frames/scene1_dji_static_side_10fps --sample 700 1000 1400] \
      [--fps 10] [--smooth one_euro|savgol]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bike_ai.kinematics import analyze_pedaling  # noqa: E402


def _nan_to_none(a: np.ndarray) -> list:
    return [None if not np.isfinite(v) else round(float(v), 4) for v in a]


def make_plots(res, out_path: Path) -> None:
    n = res.num_frames
    t = np.arange(n) / res.fps
    fig, ax = plt.subplots(3, 1, figsize=(12, 10))

    ax[0].plot(t, res.cadence_rpm, color="tab:blue", lw=1)
    ax[0].axhline(res.median_cadence_rpm, color="k", ls="--", lw=0.8,
                  label=f"median {res.median_cadence_rpm:.0f} rpm")
    ax[0].set_ylabel("cadence (rpm)")
    ax[0].set_ylim(0, np.nanpercentile(res.cadence_rpm, 98) * 1.3 + 1)
    ax[0].legend(loc="upper right")
    ax[0].set_title("Cadence over time")

    ax[1].plot(t, np.abs(res.phase_diff_deg), color="tab:orange", lw=1)
    ax[1].axhline(180, color="k", ls="--", lw=0.8, label="ideal 180 deg")
    ax[1].set_ylabel("|L-R phase| (deg)")
    ax[1].set_ylim(0, 360)
    ax[1].legend(loc="upper right")
    sep_note = "separable" if res.lr_separability > 0.4 else "NOT separable (both ankles on near leg)"
    ax[1].set_title(
        f"Left/right phase diff  |  ankle sep {res.ankle_sep_px_median:.0f}px "
        f"= {res.lr_separability:.2f}x circle -> {sep_note}"
    )

    lx, ly = res.foot_xy["lx"], res.foot_xy["ly"]
    rx, ry = res.foot_xy["rx"], res.foot_xy["ry"]
    ax[2].scatter(lx, ly, s=3, color="tab:green", label="left ankle")
    ax[2].scatter(rx, ry, s=3, color="tab:red", label="right ankle")
    c = res.circle
    ell = np.linspace(0, 2 * np.pi, 200)
    ex = c.cx + c.a * np.cos(ell) * np.cos(c.theta) - c.b * np.sin(ell) * np.sin(c.theta)
    ey = c.cy + c.a * np.cos(ell) * np.sin(c.theta) + c.b * np.sin(ell) * np.cos(c.theta)
    ax[2].plot(ex, ey, color="k", lw=1.5, label="fitted pedal ellipse")
    ax[2].plot(c.cx, c.cy, "k+", ms=12)
    ax[2].invert_yaxis()
    ax[2].set_aspect("equal", "datalim")
    ax[2].legend(loc="upper right")
    ax[2].set_title(f"Foot trajectories + pedal ellipse (fit rmse {c.rmse_norm:.3f})")

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def render_overlays(res, athlete_data, frames_dir: Path, out_dir: Path, sample: list[int]) -> None:
    import cv2

    c = res.circle
    out_dir.mkdir(parents=True, exist_ok=True)
    lx, ly = res.foot_xy["lx"], res.foot_xy["ly"]
    rx, ry = res.foot_xy["rx"], res.foot_xy["ry"]
    for fi in sample:
        frame = athlete_data["frames"][fi]
        img = cv2.imread(str(frames_dir / frame["image_name"]))
        if img is None:
            continue
        cv2.ellipse(img, (int(c.cx), int(c.cy)),
                    (int(c.a), int(c.b)), np.degrees(c.theta), 0, 360, (0, 255, 255), 4)
        cv2.drawMarker(img, (int(c.cx), int(c.cy)), (0, 255, 255),
                       cv2.MARKER_CROSS, 40, 4)
        for i in range(max(0, fi - 20), fi + 1):
            if np.isfinite(lx[i]):
                cv2.circle(img, (int(lx[i]), int(ly[i])), 4, (0, 255, 0), -1)
            if np.isfinite(rx[i]):
                cv2.circle(img, (int(rx[i]), int(ry[i])), 4, (0, 0, 255), -1)
        if np.isfinite(lx[fi]):
            cv2.circle(img, (int(lx[fi]), int(ly[fi])), 12, (0, 255, 0), 3)
        if np.isfinite(rx[fi]):
            cv2.circle(img, (int(rx[fi]), int(ry[fi])), 12, (0, 0, 255), 3)
        cad = res.cadence_rpm[fi]
        pd = res.phase_diff_deg[fi]
        cv2.rectangle(img, (0, 0), (1100, 100), (0, 0, 0), -1)
        txt = f"#{fi}  cadence={cad:.0f}rpm  L-R={pd:.0f}deg" if np.isfinite(cad) else f"#{fi}  (no cadence)"
        cv2.putText(img, txt, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 4)
        out = out_dir / f"qa_frame_{fi:06d}.jpg"
        cv2.imwrite(str(out), img)
        print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step-1 pedal analysis.")
    parser.add_argument("--athlete", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--smooth", default="one_euro", choices=["one_euro", "savgol"])
    parser.add_argument("--use-toe", action="store_true")
    parser.add_argument("--frames-dir", type=Path, default=None)
    parser.add_argument("--sample", type=int, nargs="*", default=None)
    args = parser.parse_args()

    data = json.loads(args.athlete.read_text())
    res = analyze_pedaling(data, fps=args.fps, smooth_method=args.smooth, use_toe=args.use_toe)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    c = res.circle
    analysis = {
        "num_frames": res.num_frames,
        "fps": res.fps,
        "median_cadence_rpm": round(res.median_cadence_rpm, 2),
        "antiphase_error_deg": round(res.antiphase_error_deg, 2),
        "ankle_sep_px_median": round(res.ankle_sep_px_median, 1),
        "lr_separability": round(res.lr_separability, 3),
        "lr_separable": bool(res.lr_separability > 0.4),
        "direction": res.direction,
        "num_lr_swaps_fixed": int(res.swapped.sum()),
        "pedal_ellipse": {
            "cx": round(c.cx, 2), "cy": round(c.cy, 2),
            "semi_a": round(c.a, 2), "semi_b": round(c.b, 2),
            "theta_deg": round(np.degrees(c.theta), 2),
            "fit_rmse_norm": round(c.rmse_norm, 4), "n_points": c.n_points,
        },
        "per_frame": {
            "phase_left_deg": _nan_to_none(np.degrees(res.phase_left)),
            "phase_right_deg": _nan_to_none(np.degrees(res.phase_right)),
            "cadence_rpm": _nan_to_none(res.cadence_rpm),
            "phase_diff_deg": _nan_to_none(res.phase_diff_deg),
            "swapped": [bool(v) for v in res.swapped],
        },
    }
    (args.out_dir / "pedaling_analysis.json").write_text(json.dumps(analysis))
    make_plots(res, args.out_dir / "pedaling_summary.png")

    print(f"median cadence: {res.median_cadence_rpm:.1f} rpm")
    print(f"L/R ankle sep (median): {res.ankle_sep_px_median:.0f} px  "
          f"separability={res.lr_separability:.2f} "
          f"({'separable' if res.lr_separability > 0.4 else 'NOT separable in this view'})")
    print(f"antiphase error (median |L-R|-180): {res.antiphase_error_deg:.1f} deg")
    print(f"pedal ellipse: center=({c.cx:.0f},{c.cy:.0f}) axes=({c.a:.0f},{c.b:.0f}) "
          f"theta={np.degrees(c.theta):.1f}deg rmse={c.rmse_norm:.3f}")
    print(f"wrote {args.out_dir/'pedaling_analysis.json'}")
    print(f"wrote {args.out_dir/'pedaling_summary.png'}")

    if args.frames_dir and args.sample:
        render_overlays(res, data, args.frames_dir, args.out_dir / "overlay", args.sample)


if __name__ == "__main__":
    main()
