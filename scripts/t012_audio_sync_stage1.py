#!/usr/bin/env python3
"""Stage 1: T012 three-camera audio sync diagnostics.

Writes diagnostics/t012_audio_sync.json under the bike-project root.
Does not modify original MP4 files.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bike_ai.sync.audio_sync import extract_mono_pcm, estimate_lag_sec, _normalize  # noqa: E402


FPS = 120000 / 1001  # ~119.88


def gcc_phat_lag(
    a: np.ndarray,
    b: np.ndarray,
    sample_rate: int,
    max_lag_sec: float,
) -> tuple[float, float]:
    """GCC-PHAT lag. Positive => b starts later than a (same convention as estimate_lag_sec)."""
    a = _normalize(a)
    b = _normalize(b)
    n = int(2 ** np.ceil(np.log2(len(a) + len(b) - 1)))
    fa = np.fft.rfft(a, n=n)
    fb = np.fft.rfft(b, n=n)
    R = fa * np.conj(fb)
    R /= np.abs(R) + 1e-12
    cc = np.fft.irfft(R, n=n)
    max_lag = int(max_lag_sec * sample_rate)
    # cc[0] = lag 0; cc[-k] = lag +k in circular sense for "b delayed"
    # Align with correlate convention used in estimate_lag_sec
    lags = np.arange(-(len(b) - 1), len(a))
    # Map circular to linear: use first max_lag and last max_lag bins
    # Simpler: rebuild via np.correlate on whitened? Use cc shift:
    # irfft GCC peak index i means lag = i if i < n/2 else i - n for "a relative to b"
    peak_region = np.concatenate([cc[-max_lag:], cc[: max_lag + 1]])
    lag_axis = np.arange(-max_lag, max_lag + 1)
    idx = int(np.argmax(np.abs(peak_region)))
    # Flip sign so positive => other (b) starts later than reference (a)
    lag_samples = -int(lag_axis[idx])
    # Peak sharpness: peak / median of abs
    mag = np.abs(peak_region)
    peak = float(mag[idx])
    med = float(np.median(mag) + 1e-12)
    score = peak / med  # not Pearson; relative peak height
    return lag_samples / float(sample_rate), float(peak / (mag.sum() + 1e-12)), score


def segment_lags(
    ref: np.ndarray,
    other: np.ndarray,
    sample_rate: int,
    max_lag_sec: float,
    segment_sec: float,
    hop_sec: float,
) -> list[dict]:
    """Estimate lag on sliding segments of the overlap window."""
    # First get coarse full lag to align, then refine per segment in aligned domain
    coarse, coarse_score = estimate_lag_sec(ref, other, sample_rate, max_lag_sec)
    # Align to common timeline
    lag_samp = int(round(coarse * sample_rate))
    if lag_samp >= 0:
        a = ref[lag_samp:]
        b = other[: len(a)]
    else:
        b = other[-lag_samp:]
        a = ref[: len(b)]
    m = min(len(a), len(b))
    a, b = a[:m], b[:m]

    seg = int(segment_sec * sample_rate)
    hop = int(hop_sec * sample_rate)
    out: list[dict] = []
    if m < seg:
        return [
            {
                "t0_sec": 0.0,
                "t1_sec": m / sample_rate,
                "delta_from_coarse_sec": 0.0,
                "lag_sec": coarse,
                "xcorr_score": coarse_score,
                "note": "audio shorter than one segment; only coarse lag",
            }
        ]

    t = 0
    while t + seg <= m:
        aa = a[t : t + seg]
        bb = b[t : t + seg]
        # Fine lag within ±0.5s around 0 (already coarse-aligned)
        dlag, dscore = estimate_lag_sec(aa, bb, sample_rate, max_lag_sec=0.5)
        out.append(
            {
                "t0_sec": round(t / sample_rate, 4),
                "t1_sec": round((t + seg) / sample_rate, 4),
                "delta_from_coarse_sec": round(dlag, 6),
                "lag_sec": round(coarse + dlag, 6),
                "xcorr_score": round(dscore, 6),
            }
        )
        t += hop
    return out


def pairwise_report(
    name_a: str,
    name_b: str,
    pcm_a: np.ndarray,
    pcm_b: np.ndarray,
    sample_rate: int,
    max_lag_sec: float,
    fps: float,
) -> dict:
    lag_x, score_x = estimate_lag_sec(pcm_a, pcm_b, sample_rate, max_lag_sec)
    lag_g, peak_frac, peak_ratio = gcc_phat_lag(pcm_a, pcm_b, sample_rate, max_lag_sec)
    segs = segment_lags(pcm_a, pcm_b, sample_rate, max_lag_sec, segment_sec=8.0, hop_sec=4.0)
    deltas = [s["delta_from_coarse_sec"] for s in segs if "delta_from_coarse_sec" in s]
    scores = [s["xcorr_score"] for s in segs if "xcorr_score" in s]

    stable = False
    stability_note = ""
    if len(deltas) >= 2:
        spread = float(np.ptp(deltas))
        std = float(np.std(deltas))
        median_score = float(np.median(scores))
        # Stable if segment fine-lags within ~1 frame and scores not tiny
        frame_tol = 1.5 / fps  # ~12.5 ms
        stable = spread <= frame_tol * 3 and median_score >= 0.05
        stability_note = (
            f"segment_delta_spread_sec={spread:.6f}, std={std:.6f}, "
            f"median_xcorr={median_score:.4f}, frame_tol_3x={frame_tol*3:.6f}"
        )
    else:
        stability_note = "insufficient segments"
        stable = score_x >= 0.15

    clear_peak = score_x >= 0.12 or peak_ratio >= 8.0
    return {
        "pair": f"{name_a}-{name_b}",
        "convention": f"positive lag => {name_b} starts later than {name_a}",
        "xcorr": {
            "lag_sec": round(lag_x, 6),
            "lag_frames": round(lag_x * fps, 3),
            "peak_score_pearson": round(score_x, 6),
        },
        "gcc_phat": {
            "lag_sec": round(lag_g, 6),
            "lag_frames": round(lag_g * fps, 3),
            "peak_fraction": round(peak_frac, 8),
            "peak_to_median_ratio": round(peak_ratio, 3),
        },
        "xcorr_vs_gcc_delta_sec": round(abs(lag_x - lag_g), 6),
        "segments": segs,
        "stable_across_segments": stable,
        "clear_peak": clear_peak,
        "stability_note": stability_note,
    }


def build_common_timeline(offsets_vs_a: dict[str, float], durations: dict[str, float]) -> dict:
    """offsets_vs_a: cam -> start time of that cam on A's timeline (A=0).

    If B lag vs A is +0.2, B starts 0.2s later on wall clock relative to A's t=0
    meaning when A is at t, B content is at t - 0.2? 

    estimate_lag_sec: positive lag => other starts later than reference.
    So offset_B = lag(A,B): B's file t=0 corresponds to A's timeline t = lag.
    On common timeline T:
      A local time = T - offset_A  (offset_A=0)
      B local time = T - offset_B
    Common overlap: T in [max(offsets), min(offset_i + duration_i)]
    """
    cams = list(offsets_vs_a.keys())
    t0 = max(offsets_vs_a[c] for c in cams)
    t1 = min(offsets_vs_a[c] + durations[c] for c in cams)
    return {
        "t0_on_camera_a_timeline_sec": round(t0, 6),
        "t1_on_camera_a_timeline_sec": round(t1, 6),
        "overlap_duration_sec": round(max(0.0, t1 - t0), 6),
        "per_camera_local_start_sec": {
            c: round(t0 - offsets_vs_a[c], 6) for c in cams
        },
        "per_camera_local_end_sec": {
            c: round(t1 - offsets_vs_a[c], 6) for c in cams
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--project-root",
        default="/root/autodl-tmp/bike_projects/bike-project",
    )
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--max-lag-sec", type=float, default=15.0)
    ap.add_argument("--fps", type=float, default=FPS)
    args = ap.parse_args()

    root = Path(args.project_root)
    videos = {
        "camera_a": root / "calibration/T012/camera_a/original.mp4",
        "camera_b": root / "calibration/T012/camera_b/original.mp4",
        "camera_c": root / "calibration/T012/camera_c/original.mp4",
    }
    for p in videos.values():
        if not p.exists():
            raise SystemExit(f"missing {p}")

    diag_dir = root / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = diag_dir / "t012_audio_wav"
    wav_dir.mkdir(exist_ok=True)

    print("Extracting mono PCM @", args.sample_rate, "Hz ...")
    pcm: dict[str, np.ndarray] = {}
    durations: dict[str, float] = {}
    for name, path in videos.items():
        x = extract_mono_pcm(path, sample_rate=args.sample_rate)
        pcm[name] = x
        durations[name] = len(x) / args.sample_rate
        # optional save float32 wav-like raw for inspection (npy)
        np.save(wav_dir / f"{name}.npy", x)
        print(f"  {name}: {durations[name]:.3f}s, {len(x)} samples")

    pairs = [
        ("camera_a", "camera_b"),
        ("camera_a", "camera_c"),
        ("camera_b", "camera_c"),
    ]
    pair_reports = []
    for na, nb in pairs:
        print(f"Pair {na}-{nb} ...")
        rep = pairwise_report(
            na, nb, pcm[na], pcm[nb], args.sample_rate, args.max_lag_sec, args.fps
        )
        pair_reports.append(rep)
        print(
            f"  xcorr lag={rep['xcorr']['lag_sec']:.4f}s "
            f"({rep['xcorr']['lag_frames']:.2f}f) score={rep['xcorr']['peak_score_pearson']:.4f} "
            f"stable={rep['stable_across_segments']} clear={rep['clear_peak']}"
        )

    # Offsets relative to camera_a using A-B and A-C xcorr
    ab = next(r for r in pair_reports if r["pair"] == "camera_a-camera_b")
    ac = next(r for r in pair_reports if r["pair"] == "camera_a-camera_c")
    bc = next(r for r in pair_reports if r["pair"] == "camera_b-camera_c")
    offsets_vs_a = {
        "camera_a": 0.0,
        "camera_b": ab["xcorr"]["lag_sec"],
        "camera_c": ac["xcorr"]["lag_sec"],
    }
    # Consistency: lag(A,C) - lag(A,B) should ≈ lag(B,C)
    predicted_bc = offsets_vs_a["camera_c"] - offsets_vs_a["camera_b"]
    measured_bc = bc["xcorr"]["lag_sec"]
    triad_residual = measured_bc - predicted_bc

    timeline = build_common_timeline(offsets_vs_a, durations)

    n_clear = sum(1 for r in pair_reports if r["clear_peak"])
    n_stable = sum(1 for r in pair_reports if r["stable_across_segments"])
    triad_ok = abs(triad_residual) <= (2.0 / args.fps)  # within 2 frames

    stage_pass = n_clear >= 2 and n_stable >= 2 and timeline["overlap_duration_sec"] > 5.0
    need_visual = not stage_pass or abs(triad_residual) > (1.0 / args.fps) or any(
        r["xcorr"]["peak_score_pearson"] < 0.2 for r in pair_reports
    )

    if stage_pass and triad_ok and not need_visual:
        recommendation = "audio_sync_ok_proceed_stage2"
    elif stage_pass:
        recommendation = "audio_usable_but_visual_refine_recommended"
    else:
        recommendation = "audio_weak_use_visual_checkerboard_sync_fallback"

    report = {
        "stage": 1,
        "trial": "T012",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "videos": {k: str(v) for k, v in videos.items()},
        "audio": {
            "sample_rate": args.sample_rate,
            "source": "mono downmix via ffmpeg from AAC 48kHz stereo",
            "max_lag_sec": args.max_lag_sec,
            "fps_assumed": args.fps,
            "pcm_npy_dir": str(wav_dir),
        },
        "durations_sec": {k: round(v, 6) for k, v in durations.items()},
        "reference_camera": "camera_a",
        "offsets_vs_reference": {
            cam: {
                "offset_sec": round(off, 6),
                "offset_frames": round(off * args.fps, 3),
                "meaning": (
                    "camera file t=0 maps to reference timeline t=offset_sec; "
                    "positive => this camera started recording later"
                ),
            }
            for cam, off in offsets_vs_a.items()
        },
        "pairwise": pair_reports,
        "triad_consistency": {
            "predicted_bc_from_ab_ac_sec": round(predicted_bc, 6),
            "measured_bc_sec": round(measured_bc, 6),
            "residual_sec": round(triad_residual, 6),
            "residual_frames": round(triad_residual * args.fps, 3),
            "ok_within_2_frames": triad_ok,
        },
        "common_timeline": timeline,
        "success_criteria": {
            "at_least_two_clear_peaks": n_clear >= 2,
            "n_clear_peaks": n_clear,
            "at_least_two_stable_pairs": n_stable >= 2,
            "n_stable_pairs": n_stable,
            "common_timeline_possible": timeline["overlap_duration_sec"] > 0,
            "overlap_gt_5s": timeline["overlap_duration_sec"] > 5.0,
        },
        "stage_pass": stage_pass,
        "need_visual_refine": need_visual,
        "recommendation": recommendation,
        "note": (
            "Original MP4s were not modified. "
            "Offsets from A-B / A-C xcorr; B-C used for triad check."
        ),
    }

    out = diag_dir / "t012_audio_sync.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nWrote", out)
    print("stage_pass=", stage_pass, "need_visual_refine=", need_visual)
    print("recommendation=", recommendation)
    print("common overlap=", timeline["overlap_duration_sec"], "s")
    print("triad residual=", round(triad_residual, 6), "s")


if __name__ == "__main__":
    main()
