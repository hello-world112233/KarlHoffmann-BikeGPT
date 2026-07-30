#!/usr/bin/env python3
"""Pilot side-view standing-start analysis (single camera, no GPU / no pose model).

Purpose: show what ONE side camera can already extract from a start window:
  - departure time (T_go) from motion persistence
  - upper-body silhouette "rise" (up + forward proxy) via GrabCut
  - preview montage + motion curve + JSON summary

Standalone: depends only on OpenCV + NumPy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def list_frames(d: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return sorted(p for p in d.iterdir() if p.suffix.lower() in exts)


def athlete_roi(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    return gray[int(h * 0.20):int(h * 0.85), int(w * 0.02):int(w * 0.48)]


def motion_series(frames: list[Path]) -> np.ndarray:
    prev = None
    out = []
    for p in frames:
        g = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(athlete_roi(g), (160, 120))
        if prev is None:
            out.append(0.0)
        else:
            out.append(float(np.mean(np.abs(roi.astype(np.float32) - prev))))
        prev = roi
    return np.asarray(out, dtype=np.float32)


def estimate_departure(motion: np.ndarray, fps: float, persist_s: float = 0.6) -> int:
    if len(motion) < 5:
        return 0
    base = float(np.median(motion[:max(5, int(0.5 * fps))]))
    thr = max(base * 2.5, base + 8.0)
    need = max(3, int(persist_s * fps))
    run = 0
    for i, m in enumerate(motion):
        if m >= thr:
            run += 1
            if run >= need:
                return i - need + 1
        else:
            run = 0
    return int(np.argmax(motion))


def dorsal_rise_proxy(bgr: np.ndarray, scale_w: int = 240) -> dict:
    h, w = bgr.shape[:2]
    y0, y1 = int(h * 0.18), int(h * 0.82)
    x0, x1 = int(w * 0.02), int(w * 0.48)
    roi_full = bgr[y0:y1, x0:x1]
    fh, fw = roi_full.shape[:2]
    # downscale ROI so GrabCut is fast; keep scale factor to map back to full-res px
    s = scale_w / fw
    roi = cv2.resize(roi_full, (scale_w, max(1, int(fh * s))))
    rh, rw = roi.shape[:2]
    mask = np.zeros((rh, rw), np.uint8)
    rect = (int(rw * 0.15), int(rh * 0.05), int(rw * 0.55), int(rh * 0.90))
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(roi, mask, rect, bgd, fgd, 2, cv2.GC_INIT_WITH_RECT)
        fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except cv2.error:
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, fg = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        fg = 255 - fg

    ys, xs = np.where(fg > 0)
    if len(xs) < 40:
        return {"ok": False}

    y_cut = np.percentile(ys, 35)
    upper = (ys <= y_cut)
    if upper.sum() < 12:
        return {"ok": False}
    back_x = float(np.percentile(xs[upper], 15))
    top_y = float(np.percentile(ys[upper], 10))

    # vectorized left-edge (min x per row) over upper-torso band
    lo = int(np.percentile(ys, 10))
    hi = int(np.percentile(ys, 55))
    order = np.argsort(ys)
    ys_s, xs_s = ys[order], xs[order]
    row_first = np.searchsorted(ys_s, np.arange(rh), side="left")
    row_last = np.searchsorted(ys_s, np.arange(rh), side="right")
    edge_pts = []
    for yy in range(lo, hi, 2):
        a, b = row_first[yy], row_last[yy]
        if b > a:
            edge_pts.append((float(xs_s[a:b].min()), float(yy)))
    angle = np.nan
    if len(edge_pts) >= 6:
        pts = np.asarray(edge_pts)
        vx = pts[-1, 0] - pts[0, 0]
        vy = pts[-1, 1] - pts[0, 1]
        angle = float(np.degrees(np.arctan2(vy, vx)))

    # map back to full-res coordinates
    inv = 1.0 / s
    return {
        "ok": True,
        "back_x": back_x * inv + x0,
        "top_y": top_y * inv + y0,
        "torso_line_deg": angle,
        "fg_area": int(len(xs) * inv * inv),
    }


def draw_preview(frames, idxs, labels, out: Path) -> None:
    tiles = []
    for i, lab in zip(idxs, labels):
        im = cv2.resize(cv2.imread(str(frames[i])), (640, 360))
        cv2.putText(im, lab, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        tiles.append(im)
    while len(tiles) < 4:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])])
    cv2.imwrite(str(out), grid)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--t0", type=float, default=14.0)
    args = ap.parse_args()

    frames = list_frames(args.frames)
    if not frames:
        raise SystemExit(f"no frames in {args.frames}")
    args.out.mkdir(parents=True, exist_ok=True)

    motion = motion_series(frames)
    dep_i = estimate_departure(motion, args.fps)
    t = args.t0 + np.arange(len(frames)) / args.fps

    sample = sorted(set(
        list(range(0, len(frames), max(1, int(args.fps // 5))))
        + list(range(max(0, dep_i - int(args.fps)), min(len(frames), dep_i + int(2 * args.fps)), 2))
    ))
    series = []
    for i in sample:
        r = dorsal_rise_proxy(cv2.imread(str(frames[i])))
        r["frame_idx"] = i
        r["t"] = float(t[i])
        series.append(r)

    ok = [s for s in series if s.get("ok")]
    rise = {}
    if ok:
        pre = [s for s in ok if s["t"] < t[dep_i]]
        ref_y = float(np.median([s["top_y"] for s in pre])) if pre else ok[0]["top_y"]
        ref_x = float(np.median([s["back_x"] for s in pre])) if pre else ok[0]["back_x"]
        dy = [ref_y - s["top_y"] for s in ok]
        dx = [s["back_x"] - ref_x for s in ok]
        post = [s for s in ok if s["t"] >= t[dep_i]]
        rise = {
            "ref_top_y": ref_y,
            "ref_back_x": ref_x,
            "peak_up_px": float(max(dy)) if dy else None,
            "peak_forward_px_proxy": float(max(dx)) if dx else None,
            "torso_line_pre_deg": float(np.nanmedian([s["torso_line_deg"] for s in pre])) if pre else None,
            "torso_line_post_deg": float(np.nanmedian([s["torso_line_deg"] for s in post[:8]])) if post else None,
            "n_ok": len(ok),
        }

    def clamp(i: int) -> int:
        return max(0, min(len(frames) - 1, i))

    idxs = [clamp(dep_i - int(1.0 * args.fps)), clamp(dep_i),
            clamp(dep_i + int(0.5 * args.fps)), clamp(dep_i + int(1.5 * args.fps))]
    labels = [f"held t={t[idxs[0]]:.1f}s", f"depart~ t={t[idxs[1]]:.1f}s",
              f"+0.5s t={t[idxs[2]]:.1f}s", f"+1.5s t={t[idxs[3]]:.1f}s"]
    draw_preview(frames, idxs, labels, args.out / "start_montage.jpg")

    plot = np.ones((280, 900, 3), np.uint8) * 255
    m = motion
    if m.max() > 0:
        xs = np.linspace(40, 860, len(m)).astype(int)
        ys = (240 - (m / m.max()) * 200).astype(int)
        for a, b in zip(zip(xs[:-1], ys[:-1]), zip(xs[1:], ys[1:])):
            cv2.line(plot, a, b, (40, 40, 200), 2)
        xdep = int(40 + (dep_i / max(1, len(m) - 1)) * 820)
        cv2.line(plot, (xdep, 20), (xdep, 260), (0, 160, 0), 2)
        cv2.putText(plot, f"depart~{t[dep_i]:.2f}s", (xdep + 6, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 120, 0), 2)
    cv2.imwrite(str(args.out / "motion_curve.jpg"), plot)

    for i, lab in zip(idxs, ["held", "depart", "p05", "p15"]):
        cv2.imwrite(str(args.out / f"qa_{lab}.jpg"), cv2.imread(str(frames[i])))

    summary = {
        "video_window": {"t0": args.t0, "fps": args.fps, "n_frames": len(frames)},
        "departure_frame_idx": int(dep_i),
        "departure_t_abs_s": float(t[dep_i]),
        "motion_peak": float(motion.max()),
        "rise_proxy": rise,
        "notes": [
            "Departure from ROI motion persistence; verify against countdown board.",
            "Rise proxy from GrabCut silhouette - pilot only, replace with pose/EMG later.",
            "Side-elevated broadcast cam: angles are projective, use for trends not absolute.",
        ],
    }
    (args.out / "start_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
