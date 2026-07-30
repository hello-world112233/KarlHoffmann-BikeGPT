#!/usr/bin/env python3
"""Lightning-bolt dorsal-contour hip finder for side-view cyclists.

Side-view silhouette ≈ lightning bolt: spine → thigh → shank.
Hip = first strong bend where spine meets thigh along the BACK contour.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

NOSE = 0
L_SH, R_SH = 5, 6
L_EL, R_EL = 7, 8
L_WR, R_WR = 9, 10
L_HIP, R_HIP = 11, 12
L_KN, R_KN = 13, 14
L_AN, R_AN = 15, 16

# Priors: torso/limbs EXCEPT hips (often wrong on side-view)
FG_JOINTS = (NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_KN, R_KN, L_AN, R_AN)


def load_frame(pred: dict, frame_idx: int):
    fr = pred["frames"][frame_idx]
    ins = fr["instances"][0]
    return (
        np.asarray(ins["bbox"], dtype=float),
        np.asarray(ins["keypoints"], dtype=float),
        np.asarray(ins["keypoint_scores"], dtype=float),
    )


def facing_right(kp: np.ndarray, sc: np.ndarray, thr: float = 0.3) -> bool:
    sh_xs, wr_xs = [], []
    for s, w in ((L_SH, L_WR), (R_SH, R_WR)):
        if sc[s] > thr and sc[w] > thr:
            sh_xs.append(kp[s, 0])
            wr_xs.append(kp[w, 0])
    if not sh_xs:
        sh = (kp[L_SH] + kp[R_SH]) / 2
        return float(kp[NOSE, 0]) > float(sh[0])
    return float(np.mean(wr_xs)) > float(np.mean(sh_xs))


def _body_hull_mask(
    shape_hw: tuple[int, int],
    kp: np.ndarray,
    sc: np.ndarray,
    origin_xy: tuple[float, float],
    scale: float,
    thr: float = 0.3,
) -> np.ndarray:
    """Dilated convex hull of FG keypoints in crop coordinates."""
    ch, cw = shape_hw
    pts = []
    ox, oy = origin_xy
    for j in FG_JOINTS:
        if sc[j] < thr:
            continue
        pts.append([(kp[j, 0] - ox) * scale, (kp[j, 1] - oy) * scale])
    # Add a synthetic mid-torso point below shoulders (helps cover butt without using bad hips)
    if sc[L_SH] > thr and sc[R_SH] > thr:
        sh = 0.5 * (kp[L_SH] + kp[R_SH])
        kns = [kp[j] for j in (L_KN, R_KN) if sc[j] > thr]
        if kns:
            kn = np.mean(kns, axis=0)
            # points along torso axis at 40% and 55% toward knees, shifted slightly rear
            for t in (0.35, 0.5, 0.6):
                p = sh + t * (kn - sh)
                pts.append([(p[0] - ox) * scale, (p[1] - oy) * scale])
    hull_m = np.zeros((ch, cw), np.uint8)
    if len(pts) >= 3:
        arr = np.asarray(pts, dtype=np.float32)
        hull = cv2.convexHull(arr)
        cv2.fillConvexPoly(hull_m, hull.astype(np.int32), 255)
        dil = max(20, int(0.08 * min(cw, ch)))
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil * 2 + 1, dil * 2 + 1))
        hull_m = cv2.dilate(hull_m, ker)
    else:
        hull_m[:] = 255
    return hull_m


def grabcut_mask(
    img: np.ndarray,
    bbox: np.ndarray,
    kp: np.ndarray,
    sc: np.ndarray,
    *,
    thr: float = 0.3,
    iters: int = 5,
) -> np.ndarray:
    """GrabCut on bbox crop with keypoint priors; intersect with body hull."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    x1i = max(0, int(np.floor(x1)))
    y1i = max(0, int(np.floor(y1)))
    x2i = min(w, int(np.ceil(x2)))
    y2i = min(h, int(np.ceil(y2)))
    bw, bh = x2i - x1i, y2i - y1i
    if bw < 10 or bh < 10:
        raise ValueError(f"degenerate bbox: {bbox}")

    crop = img[y1i:y2i, x1i:x2i]
    scale = 1.0
    max_side = 800
    if max(bw, bh) > max_side:
        scale = max_side / float(max(bw, bh))
        crop_s = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        crop_s = crop
    ch, cw = crop_s.shape[:2]

    hull = _body_hull_mask((ch, cw), kp, sc, (x1i, y1i), scale, thr)

    mask_c = np.full((ch, cw), cv2.GC_BGD, dtype=np.uint8)
    mask_c[hull > 0] = cv2.GC_PR_FGD

    border = max(10, int(0.055 * min(cw, ch)))
    mask_c[:border, :] = cv2.GC_BGD
    mask_c[-border:, :] = cv2.GC_BGD
    mask_c[:, :border] = cv2.GC_BGD
    mask_c[:, -border:] = cv2.GC_BGD

    r = max(8, int(0.03 * min(cw, ch)))
    for j in FG_JOINTS:
        if sc[j] < thr:
            continue
        cx = int(round((kp[j, 0] - x1i) * scale))
        cy = int(round((kp[j, 1] - y1i) * scale))
        if border <= cx < cw - border and border <= cy < ch - border:
            cv2.circle(mask_c, (cx, cy), r, int(cv2.GC_FGD), -1)

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(crop_s, mask_c, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    binary_s = np.where(
        (mask_c == cv2.GC_FGD) | (mask_c == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)
    # Keep inside hull (cuts GrabCut leaks to bike/bg)
    binary_s[hull == 0] = 0
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_s = cv2.morphologyEx(binary_s, cv2.MORPH_OPEN, ker)
    binary_s = cv2.morphologyEx(binary_s, cv2.MORPH_CLOSE, ker)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary_s, 8)
    if num > 1:
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        binary_s = np.where(labels == best, 255, 0).astype(np.uint8)

    if scale != 1.0:
        binary_c = cv2.resize(binary_s, (bw, bh), interpolation=cv2.INTER_NEAREST)
    else:
        binary_c = binary_s

    out = np.zeros((h, w), dtype=np.uint8)
    out[y1i:y2i, x1i:x2i] = binary_c
    return out


def largest_contour(mask: np.ndarray) -> np.ndarray:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        raise RuntimeError("no contours")
    c = max(cnts, key=cv2.contourArea)
    return c.reshape(-1, 2).astype(float)


def shoulder_knee_y(kp: np.ndarray, sc: np.ndarray, thr: float = 0.3):
    sh = [kp[j, 1] for j in (L_SH, R_SH) if sc[j] > thr]
    kn = [kp[j, 1] for j in (L_KN, R_KN) if sc[j] > thr]
    y_sh = float(np.mean(sh)) if sh else float(kp[NOSE, 1])
    y_kn = float(np.mean(kn)) if kn else y_sh + 400
    return y_sh, y_kn


def dorsal_chain(
    contour: np.ndarray,
    kp: np.ndarray,
    sc: np.ndarray,
    faces_right: bool,
    *,
    thr: float = 0.3,
) -> np.ndarray:
    """Walk contour from rear-at-shoulder down to knee along the back side."""
    y_sh, y_kn = shoulder_knee_y(kp, sc, thr)
    n = len(contour)

    # Seed: at shoulder height, rear-most contour point
    band = np.abs(contour[:, 1] - y_sh) < 30
    if band.sum() < 5:
        band = np.abs(contour[:, 1] - y_sh) < 60
    if band.sum() < 3:
        # fall back: topmost quartile of contour near head / shoulders
        y_top = np.percentile(contour[:, 1], 15)
        band = contour[:, 1] <= y_top + 40
    cands = contour[band]
    if len(cands) == 0:
        cands = contour
    if faces_right:
        start = cands[np.argmin(cands[:, 0])]
    else:
        start = cands[np.argmax(cands[:, 0])]
    i0 = int(np.argmin(np.linalg.norm(contour - start, axis=1)))

    def walk(direction: int) -> np.ndarray:
        pts = [contour[i0].copy()]
        i = i0
        for _ in range(n):
            i = (i + direction) % n
            pts.append(contour[i].copy())
            if contour[i, 1] >= y_kn + 25:
                break
            if len(pts) > 80 and contour[i, 1] < y_sh - 40:
                break
        return np.asarray(pts)

    w1, w2 = walk(+1), walk(-1)
    chain = w1 if w1[:, 1].max() >= w2[:, 1].max() else w2
    chain = chain[(chain[:, 1] >= y_sh - 40) & (chain[:, 1] <= y_kn + 30)]

    # Resample by y using rear envelope along the walked chain's x-neighborhood
    # (stabilizes noisy contour). Prefer points matching rear side.
    if len(chain) < 8:
        return chain
    ys = np.linspace(max(chain[:, 1].min(), y_sh - 20), min(chain[:, 1].max(), y_kn + 10), 100)
    samp = []
    for y in ys:
        near = np.abs(chain[:, 1] - y) <= max(4.0, (y_kn - y_sh) / 50)
        if not np.any(near):
            continue
        cand = chain[near]
        if faces_right:
            samp.append(cand[np.argmin(cand[:, 0])])
        else:
            samp.append(cand[np.argmax(cand[:, 0])])
    samp = np.asarray(samp, dtype=float)
    keep = [0]
    for i in range(1, len(samp)):
        if np.linalg.norm(samp[i] - samp[keep[-1]]) >= 2.5:
            keep.append(i)
    return samp[keep]


def turning_angles(chain: np.ndarray) -> np.ndarray:
    n = len(chain)
    ang = np.zeros(n)
    for i in range(1, n - 1):
        v1 = chain[i] - chain[i - 1]
        v2 = chain[i + 1] - chain[i]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        v1, v2 = v1 / n1, v2 / n2
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        dot = np.clip(float(np.dot(v1, v2)), -1.0, 1.0)
        ang[i] = abs(float(np.arctan2(cross, dot)))
    return ang


def _hip_zone_mask(
    pts: np.ndarray,
    kp: np.ndarray,
    sc: np.ndarray,
    faces_right: bool,
    y_sh: float,
    y_kn: float,
) -> np.ndarray:
    """Spatial prior: hip bend is lower torso (shorts/saddle), not arm or knee."""
    span = max(1.0, y_kn - y_sh)
    sh_x = float(np.mean([kp[j, 0] for j in (L_SH, R_SH) if sc[j] > 0.3] or [kp[L_SH, 0]]))
    kn_xs = [kp[j, 0] for j in (L_KN, R_KN) if sc[j] > 0.3]
    kn_x = float(np.mean(kn_xs)) if kn_xs else sh_x + (200 if faces_right else -200)

    # Spine/thigh junction sits mid-low on shoulder→knee (blue shorts / saddle band)
    lo = y_sh + 0.55 * span
    hi = y_kn - 0.16 * span
    valid = (pts[:, 1] >= lo) & (pts[:, 1] <= hi)

    if faces_right:
        x_lo = sh_x - 100
        x_hi = max(sh_x, kn_x) - 20
        valid &= (pts[:, 0] >= x_lo) & (pts[:, 0] <= x_hi + 100)
    else:
        x_hi = sh_x + 100
        x_lo = min(sh_x, kn_x) + 20
        valid &= (pts[:, 0] <= x_hi) & (pts[:, 0] >= x_lo - 100)

    if np.any(valid):
        return valid
    valid = (pts[:, 1] >= y_sh + 0.42 * span) & (pts[:, 1] <= y_kn - 0.12 * span)
    return valid


def butt_tip_on_chain(
    chain: np.ndarray,
    faces_right: bool,
    y_sh: float,
    y_kn: float,
) -> np.ndarray | None:
    """Rearmost local extremum on dorsal chain in the hip band (butt corner)."""
    if len(chain) < 12:
        return None
    span = y_kn - y_sh
    xs = np.convolve(chain[:, 0], np.ones(5) / 5.0, mode="same")
    ys = chain[:, 1]
    best_i, best_x = None, None
    for i in range(5, len(chain) - 5):
        if ys[i] < y_sh + 0.50 * span or ys[i] > y_kn - 0.18 * span:
            continue
        window_l, window_r = xs[i - 3 : i], xs[i + 1 : i + 4]
        if faces_right:
            if xs[i] <= window_l.min() and xs[i] <= window_r.min():
                if best_x is None or xs[i] < best_x:
                    best_x, best_i = float(xs[i]), i
        else:
            if xs[i] >= window_l.max() and xs[i] >= window_r.max():
                if best_x is None or xs[i] > best_x:
                    best_x, best_i = float(xs[i]), i
    if best_i is None:
        return None
    return chain[best_i].copy()


def hip_from_turning(
    chain: np.ndarray,
    kp: np.ndarray,
    sc: np.ndarray,
    faces_right: bool,
    y_sh: float,
    y_kn: float,
    *,
    smooth: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Strongest bend; prefer down→forward (thigh) turn along the back chain."""
    ang = turning_angles(chain)
    if smooth > 1 and len(ang) >= smooth:
        ang_s = np.convolve(ang, np.ones(smooth) / smooth, mode="same")
    else:
        ang_s = ang
    valid = _hip_zone_mask(chain, kp, sc, faces_right, y_sh, y_kn)
    valid[0] = valid[-1] = False

    diffs = np.diff(chain, axis=0)
    heads = np.unwrap(np.arctan2(diffs[:, 1], diffs[:, 0]))
    scores = np.full(len(chain), -1.0)
    for i in range(2, len(chain) - 2):
        if not valid[i]:
            continue
        h_before = float(np.median(heads[max(0, i - 4) : i]))
        h_after = float(np.median(heads[i : min(len(heads), i + 4)]))
        # faces-right: heading pi/2 (down) → 0 (right) => positive (h_before-h_after)
        turn = (h_before - h_after) if faces_right else (h_after - h_before)
        scores[i] = float(ang_s[i]) * (0.35 + max(0.0, turn))

    i = int(np.argmax(scores))
    return chain[i].copy(), ang_s


def hip_from_douglas(
    chain: np.ndarray,
    kp: np.ndarray,
    sc: np.ndarray,
    faces_right: bool,
    y_sh: float,
    y_kn: float,
    *,
    eps_frac: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    """Douglas-Peucker: largest exterior angle in hip zone (butt corner of bolt)."""
    if len(chain) < 3:
        return chain[len(chain) // 2].copy(), chain
    length = float(np.sum(np.linalg.norm(np.diff(chain, axis=0), axis=1)))
    eps = max(4.0, eps_frac * length)
    approx = cv2.approxPolyDP(chain.astype(np.float32).reshape(-1, 1, 2), eps, False)
    verts = approx.reshape(-1, 2).astype(float)
    if len(verts) < 3:
        approx = cv2.approxPolyDP(
            chain.astype(np.float32).reshape(-1, 1, 2), max(3.0, eps * 0.5), False
        )
        verts = approx.reshape(-1, 2).astype(float)

    ang = turning_angles(verts)
    valid = _hip_zone_mask(verts, kp, sc, faces_right, y_sh, y_kn)
    if len(valid):
        valid[0] = False
        if len(valid) > 1:
            valid[-1] = False
    scores = np.where(valid, ang, -1.0)
    if float(np.max(scores)) < 0:
        span = y_kn - y_sh
        valid = (verts[:, 1] > y_sh + 0.48 * span) & (verts[:, 1] < y_kn - 0.14 * span)
        valid[0] = valid[-1] = False
        scores = np.where(valid, ang, -1.0)
    i = int(np.argmax(scores))
    return verts[i].copy(), verts


def pick_leg(kp: np.ndarray, sc: np.ndarray):
    l_c = float(np.mean([sc[L_KN], sc[L_AN]]))
    r_c = float(np.mean([sc[R_KN], sc[R_AN]]))
    if l_c >= r_c:
        return kp[L_SH], kp[L_KN], kp[L_AN]
    return kp[R_SH], kp[R_KN], kp[R_AN]


def render(img, mask, chain, hip, lightning, dp_verts, out_path, zoom_path, title):
    vis = img.copy()
    tint = np.zeros_like(vis)
    tint[:, :] = (0, 180, 255)
    m = mask > 0
    vis[m] = cv2.addWeighted(vis, 0.55, tint, 0.45, 0)[m]

    if len(chain) >= 2:
        cv2.polylines(
            vis,
            [chain.astype(np.int32).reshape(-1, 1, 2)],
            False,
            (255, 255, 0),
            3,
            cv2.LINE_AA,
        )
    if dp_verts is not None and len(dp_verts) >= 2:
        cv2.polylines(
            vis,
            [dp_verts.astype(np.int32).reshape(-1, 1, 2)],
            False,
            (255, 128, 0),
            2,
            cv2.LINE_AA,
        )
        for p in dp_verts:
            cv2.circle(vis, (int(p[0]), int(p[1])), 5, (255, 128, 0), -1, cv2.LINE_AA)

    if len(lightning) >= 2:
        pts = np.array([[int(p[0]), int(p[1])] for p in lightning], np.int32)
        cv2.polylines(vis, [pts.reshape(-1, 1, 2)], False, (0, 255, 0), 3, cv2.LINE_AA)
        for p in pts:
            cv2.circle(vis, tuple(p), 6, (0, 255, 0), -1, cv2.LINE_AA)

    hx, hy = int(round(hip[0])), int(round(hip[1]))
    s = 18
    cv2.line(vis, (hx - s, hy - s), (hx + s, hy + s), (0, 255, 255), 3, cv2.LINE_AA)
    cv2.line(vis, (hx - s, hy + s), (hx + s, hy - s), (0, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(vis, title, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    cv2.putText(
        vis,
        f"hip=({hx},{hy})",
        (30, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)

    if zoom_path is not None:
        xs = [hip[0]] + [p[0] for p in lightning] + list(chain[:: max(1, len(chain)//20), 0])
        ys = [hip[1]] + [p[1] for p in lightning] + list(chain[:: max(1, len(chain)//20), 1])
        cx, cy = int(np.median(xs)), int(np.median(ys))
        rw, rh = 560, 680
        x0, y0 = max(0, cx - rw // 2), max(0, cy - rh // 2)
        x1, y1 = min(vis.shape[1], x0 + rw), min(vis.shape[0], y0 + rh)
        crop = cv2.resize(vis[y0:y1, x0:x1], None, fx=2.0, fy=2.0)
        cv2.imwrite(str(zoom_path), crop)


def process_frame(img_path, bbox, kp, sc, out_dir, frame_idx, *, save_render=True):
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(img_path)

    faces_r = facing_right(kp, sc)
    mask = grabcut_mask(img, bbox, kp, sc)
    contour = largest_contour(mask)
    chain = dorsal_chain(contour, kp, sc, faces_r)
    y_sh, y_kn = shoulder_knee_y(kp, sc)

    hip_turn, _ = hip_from_turning(chain, kp, sc, faces_r, y_sh, y_kn)
    hip_dp, dp_verts = hip_from_douglas(chain, kp, sc, faces_r, y_sh, y_kn)
    butt = butt_tip_on_chain(chain, faces_r, y_sh, y_kn)

    # Fuse: if butt tip exists, snap to nearest bend candidate / average when close
    cands = [("turning", hip_turn), ("douglas", hip_dp)]
    if butt is not None:
        cands.append(("butt_tip", butt))
        # refine butt to nearest strong candidate if within 70px
        near = sorted(
            [("turning", hip_turn), ("douglas", hip_dp)],
            key=lambda t: float(np.linalg.norm(t[1] - butt)),
        )[0]
        if float(np.linalg.norm(near[1] - butt)) < 70:
            hip = 0.5 * (near[1] + butt)
            method = f"avg({near[0]},butt)"
        else:
            # prefer butt tip (silhouette sit-bone corner) when bends disagree
            hip = butt
            method = "butt_tip"
    else:
        dist = float(np.linalg.norm(hip_turn - hip_dp))
        if dist < 55:
            hip = 0.5 * (hip_turn + hip_dp)
            method = "avg(turn,dp)"
        else:
            hip = hip_dp
            method = "douglas"

    sh_leg, kn, an = pick_leg(kp, sc)
    # rear shoulder for lightning start
    sh_pts = [kp[j] for j in (L_SH, R_SH) if sc[j] > 0.3]
    if sh_pts:
        sh = min(sh_pts, key=lambda p: p[0]) if faces_r else max(sh_pts, key=lambda p: p[0])
    else:
        sh = sh_leg

    lightning = [sh, hip, kn, an]
    result = {
        "frame": frame_idx,
        "faces_right": faces_r,
        "hip_turning": hip_turn.tolist(),
        "hip_douglas": hip_dp.tolist(),
        "hip_butt_tip": None if butt is None else butt.tolist(),
        "hip": hip.tolist(),
        "method": method,
        "shoulder": np.asarray(sh).tolist(),
        "knee": np.asarray(kn).tolist(),
        "ankle": np.asarray(an).tolist(),
        "dp_verts": int(len(dp_verts)),
        "sapiens_l_hip": kp[L_HIP].tolist(),
        "sapiens_r_hip": kp[R_HIP].tolist(),
    }
    if save_render:
        render(
            img,
            mask,
            chain,
            hip,
            lightning,
            dp_verts,
            Path(out_dir) / f"qa_lightning_{frame_idx}.jpg",
            Path(out_dir) / f"qa_lightning_{frame_idx}_zoom.jpg",
            f"f{frame_idx} lightning ({method}) faceR={faces_r}",
        )
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pred",
        default="/root/autodl-tmp/bike-ai-data/inference/scene1_dji_static_side_10fps/athlete/athlete_predictions.json",
    )
    ap.add_argument(
        "--frames-dir",
        default="/root/autodl-tmp/bike-ai-data/frames/scene1_dji_static_side_10fps",
    )
    ap.add_argument(
        "--out-dir",
        default="/root/autodl-tmp/bike-ai-data/inference/scene1_dji_static_side_10fps/report",
    )
    ap.add_argument("--frames", nargs="+", type=int, default=[500])
    ap.add_argument("--seat-ref", type=float, nargs=2, default=None)
    args = ap.parse_args()

    with open(args.pred) as f:
        pred = json.load(f)

    results = []
    for fi in args.frames:
        bbox, kp, sc = load_frame(pred, fi)
        img_path = Path(args.frames_dir) / f"frame_{fi:06d}.jpg"
        print(f"\n=== Frame {fi} ===")
        r = process_frame(img_path, bbox, kp, sc, Path(args.out_dir), fi)
        results.append(r)
        print(f"faces_right={r['faces_right']} method={r['method']}")
        print(f"hip_turning  = ({r['hip_turning'][0]:.1f}, {r['hip_turning'][1]:.1f})")
        print(f"hip_douglas  = ({r['hip_douglas'][0]:.1f}, {r['hip_douglas'][1]:.1f})")
        bt = r.get("hip_butt_tip")
        if bt:
            print(f"hip_butt_tip = ({bt[0]:.1f}, {bt[1]:.1f})")
        print(f"hip_final    = ({r['hip'][0]:.1f}, {r['hip'][1]:.1f})")
        print(f"sapiens L/R  = {r['sapiens_l_hip']} / {r['sapiens_r_hip']}")
        print(f"saved {args.out_dir}/qa_lightning_{fi}.jpg")

    hips = np.array([r["hip"] for r in results], float)
    if len(hips) > 1:
        mean, std = hips.mean(0), hips.std(0)
        spreads = np.linalg.norm(hips - mean, axis=1)
        print("\n=== Cluster ===")
        for r in results:
            print(f"  f{r['frame']}: hip=({r['hip'][0]:.1f}, {r['hip'][1]:.1f})")
        print(f"mean=({mean[0]:.1f}, {mean[1]:.1f}) std=({std[0]:.1f}, {std[1]:.1f})")
        print(f"max_dist_to_mean={spreads.max():.1f}px  cluster_tight={bool(spreads.max()<80)}")

    if args.seat_ref is not None:
        ref = np.array(args.seat_ref, float)
        print(f"\n=== vs saddle ref {ref.tolist()} ===")
        for r in results:
            d = float(np.linalg.norm(np.array(r["hip"]) - ref))
            print(f"  f{r['frame']}: dist={d:.1f}px")

    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from bike_ai.kinematics.coach_metrics import estimate_seat_hip
        seat = estimate_seat_hip(pred["frames"])
        print(f"\nestimate_seat_hip = ({seat[0]:.1f}, {seat[1]:.1f})")
        for r in results:
            d = float(np.linalg.norm(np.array(r["hip"]) - seat))
            print(f"  f{r['frame']}: dist={d:.1f}px")
    except Exception as e:
        print("seat est skip:", e)


if __name__ == "__main__":
    main()
