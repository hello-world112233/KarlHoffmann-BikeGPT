"""Offline athlete selection from multi-person pose predictions.

Pipeline:
1. Parse a Sapiens2-style ``predictions.json`` (per-frame instances).
2. Link instances across frames into tracks via greedy/Hungarian association
   on bbox center distance + IoU (tolerating short gaps).
3. Score each track by persistence, center stability, keypoint confidence and
   pedaling periodicity (FFT of ankle vertical motion in the cadence band).
4. Select the athlete track (the persistent, central, pedaling one).

No re-inference is performed; this is a pure offline post-process so it can be
iterated cheaply and re-run on any predictions file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from .keypoints import LEFT_HIP, PEDAL_JOINTS, RIGHT_HIP

BODY_KEYPOINTS = tuple(range(23))  # body + foot subset used for confidence


@dataclass
class Detection:
    frame_idx: int
    bbox: np.ndarray  # (4,) x1, y1, x2, y2
    keypoints: np.ndarray  # (K, 2)
    scores: np.ndarray  # (K,)

    @property
    def center(self) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return float(abs((x2 - x1) * (y2 - y1)))

    @property
    def height(self) -> float:
        return float(abs(self.bbox[3] - self.bbox[1]))


@dataclass
class Track:
    track_id: int
    detections: dict[int, Detection] = field(default_factory=dict)

    def add(self, det: Detection) -> None:
        self.detections[det.frame_idx] = det

    @property
    def frames(self) -> list[int]:
        return sorted(self.detections)

    @property
    def last_frame(self) -> int:
        return max(self.detections)

    def __len__(self) -> int:
        return len(self.detections)


@dataclass
class TrackScore:
    track_id: int
    coverage: float
    center_norm: tuple[float, float]
    center_std_norm: float
    mean_conf: float
    pedal_ratio: float
    pedal_freq_hz: float
    pedal_amp_norm: float
    pedal_score: float
    total: float


@dataclass
class AthleteSelection:
    track_id: int
    num_frames_total: int
    num_frames_present: int
    scores: list[TrackScore]
    riding_segment: tuple[int, int] | None
    anchor_norm: tuple[float, float]
    cadence_hz: float
    method: str


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def parse_predictions(data: dict[str, Any]) -> tuple[list[list[Detection]], tuple[int, int]]:
    """Return (per-frame detections, image_size=(height, width))."""
    h, w = data["image_size"]
    frames: list[list[Detection]] = []
    for fi, frame in enumerate(data["frames"]):
        dets: list[Detection] = []
        for ins in frame.get("instances", []):
            dets.append(
                Detection(
                    frame_idx=fi,
                    bbox=np.asarray(ins["bbox"], dtype=float),
                    keypoints=np.asarray(ins["keypoints"], dtype=float),
                    scores=np.asarray(ins["keypoint_scores"], dtype=float),
                )
            )
        frames.append(dets)
    return frames, (int(h), int(w))


def build_tracks(
    frames: list[list[Detection]],
    image_size: tuple[int, int],
    *,
    max_center_dist_frac: float = 0.08,
    max_gap: int = 8,
    iou_weight: float = 0.5,
) -> list[Track]:
    """Link per-frame detections into tracks (Hungarian matching + gap memory).

    ``max_center_dist_frac`` gates matches to this fraction of the image
    diagonal; ``max_gap`` lets a track survive short detection dropouts.
    """
    h, w = image_size
    diag = float(np.hypot(w, h))
    gate = max_center_dist_frac * diag

    tracks: list[Track] = []
    active: list[Track] = []  # tracks eligible for matching
    next_id = 0

    for fi, dets in enumerate(frames):
        if not active:
            for det in dets:
                t = Track(track_id=next_id)
                next_id += 1
                t.add(det)
                tracks.append(t)
                active.append(t)
            continue
        if not dets:
            active = [t for t in active if fi - t.last_frame <= max_gap]
            continue

        # Cost matrix between active tracks (last det) and current detections.
        cost = np.full((len(active), len(dets)), 1e6)
        for ti, t in enumerate(active):
            last = t.detections[t.last_frame]
            for di, det in enumerate(dets):
                dist = float(np.linalg.norm(last.center - det.center))
                if dist > gate:
                    continue
                iou = _iou(last.bbox, det.bbox)
                cost[ti, di] = (dist / gate) + iou_weight * (1.0 - iou)

        row, col = linear_sum_assignment(cost)
        matched_det: set[int] = set()
        matched_track: set[int] = set()
        for ti, di in zip(row, col):
            if cost[ti, di] >= 1e5:
                continue
            active[ti].add(dets[di])
            matched_det.add(di)
            matched_track.add(ti)

        for di, det in enumerate(dets):
            if di in matched_det:
                continue
            t = Track(track_id=next_id)
            next_id += 1
            t.add(det)
            tracks.append(t)
            active.append(t)

        active = [t for t in active if fi - t.last_frame <= max_gap]

    return tracks


def _periodicity(series: np.ndarray, fps: float, fmin: float, fmax: float) -> tuple[float, float]:
    """Return (peak_power_ratio, peak_freq_hz) in the cadence band.

    ``series`` may contain NaN (missing frames); the analysis is restricted to
    the span between the first and last valid sample, and gaps within that span
    are linearly filled. The ratio is peak band power over total (non-DC)
    power -> how periodic the signal is.
    """
    y = series.astype(float)
    valid = np.isfinite(y)
    if valid.sum() < 24:
        return 0.0, 0.0
    lo, hi = int(np.argmax(valid)), len(valid) - int(np.argmax(valid[::-1]))
    y = y[lo:hi]
    valid = np.isfinite(y)
    n = len(y)
    if n < 32 or valid.sum() < n * 0.5:
        return 0.0, 0.0
    idx = np.arange(n)
    y = np.interp(idx, idx[valid], y[valid])
    y = y - np.polyval(np.polyfit(idx, y, 1), idx)  # linear detrend
    y = y * np.hanning(n)
    power = np.abs(np.fft.rfft(y)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    band = (freqs >= fmin) & (freqs <= fmax)
    if not band.any():
        return 0.0, 0.0
    total = power[1:].sum()
    if total <= 0:
        return 0.0, 0.0
    bi = np.where(band)[0]
    peak_i = bi[int(np.argmax(power[bi]))]
    return float(power[peak_i] / total), float(freqs[peak_i])


def _pedal_score(
    track: Track,
    num_frames: int,
    fps: float,
    fmin: float,
    fmax: float,
) -> tuple[float, float, float, float]:
    """Return (pedal_score, ratio, freq_hz, amp_norm) from ankle/knee motion."""
    best = (0.0, 0.0, 0.0, 0.0)
    heights = np.array([d.height for d in track.detections.values()])
    med_h = float(np.median(heights)) if len(heights) else 1.0
    med_h = max(med_h, 1.0)
    for j in PEDAL_JOINTS:
        ys = np.full(num_frames, np.nan)
        for fi, det in track.detections.items():
            if det.scores[j] > 0.3:
                ys[fi] = det.keypoints[j, 1]
        ratio, freq = _periodicity(ys, fps, fmin, fmax)
        finite = ys[np.isfinite(ys)]
        amp = float(np.std(finite)) / med_h if len(finite) > 8 else 0.0
        amp_norm = min(1.0, amp / 0.06)  # ~6% of body height saturates
        score = ratio * amp_norm
        if score > best[0]:
            best = (score, ratio, freq, amp)
    return best


def score_tracks(
    tracks: list[Track],
    num_frames: int,
    image_size: tuple[int, int],
    *,
    fps: float = 10.0,
    cadence_hz: tuple[float, float] = (0.5, 2.0),
    weights: tuple[float, float, float, float] = (0.30, 0.15, 0.15, 0.40),
) -> list[TrackScore]:
    """Score every track. ``weights`` = (coverage, stability, conf, pedal)."""
    h, w = image_size
    diag = float(np.hypot(w, h))
    fmin, fmax = cadence_hz
    w_cov, w_stab, w_conf, w_pedal = weights

    scores: list[TrackScore] = []
    for t in tracks:
        centers = np.array([d.center for d in t.detections.values()])
        coverage = len(t) / num_frames
        cx, cy = centers.mean(axis=0)
        center_std = float(np.linalg.norm(centers.std(axis=0)) / diag)
        stability = max(0.0, 1.0 - center_std / 0.15)
        confs = [float(d.scores[list(BODY_KEYPOINTS)].mean()) for d in t.detections.values()]
        mean_conf = float(np.mean(confs)) if confs else 0.0
        pedal_score, ratio, freq, amp = _pedal_score(t, num_frames, fps, fmin, fmax)

        total = (
            w_cov * coverage
            + w_stab * stability
            + w_conf * mean_conf
            + w_pedal * pedal_score
        )
        scores.append(
            TrackScore(
                track_id=t.track_id,
                coverage=coverage,
                center_norm=(float(cx / w), float(cy / h)),
                center_std_norm=center_std,
                mean_conf=mean_conf,
                pedal_ratio=ratio,
                pedal_freq_hz=freq,
                pedal_amp_norm=amp,
                pedal_score=pedal_score,
                total=total,
            )
        )
    scores.sort(key=lambda s: s.total, reverse=True)
    return scores


def _riding_segment(track: Track, num_frames: int, min_run: int = 20) -> tuple[int, int] | None:
    present = np.zeros(num_frames, dtype=bool)
    for fi in track.detections:
        present[fi] = True
    best = None
    start = None
    for fi in range(num_frames):
        if present[fi] and start is None:
            start = fi
        elif not present[fi] and start is not None:
            if best is None or (fi - start) > (best[1] - best[0]):
                best = (start, fi - 1)
            start = None
    if start is not None:
        if best is None or (num_frames - start) > (best[1] - best[0]):
            best = (start, num_frames - 1)
    if best is None or (best[1] - best[0] + 1) < min_run:
        return None
    return best


def _athlete_anchor(
    tracks: list[Track],
    scores: list[TrackScore],
    image_size: tuple[int, int],
    *,
    pedal_thr: float,
    anchor_radius_frac: float,
) -> tuple[np.ndarray, float, str, set[int]]:
    """Find the athlete anchor (pixel center) using the pedaling signal.

    Aggregates the centers of all strongly-pedaling tracks that cluster around
    the top pedaling track; falls back to the best overall track if no pedaling
    is detected (e.g. a non-trainer clip). Returns
    (anchor_xy, cadence_hz, how, athlete_track_ids).
    """
    h, w = image_size
    diag = float(np.hypot(w, h))
    by_id = {t.track_id: t for t in tracks}

    pedaling = [s for s in scores if s.pedal_score >= pedal_thr]
    if pedaling:
        top = max(pedaling, key=lambda s: s.pedal_score)
        top_c = np.array([top.center_norm[0] * w, top.center_norm[1] * h])
        centers, cad, ids = [], [], set()
        for s in pedaling:
            c = np.array([s.center_norm[0] * w, s.center_norm[1] * h])
            if np.linalg.norm(c - top_c) <= anchor_radius_frac * diag:
                t = by_id[s.track_id]
                centers.extend(d.center for d in t.detections.values())
                cad.append(s.pedal_freq_hz)
                ids.add(s.track_id)
        anchor = np.median(np.array(centers), axis=0)
        return anchor, float(np.median(cad)), "pedaling", ids

    best = scores[0]
    return (
        np.array([best.center_norm[0] * w, best.center_norm[1] * h]),
        0.0,
        "fallback_best_track",
        {best.track_id},
    )


def _athlete_template(tracks: list[Track], athlete_ids: set[int]) -> np.ndarray:
    """Per-keypoint median position (K, 2) of the athlete's body over its
    pedaling detections. NaN where never observed confidently."""
    stacks: dict[int, list[np.ndarray]] = {}
    for t in tracks:
        if t.track_id not in athlete_ids:
            continue
        for det in t.detections.values():
            for j in BODY_KEYPOINTS:
                if det.scores[j] > 0.3:
                    stacks.setdefault(j, []).append(det.keypoints[j])
    n_kpt = max(BODY_KEYPOINTS) + 1
    tmpl = np.full((n_kpt, 2), np.nan)
    for j, pts in stacks.items():
        tmpl[j] = np.median(np.asarray(pts), axis=0)
    return tmpl


def _template_cost(det: Detection, tmpl: np.ndarray, diag: float) -> float:
    """Mean normalized distance of a detection's confident body keypoints to
    the athlete template. Low => the body sits where the athlete's does."""
    dists = []
    for j in BODY_KEYPOINTS:
        if det.scores[j] > 0.3 and np.isfinite(tmpl[j, 0]):
            dists.append(float(np.linalg.norm(det.keypoints[j] - tmpl[j])))
    if len(dists) < 6:
        return np.inf
    return float(np.mean(dists) / diag)


def _hip_mid(kpts: np.ndarray, scores: np.ndarray) -> np.ndarray | None:
    if min(scores[LEFT_HIP], scores[RIGHT_HIP]) <= 0.3:
        return None
    return (kpts[LEFT_HIP] + kpts[RIGHT_HIP]) / 2.0


def _athlete_track_from_anchor(
    frames: list[list[Detection]],
    tracks: list[Track],
    athlete_ids: set[int],
    anchor: np.ndarray,
    image_size: tuple[int, int],
    *,
    radius_frac: float,
    template_thr: float,
    hip_tol_frac: float,
) -> Track:
    """Per-frame: pick the detection matching the seated athlete.

    A candidate must (a) lie within ``radius_frac`` of the anchor, (b) have a
    confident hip within ``hip_tol_frac`` of the athlete's template hip (the
    seat position is the single most stable, discriminative landmark for a rider
    on a fixed trainer), and (c) match the body template within ``template_thr``.
    This rejects bent-over coaches / standing bystanders that overlap the anchor
    during setup (they have no confident hip, or their hip is far lower). Frames
    with no valid candidate are left empty rather than mis-assigned.
    """
    h, w = image_size
    diag = float(np.hypot(w, h))
    radius = radius_frac * diag
    hip_tol = hip_tol_frac * diag
    tmpl = _athlete_template(tracks, athlete_ids)
    tmpl_hip = (tmpl[LEFT_HIP] + tmpl[RIGHT_HIP]) / 2.0

    track = Track(track_id=-1)
    for fi, dets in enumerate(frames):
        best_det, best_cost = None, None
        for det in dets:
            if float(np.linalg.norm(det.center - anchor)) > radius:
                continue
            hip = _hip_mid(det.keypoints, det.scores)
            if hip is None or float(np.linalg.norm(hip - tmpl_hip)) > hip_tol:
                continue
            cost = _template_cost(det, tmpl, diag)
            if cost > template_thr:
                continue
            if best_cost is None or cost < best_cost:
                best_cost, best_det = cost, det
        if best_det is not None:
            track.add(best_det)
    return track


def select_athlete(
    data: dict[str, Any],
    *,
    fps: float = 10.0,
    min_coverage: float = 0.3,
    pedal_thr: float = 0.12,
    anchor_radius_frac: float = 0.06,
    select_radius_frac: float = 0.12,
    template_thr: float = 0.06,
    hip_tol_frac: float = 0.06,
    use_anchor: bool = True,
    **track_kwargs: Any,
) -> tuple[AthleteSelection, Track, list[Track]]:
    """Full pipeline: predictions dict -> athlete selection + athlete track.

    Stage 1 links detections into tracks and scores them (persistence, center
    stability, confidence, pedaling). Stage 2 (default) locates the athlete
    anchor from the pedaling signal and re-selects the nearest detection per
    frame, which avoids fragmentation from ID switches on a fixed rig.
    """
    frames, image_size = parse_predictions(data)
    num_frames = len(frames)
    h, w = image_size
    tracks = build_tracks(frames, image_size, **track_kwargs)
    scores = score_tracks(tracks, num_frames, image_size, fps=fps)

    if use_anchor:
        anchor, cadence, method, athlete_ids = _athlete_anchor(
            tracks, scores, image_size,
            pedal_thr=pedal_thr, anchor_radius_frac=anchor_radius_frac,
        )
        athlete_track = _athlete_track_from_anchor(
            frames, tracks, athlete_ids, anchor, image_size,
            radius_frac=select_radius_frac, template_thr=template_thr,
            hip_tol_frac=hip_tol_frac,
        )
        athlete_track.track_id = -1
        anchor_norm = (float(anchor[0] / w), float(anchor[1] / h))
        track_id = -1
    else:
        eligible = [s for s in scores if s.coverage >= min_coverage]
        chosen = (eligible or scores)[0]
        athlete_track = next(t for t in tracks if t.track_id == chosen.track_id)
        anchor_norm = chosen.center_norm
        cadence = chosen.pedal_freq_hz
        method = "best_track"
        track_id = chosen.track_id

    sel = AthleteSelection(
        track_id=track_id,
        num_frames_total=num_frames,
        num_frames_present=len(athlete_track),
        scores=scores,
        riding_segment=_riding_segment(athlete_track, num_frames),
        anchor_norm=anchor_norm,
        cadence_hz=cadence,
        method=method,
    )
    return sel, athlete_track, tracks
