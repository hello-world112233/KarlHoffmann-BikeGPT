"""Align multi-camera videos by audio cross-correlation.

Uses ffmpeg to extract mono PCM, then finds the lag that maximizes
cross-correlation against a reference camera. No hardware sync required.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SyncResult:
    """Offsets of each camera relative to the reference camera."""

    reference: str
    sample_rate: int
    offsets_sec: dict[str, float]
    peak_scores: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def _require_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg not found on PATH; install ffmpeg to sync by audio.")
    return exe


def extract_mono_pcm(video_path: str | Path, sample_rate: int = 16000) -> np.ndarray:
    """Extract mono float32 PCM from a video file via ffmpeg."""
    ffmpeg = _require_ffmpeg()
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    cmd = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(video_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg failed for {video_path}:\n{err}")
    if not proc.stdout:
        raise RuntimeError(f"No audio stream extracted from {video_path}")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def _normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    x = x - np.mean(x)
    rms = np.sqrt(np.mean(x * x) + 1e-12)
    return x / rms


def estimate_lag_sec(
    reference: np.ndarray,
    other: np.ndarray,
    sample_rate: int,
    max_lag_sec: float = 30.0,
) -> tuple[float, float]:
    """Return (lag_sec, peak_score).

    Positive lag means ``other`` starts later than ``reference``
    (i.e. trim more from the start of ``reference`` to align).
    """
    a = _normalize(reference)
    b = _normalize(other)
    max_lag = int(max_lag_sec * sample_rate)

    # corr[i] corresponds to lag = i - (len(b) - 1), same as np.correlate(a, b, 'full')
    # Positive lag => a is ahead of b => b (other) starts later.
    corr = np.correlate(a, b, mode="full")
    lags = np.arange(-(len(b) - 1), len(a))
    mask = np.abs(lags) <= max_lag
    lags = lags[mask]
    corr = corr[mask]
    if len(corr) == 0:
        raise RuntimeError("max_lag_sec too small for audio lengths")

    peak_idx = int(np.argmax(corr))
    # np.correlate(a, b) lag sign: when b is a delayed copy of a, peak is at negative lag.
    # Flip so positive => other starts later than reference.
    lag_samples = -int(lags[peak_idx])

    if lag_samples >= 0:
        a_seg = a[lag_samples:]
        b_seg = b[: len(a_seg)]
    else:
        b_seg = b[-lag_samples:]
        a_seg = a[: len(b_seg)]
    m = min(len(a_seg), len(b_seg))
    score = 0.0
    if m > sample_rate:  # at least 1s overlap
        a_seg = _normalize(a_seg[:m])
        b_seg = _normalize(b_seg[:m])
        score = float(np.dot(a_seg, b_seg) / m)
    return lag_samples / float(sample_rate), score


def sync_videos_by_audio(
    videos: dict[str, str | Path],
    reference: str | None = None,
    sample_rate: int = 16000,
    max_lag_sec: float = 30.0,
) -> SyncResult:
    """Sync a named set of videos by audio.

    Parameters
    ----------
    videos:
        Mapping camera_name -> video path. Example:
        ``{"side": "a.mp4", "front": "b.mp4", "rear": "c.mp4"}``
    reference:
        Camera name used as time zero. Defaults to first key.
    """
    if not videos:
        raise ValueError("videos must not be empty")
    names = list(videos.keys())
    ref = reference or names[0]
    if ref not in videos:
        raise KeyError(f"reference camera {ref!r} not in videos")

    pcm: dict[str, np.ndarray] = {}
    for name, path in videos.items():
        pcm[name] = extract_mono_pcm(path, sample_rate=sample_rate)

    offsets = {ref: 0.0}
    scores = {ref: 1.0}
    for name in names:
        if name == ref:
            continue
        lag, score = estimate_lag_sec(
            pcm[ref], pcm[name], sample_rate=sample_rate, max_lag_sec=max_lag_sec
        )
        offsets[name] = lag
        scores[name] = score

    return SyncResult(
        reference=ref,
        sample_rate=sample_rate,
        offsets_sec=offsets,
        peak_scores=scores,
    )


def write_aligned_clips(
    videos: dict[str, str | Path],
    sync: SyncResult,
    out_dir: str | Path,
    duration_sec: float | None = None,
) -> dict[str, Path]:
    """Trim cameras so they share a common start (and optional duration).

    The common start is ``max(offsets)`` so every camera has content.
    """
    ffmpeg = _require_ffmpeg()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_global = max(sync.offsets_sec.values())
    outputs: dict[str, Path] = {}
    for name, path in videos.items():
        # seconds to skip in this file
        ss = start_global - sync.offsets_sec[name]
        out_path = out_dir / f"{name}_aligned.mp4"
        cmd = [ffmpeg, "-y", "-v", "error", "-ss", f"{ss:.4f}", "-i", str(path)]
        if duration_sec is not None:
            cmd += ["-t", f"{duration_sec:.4f}"]
        cmd += ["-c", "copy", str(out_path)]
        proc = subprocess.run(cmd, capture_output=True, check=False)
        if proc.returncode != 0:
            # fallback re-encode if stream copy fails at keyframe
            cmd = [ffmpeg, "-y", "-v", "error", "-ss", f"{ss:.4f}", "-i", str(path)]
            if duration_sec is not None:
                cmd += ["-t", f"{duration_sec:.4f}"]
            cmd += ["-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", str(out_path)]
            proc = subprocess.run(cmd, capture_output=True, check=False)
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr.decode("utf-8", errors="replace"))
        outputs[name] = out_path
    return outputs


def sync_from_session_dir(
    session_dir: str | Path,
    pattern: str = "*.mp4",
    reference: str | None = None,
) -> SyncResult:
    """Convenience: sync all mp4 files in a session folder by stem name."""
    session_dir = Path(session_dir)
    files = sorted(session_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No videos matching {pattern} in {session_dir}")
    videos = {p.stem: p for p in files}
    return sync_videos_by_audio(videos, reference=reference)
