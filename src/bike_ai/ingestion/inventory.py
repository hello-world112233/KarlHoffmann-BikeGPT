from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mts", ".m2ts"}


@dataclass(frozen=True)
class VideoQuality:
    sampled_frames: int
    brightness_mean: float | None
    brightness_std: float | None
    blur_laplacian_mean: float | None
    motion_score_mean: float | None


@dataclass(frozen=True)
class VideoInventoryRecord:
    video_id: str
    path: str
    domain: str
    relative_path: str
    size_gb: float
    fps: float
    frame_count: int
    width: int
    height: int
    duration_sec: float
    quality: VideoQuality
    suggested_use: str
    notes: str


def iter_video_files(root: str | Path) -> list[Path]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(root_path)
    return sorted(
        p for p in root_path.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def infer_domain(path: Path) -> str:
    parts = {p.lower() for p in path.parts}
    name = path.name.lower()
    if "trainer_static" in parts or "trainer" in name or "static" in name:
        return "trainer_static"
    if "competition_full" in parts or "competition" in parts or "comp" in name or "race" in name:
        return "competition_full"
    if "field_training" in parts or "training" in parts or "velodrome" in name:
        return "field_training"
    return "unknown"


def make_video_id(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    stem = "_".join(rel.with_suffix("").parts)
    clean = []
    for ch in stem:
        clean.append(ch if ch.isalnum() else "_")
    video_id = "".join(clean).strip("_")
    while "__" in video_id:
        video_id = video_id.replace("__", "_")
    return video_id[:120] or path.stem


def _safe_float(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return float(value)


def probe_basic(path: Path) -> dict[str, float | int]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = _safe_float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration_sec = frame_count / fps if fps > 0 else 0.0
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": duration_sec,
    }


def sample_quality(path: Path, sample_frames: int = 24, resize_width: int = 320) -> VideoQuality:
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return VideoQuality(0, None, None, None, None)

    indices = np.linspace(0, max(total - 1, 0), num=min(sample_frames, total), dtype=int)
    brightness: list[float] = []
    blur: list[float] = []
    motion: list[float] = []
    prev_gray_small: np.ndarray | None = None

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness.append(float(gray.mean()))
        blur.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

        h, w = gray.shape[:2]
        if w > resize_width:
            scale = resize_width / w
            gray_small = cv2.resize(gray, (resize_width, max(1, int(h * scale))))
        else:
            gray_small = gray
        if prev_gray_small is not None and prev_gray_small.shape == gray_small.shape:
            motion.append(float(np.mean(cv2.absdiff(prev_gray_small, gray_small))))
        prev_gray_small = gray_small

    cap.release()

    def mean_or_none(values: list[float]) -> float | None:
        return float(np.mean(values)) if values else None

    def std_or_none(values: list[float]) -> float | None:
        return float(np.std(values)) if values else None

    return VideoQuality(
        sampled_frames=len(brightness),
        brightness_mean=mean_or_none(brightness),
        brightness_std=std_or_none(brightness),
        blur_laplacian_mean=mean_or_none(blur),
        motion_score_mean=mean_or_none(motion),
    )


def suggest_use(record: dict, quality: VideoQuality, domain: str) -> tuple[str, str]:
    width = int(record["width"])
    height = int(record["height"])
    fps = float(record["fps"])
    duration = float(record["duration_sec"])
    blur = quality.blur_laplacian_mean or 0.0

    notes: list[str] = []
    if width < 1280 or height < 720:
        notes.append("low_resolution")
    if fps < 50:
        notes.append("low_fps_for_start")
    if blur and blur < 80:
        notes.append("possible_blur")
    if duration > 180 and domain == "competition_full":
        notes.append("full_video_should_be_indexed_and_clipped")

    if domain == "trainer_static":
        use = "clean_baseline_and_annotation_practice"
    elif domain == "competition_full":
        use = "event_index_then_baseline_hard_cases"
    elif domain == "field_training":
        use = "target_domain_baseline_and_validation"
    else:
        use = "needs_manual_sorting"

    if width >= 1920 and fps >= 50 and blur >= 80:
        use += "+high_value"
    return use, ",".join(notes)


def build_inventory(root: str | Path, sample_frames: int = 24) -> list[VideoInventoryRecord]:
    root_path = Path(root).resolve()
    records: list[VideoInventoryRecord] = []
    for path in iter_video_files(root_path):
        basic = probe_basic(path)
        quality = sample_quality(path, sample_frames=sample_frames)
        domain = infer_domain(path)
        use, notes = suggest_use(basic, quality, domain)
        size_gb = path.stat().st_size / (1024**3)
        records.append(
            VideoInventoryRecord(
                video_id=make_video_id(path, root_path),
                path=str(path),
                domain=domain,
                relative_path=str(path.relative_to(root_path)),
                size_gb=round(size_gb, 3),
                fps=round(float(basic["fps"]), 3),
                frame_count=int(basic["frame_count"]),
                width=int(basic["width"]),
                height=int(basic["height"]),
                duration_sec=round(float(basic["duration_sec"]), 3),
                quality=quality,
                suggested_use=use,
                notes=notes,
            )
        )
    return records


def write_inventory(records: list[VideoInventoryRecord], out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "video_inventory.csv"
    jsonl_path = out / "video_inventory.jsonl"
    md_path = out / "video_inventory_report.md"

    flat_rows = []
    for r in records:
        row = asdict(r)
        quality = row.pop("quality")
        row.update({f"quality_{k}": v for k, v in quality.items()})
        flat_rows.append(row)

    fieldnames = list(flat_rows[0].keys()) if flat_rows else [
        "video_id",
        "path",
        "domain",
        "relative_path",
        "size_gb",
        "fps",
        "frame_count",
        "width",
        "height",
        "duration_sec",
        "suggested_use",
        "notes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)

    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    md_path.write_text(render_markdown_report(records), encoding="utf-8")
    return {"csv": csv_path, "jsonl": jsonl_path, "markdown": md_path}


def render_markdown_report(records: list[VideoInventoryRecord]) -> str:
    total_size = sum(r.size_gb for r in records)
    by_domain: dict[str, list[VideoInventoryRecord]] = {}
    for r in records:
        by_domain.setdefault(r.domain, []).append(r)

    lines = [
        "# Video Inventory Report",
        "",
        f"- Total videos: {len(records)}",
        f"- Total size: {total_size:.2f} GB",
        "",
        "## By Domain",
        "",
    ]
    for domain, items in sorted(by_domain.items()):
        lines.append(f"- `{domain}`: {len(items)} videos, {sum(x.size_gb for x in items):.2f} GB")

    lines.extend(
        [
            "",
            "## Recommended First Batch",
            "",
            "These videos are good candidates for the first baseline or event-index pass.",
            "",
            "| video_id | domain | duration | fps | resolution | suggested_use | notes |",
            "|---|---:|---:|---:|---|---|---|",
        ]
    )
    ranked = sorted(
        records,
        key=lambda r: (
            "high_value" not in r.suggested_use,
            r.domain == "unknown",
            -r.width * r.height,
            -r.fps,
        ),
    )
    for r in ranked[:30]:
        lines.append(
            f"| `{r.video_id}` | {r.domain} | {r.duration_sec:.1f}s | "
            f"{r.fps:.1f} | {r.width}x{r.height} | {r.suggested_use} | {r.notes} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `trainer_static`: controlled data for clean baseline, annotation practice, and tooling.",
            "- `competition_full`: should be indexed into event clips before expensive inference.",
            "- `field_training`: target domain data for validation and coach-facing analysis.",
            "- `unknown`: needs manual sorting before entering the dataset.",
            "",
            "Next step: create event indices for competition full videos, then run baseline on selected clips.",
        ]
    )
    return "\n".join(lines) + "\n"
