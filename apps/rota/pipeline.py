"""Run the monocular ROTA pipeline on an uploaded job directory."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parents[1]
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))

from bike_geometry import build_optimized_analysis, load_or_create_calibration  # noqa: E402
from metrics import compute_metrics  # noqa: E402

SAPIENS_ROOT = Path(os.environ.get("SAPIENS_ROOT", "/root/autodl-tmp/bike-cloud/repos/sapiens2/sapiens/pose"))
SAPIENS_CKPT_ROOT = Path(
    os.environ.get("SAPIENS_CHECKPOINT_ROOT", "/root/autodl-tmp/bike-cloud/models/sapiens2_host")
)
MB_ROOT = Path(os.environ.get("MOTIONBERT_ROOT", "/root/autodl-tmp/bike-cloud/repos/MotionBERT"))
DEFAULT_CALIBRATION = APP_DIR / "data" / "demo" / "bike_calibration.json"

PIPELINE_FPS = float(os.environ.get("ROTA_PIPELINE_FPS", "10"))
MAX_FRAMES = int(os.environ.get("ROTA_MAX_FRAMES", "120"))


class PipelineError(RuntimeError):
    pass


def _log(job_dir: Path, message: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()} {message}\n"
    with (job_dir / "pipeline.log").open("a", encoding="utf-8") as handle:
        handle.write(line)


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=merged,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise PipelineError(
            "command failed: "
            + " ".join(cmd)
            + "\n"
            + (result.stderr or result.stdout or "")[-4000:]
        )


def _extract_frames(video_path: Path, frames_dir: Path, fps: float, max_frames: int) -> int:
    from bike_ai.ingestion.video import extract_frames

    frames_dir.mkdir(parents=True, exist_ok=True)
    count = extract_frames(video_path, frames_dir, fps=fps)
    if count <= 0:
        raise PipelineError("no frames extracted from video")
    if count > max_frames:
        for path in sorted(frames_dir.glob("frame_*"))[max_frames:]:
            path.unlink()
        count = max_frames
    return count


def _run_sapiens(frames_dir: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    list_path = frames_dir / "image_paths_1.txt"
    paths = sorted(
        p for p in frames_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    list_path.write_text("\n".join(str(p) for p in paths) + "\n", encoding="utf-8")

    model_name = "sapiens2_0.4b"
    dataset = "shutterstock_goliath_3po"
    model = f"{model_name}_keypoints308_{dataset}-1024x768"
    config = SAPIENS_ROOT / f"configs/keypoints308/{dataset}/{model}.py"
    checkpoint = SAPIENS_CKPT_ROOT / "pose" / "sapiens2_0.4b_pose.safetensors"
    detector = SAPIENS_CKPT_ROOT / "detector" / "detr-resnet-101-dc5"
    vis_out = out_dir / model_name
    vis_out.mkdir(parents=True, exist_ok=True)

    for path, label in (
        (config, "sapiens config"),
        (checkpoint, "sapiens checkpoint"),
        (detector, "detector checkpoint"),
        (SAPIENS_ROOT / "tools/vis/vis_pose.py", "vis_pose.py"),
    ):
        if not path.exists():
            raise PipelineError(f"missing {label}: {path}")

    env = {
        "SAPIENS_CHECKPOINT_ROOT": str(SAPIENS_CKPT_ROOT),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
    }
    _run(
        [
            sys.executable,
            "tools/vis/vis_pose.py",
            str(detector),
            str(config),
            str(checkpoint),
            "--input",
            str(list_path),
            "--output",
            str(vis_out),
            "--radius",
            "6",
            "--kpt-thr",
            "0.3",
            "--thickness",
            "8",
        ],
        cwd=SAPIENS_ROOT,
        env=env,
    )
    predictions = vis_out / f"{model_name}_predictions.json"
    if not predictions.exists():
        raise PipelineError(f"sapiens predictions not found: {predictions}")
    return predictions


def _select_athlete(predictions: Path, athlete_dir: Path, fps: float) -> Path:
    athlete_dir.mkdir(parents=True, exist_ok=True)
    script = REPO_ROOT / "scripts" / "select_athlete.py"
    _run(
        [
            sys.executable,
            str(script),
            "--predictions",
            str(predictions),
            "--out-dir",
            str(athlete_dir),
            "--fps",
            str(fps),
        ]
    )
    out = athlete_dir / "athlete_predictions.json"
    if not out.exists():
        raise PipelineError("athlete selection failed")
    return out


def _run_motionbert(athlete_json: Path, out_dir: Path, fps: float) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    script = REPO_ROOT / "scripts" / "motionbert_lift_from_sapiens.py"
    ckpt = MB_ROOT / "checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin"
    config = MB_ROOT / "configs/pose3d/MB_ft_h36m_global_lite.yaml"
    if not script.exists():
        raise PipelineError(f"missing motionbert script: {script}")
    _run(
        [
            sys.executable,
            str(script),
            "--athlete-json",
            str(athlete_json),
            "--out-dir",
            str(out_dir),
            "--mb-root",
            str(MB_ROOT),
            "--ckpt",
            str(ckpt),
            "--config",
            str(config),
            "--fps",
            str(fps),
            "--no-rota-copy",
        ]
    )
    joints = out_dir / "mono3d_joints_motionbert.json"
    if not joints.exists():
        raise PipelineError("motionbert output missing")
    return joints


def _body_tracking_from_athlete(athlete: dict[str, Any]) -> dict[str, Any]:
    from bike_geometry import BODY_POINT_INDEX

    image_size = athlete.get("image_size") or [2160, 3840]
    height, width = float(image_size[0]), float(image_size[1])
    frames = []
    for index, frame in enumerate(athlete.get("frames") or []):
        instances = frame.get("instances") or []
        points: dict[str, list[float]] = {}
        scores: dict[str, float] = {}
        if instances:
            keypoints = instances[0].get("keypoints") or []
            confidences = instances[0].get("keypoint_scores") or []
            for key, joint in BODY_POINT_INDEX.items():
                if joint < len(keypoints) and len(keypoints[joint]) >= 2:
                    points[key] = [
                        float(keypoints[joint][0]) / width,
                        float(keypoints[joint][1]) / height,
                    ]
                    scores[key] = (
                        float(confidences[joint]) if joint < len(confidences) else 1.0
                    )
        frames.append({"frame": index, "points": points, "scores": scores})
    return {"joint_indices": BODY_POINT_INDEX, "frames": frames}


def run_monocular_job(
    job_dir: Path,
    video_path: Path,
    *,
    job_id: str,
    title: str,
    fps: float = PIPELINE_FPS,
    max_frames: int = MAX_FRAMES,
) -> dict[str, Any]:
    """Extract frames → Sapiens2 → athlete → MotionBERT → constrained 3D."""
    job_dir.mkdir(parents=True, exist_ok=True)
    _log(job_dir, f"start job={job_id} video={video_path.name}")

    frames_dir = job_dir / "frames"
    inference_dir = job_dir / "inference" / "sapiens2"
    athlete_dir = job_dir / "athlete"
    motionbert_dir = job_dir / "motionbert"
    calibration_path = job_dir / "bike_calibration.json"

    if not calibration_path.exists() and DEFAULT_CALIBRATION.exists():
        shutil.copy2(DEFAULT_CALIBRATION, calibration_path)

    n_frames = _extract_frames(video_path, frames_dir, fps=fps, max_frames=max_frames)
    _log(job_dir, f"extracted {n_frames} frames @ {fps} fps")

    predictions = _run_sapiens(frames_dir, inference_dir)
    _log(job_dir, f"sapiens done: {predictions}")

    athlete_json = _select_athlete(predictions, athlete_dir, fps=fps)
    _log(job_dir, f"athlete locked: {athlete_json}")

    joints_json = _run_motionbert(athlete_json, motionbert_dir, fps=fps)
    _log(job_dir, f"motionbert done: {joints_json}")

    motionbert = json.loads(joints_json.read_text(encoding="utf-8"))
    athlete = json.loads(athlete_json.read_text(encoding="utf-8"))
    calibration = load_or_create_calibration(calibration_path)

    optimized, report = build_optimized_analysis(
        motionbert,
        athlete,
        calibration,
        body_calibration=None,
        fps=fps,
        steps=180,
    )
    constrained_path = job_dir / "joints_constrained.json"
    constrained_path.write_text(json.dumps(optimized, ensure_ascii=False), encoding="utf-8")
    athlete_path = job_dir / "athlete_2d.json"
    athlete_path.write_text(json.dumps(athlete, ensure_ascii=False), encoding="utf-8")

    frames = optimized.get("frames") or []
    quality = (report.get("quality_after") or {}) if isinstance(report, dict) else {}
    metrics = compute_metrics(frames, fps=fps)
    if quality:
        metrics["geometry_quality"] = quality
        metrics["geometry_quality_before"] = report.get("quality_before")
        metrics["form_index"] = None
        metrics["hip_stability_pct"] = None
        metrics["narrative"] = [
            item for item in metrics.get("narrative", []) if "骨盆稳定性" not in item
        ]
        metrics["narrative"].extend(
            [
                f"约束后二维重投影误差约 {quality.get('reprojection_rmse_px', 0):.1f} px。",
                f"最大骨长波动约 {quality.get('bone_length_cv_max_pct', 0):.1f}%。",
                "上传视频已跑完整单目管线；五车点标定可在网页调整后重算。",
            ]
        )

    video_name = video_path.name
    analysis = {
        "job_id": job_id,
        "status": "ready",
        "mode": "monocular",
        "title": title,
        "source": f"上传视频 · {video_name} · {n_frames} 帧 @ {fps} fps",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fps": fps,
        "n_frames": len(frames),
        "frames": frames,
        "metrics": metrics,
        "coordinate_system": report.get("coordinate_system", "bike-wheelbase"),
        "bike_calibration": calibration,
        "bike_geometry": optimized.get("bike_geometry"),
        "body_calibration": None,
        "body_tracking": _body_tracking_from_athlete(athlete),
        "video_url": f"/job-assets/{job_id}/{video_name}",
        "pipeline": [
            {"id": "ingest", "label": "导入视频", "state": "done"},
            {"id": "pose2d", "label": "2D 关键点检测", "state": "done"},
            {"id": "track", "label": "运动员锁定", "state": "done"},
            {"id": "lift3d", "label": "MotionBERT 三维初值", "state": "done"},
            {"id": "bikefit", "label": "骑行人车联合运动学", "state": "done"},
            {"id": "metrics", "label": "几何质检", "state": "done"},
            {"id": "report", "label": "报告生成", "state": "done"},
        ],
        "disclaimer": (
            "本结果由上传视频自动跑通：Sapiens2 2D → MotionBERT 初值 → 骑行约束 3D。"
            "五车点默认沿用演示初值，请在网页校正。单目相对尺度，不作训练诊断。"
        ),
        "report": report,
    }
    (job_dir / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False), encoding="utf-8")
    _log(job_dir, "pipeline complete")
    return analysis
