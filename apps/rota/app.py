"""ROTA — monocular cycling 3D analysis SaaS (demo + API)."""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from bike_geometry import (
    BODY_POINT_INDEX,
    build_optimized_analysis,
    load_body_calibration,
    load_or_create_calibration,
    save_body_calibration,
    save_calibration,
)
from metrics import compute_metrics
from pipeline import PipelineError, run_monocular_job

APP_DIR = Path(__file__).resolve().parent
DEMO_DIR = APP_DIR / "data" / "demo"
JOBS_DIR = APP_DIR / "data" / "jobs"
DEMO_MOTIONBERT = DEMO_DIR / "joints.json"
DEMO_CONSTRAINED = DEMO_DIR / "joints_constrained.json"
DEMO_ATHLETE_2D = DEMO_DIR / "athlete_2d.json"
DEMO_CALIBRATION = DEMO_DIR / "bike_calibration.json"
DEMO_BODY_CALIBRATION = DEMO_DIR / "body_calibration.json"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="ROTA", version="0.1.0")
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
app.mount("/demo-assets", StaticFiles(directory=DEMO_DIR), name="demo-assets")
app.mount("/job-assets", StaticFiles(directory=JOBS_DIR), name="job-assets")


class BikeCalibrationRequest(BaseModel):
    points: dict[str, list[float]]
    confirmed: bool = True


class BodyCalibrationRequest(BaseModel):
    reference_frame: int
    points: dict[str, list[float]]
    confirmed: bool = True


def _load_body_tracking() -> dict:
    """Expose the 12 useful Sapiens joints as normalized drag-handle defaults."""
    if not DEMO_ATHLETE_2D.exists():
        return {"joint_indices": BODY_POINT_INDEX, "frames": []}
    athlete = json.loads(DEMO_ATHLETE_2D.read_text(encoding="utf-8"))
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


def _load_demo_analysis() -> dict:
    joints_path = DEMO_CONSTRAINED if DEMO_CONSTRAINED.exists() else DEMO_MOTIONBERT
    data = json.loads(joints_path.read_text(encoding="utf-8"))
    frames = data.get("frames") or []
    fps = float((data.get("report") or {}).get("fps") or 10.0)
    metrics = compute_metrics(frames, fps=fps)
    report = data.get("report") or {}
    quality = report.get("quality_after")
    if quality:
        metrics["geometry_quality"] = quality
        metrics["geometry_quality_before"] = report.get("quality_before")
        # These scores were previously computed after root-normalizing every frame;
        # do not present them as validated coaching conclusions in constrained mode.
        metrics["form_index"] = None
        metrics["hip_stability_pct"] = None
        metrics["narrative"] = [
            item for item in metrics.get("narrative", []) if "骨盆稳定性" not in item
        ]
        metrics["narrative"].extend(
            [
                f"约束后二维重投影误差约 {quality['reprojection_rmse_px']:.1f} px。",
                f"最大骨长波动约 {quality['bone_length_cv_max_pct']:.1f}%。",
                "当前结果用于几何质检；车辆与人体标定、多视角验证完成前不输出综合训练评分。",
            ]
        )
    video_url = "/demo-assets/source.mp4" if (DEMO_DIR / "source.mp4").exists() else None
    calibration = load_or_create_calibration(DEMO_CALIBRATION)
    body_calibration = (
        data.get("body_calibration")
        or load_body_calibration(DEMO_BODY_CALIBRATION, n_frames=len(frames))
    )
    return {
        "job_id": "demo-t014",
        "status": "ready",
        "mode": "monocular",
        "title": "演示场次 · 室内滚筒",
        "athlete": "运动员 A",
        "source": "单镜头 · T014 camera_a · 8 秒窗口",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fps": fps,
        "n_frames": len(frames),
        "frames": frames,
        "metrics": metrics,
        "coordinate_system": report.get("coordinate_system", "motionbert-relative"),
        "bike_calibration": data.get("bike_calibration") or calibration,
        "bike_geometry": data.get("bike_geometry"),
        "body_calibration": body_calibration,
        "body_tracking": _load_body_tracking(),
        "video_url": video_url,
        "pipeline": [
            {"id": "ingest", "label": "导入视频", "state": "done"},
            {"id": "pose2d", "label": "2D 关键点检测", "state": "done"},
            {"id": "track", "label": "运动员锁定", "state": "done"},
            {"id": "lift3d", "label": "MotionBERT 三维初值", "state": "done"},
            {
                "id": "bikefit",
                "label": "骑行人车联合运动学",
                "state": "done" if quality else "active",
            },
            {"id": "metrics", "label": "几何质检", "state": "done" if quality else "active"},
            {"id": "report", "label": "报告生成", "state": "done"},
        ],
        "disclaimer": (
            "3D 骑姿按公路车人体模型生成：坐在鞍座上、躯干前倾到把、手在把套、"
            "膝在矢状面内前上方摆动、左右脚踏相位相反。2D 只用来估踏频相位；"
            "右侧机位看不到的左膝/左肘不跟 2D。单目相对尺度，不作训练诊断。"
        ),
    }


@app.get("/", response_class=HTMLResponse)
def landing() -> HTMLResponse:
    return HTMLResponse((APP_DIR / "templates" / "index.html").read_text(encoding="utf-8"))


@app.get("/app", response_class=HTMLResponse)
def workspace() -> HTMLResponse:
    return HTMLResponse((APP_DIR / "templates" / "app.html").read_text(encoding="utf-8"))


@app.get("/api/health")
def health():
    return {"ok": True, "product": "ROTA", "mode": "monocular-saas"}


@app.get("/api/demo")
def demo_analysis():
    return _load_demo_analysis()


@app.post("/api/demo/calibration")
def update_demo_calibration(payload: BikeCalibrationRequest):
    """Save five normalized bike points and rebuild the constrained T014 sequence."""
    try:
        calibration = save_calibration(DEMO_CALIBRATION, payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not DEMO_ATHLETE_2D.exists():
        return {
            "ok": False,
            "error": "missing_demo_2d",
            "message": "演示二维关键点尚未复制到 ROTA data/demo。",
        }
    motionbert = json.loads(DEMO_MOTIONBERT.read_text(encoding="utf-8"))
    athlete = json.loads(DEMO_ATHLETE_2D.read_text(encoding="utf-8"))
    body_calibration = load_body_calibration(
        DEMO_BODY_CALIBRATION, n_frames=len(athlete.get("frames") or [])
    )
    optimized, report = build_optimized_analysis(
        motionbert,
        athlete,
        calibration,
        body_calibration,
        fps=10.0,
        steps=180,
    )
    DEMO_CONSTRAINED.write_text(
        json.dumps(optimized, ensure_ascii=False), encoding="utf-8"
    )
    return {"ok": True, "quality": report, "analysis": _load_demo_analysis()}


@app.post("/api/demo/body-calibration")
def update_demo_body_calibration(payload: BodyCalibrationRequest):
    """Save one marked pose and rebuild the sequence with personal bone lengths."""
    if not DEMO_ATHLETE_2D.exists():
        raise HTTPException(status_code=409, detail="missing demo 2D keypoints")
    athlete = json.loads(DEMO_ATHLETE_2D.read_text(encoding="utf-8"))
    n_frames = len(athlete.get("frames") or [])
    try:
        body_calibration = save_body_calibration(
            DEMO_BODY_CALIBRATION, payload.model_dump(), n_frames=n_frames
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    calibration = load_or_create_calibration(DEMO_CALIBRATION)
    motionbert = json.loads(DEMO_MOTIONBERT.read_text(encoding="utf-8"))
    optimized, report = build_optimized_analysis(
        motionbert,
        athlete,
        calibration,
        body_calibration,
        fps=10.0,
        steps=180,
    )
    DEMO_CONSTRAINED.write_text(
        json.dumps(optimized, ensure_ascii=False), encoding="utf-8"
    )
    return {"ok": True, "quality": report, "analysis": _load_demo_analysis()}


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...)):
    """Upload MP4 and run the monocular pipeline (Sapiens2 → MotionBERT → constraints)."""
    job_id = uuid.uuid4().hex[:10]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    filename = file.filename or "upload.mp4"
    if not filename.lower().endswith((".mp4", ".mov", ".mkv", ".webm")):
        raise HTTPException(status_code=400, detail="请上传 MP4/MOV 视频")
    dest = job_dir / filename
    with dest.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)

    title = f"场次 · {filename}"
    try:
        analysis = run_monocular_job(
            job_dir,
            dest,
            job_id=job_id,
            title=title,
        )
    except PipelineError as exc:
        failed = {
            "job_id": job_id,
            "status": "failed",
            "title": title,
            "source": f"上传视频 · {filename}",
            "error": str(exc),
            "video_url": f"/job-assets/{job_id}/{filename}",
        }
        (job_dir / "analysis.json").write_text(
            json.dumps(failed, ensure_ascii=False), encoding="utf-8"
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"job_id": job_id, "status": analysis.get("status", "ready")}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    if job_id == "demo-t014":
        return _load_demo_analysis()
    path = JOBS_DIR / job_id / "analysis.json"
    if not path.exists():
        return {"job_id": job_id, "status": "missing"}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") == "failed":
        raise HTTPException(status_code=500, detail=data.get("error", "pipeline failed"))
    return data


@app.get("/favicon.ico")
def favicon():
    return FileResponse(APP_DIR / "static" / "favicon.svg")


def main():
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8787, reload=False)


if __name__ == "__main__":
    main()
