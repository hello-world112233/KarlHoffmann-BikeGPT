from __future__ import annotations

from datetime import date
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class SceneType(str, Enum):
    standing_start = "standing_start"
    rolling_start = "rolling_start"
    sprint = "sprint"
    aero_descent = "aero_descent"
    steady_ride = "steady_ride"
    other = "other"


class CameraView(str, Enum):
    side = "side"
    front = "front"
    rear = "rear"
    front_oblique = "front_oblique"
    rear_oblique = "rear_oblique"
    high_side = "high_side"
    other = "other"


class OcclusionLevel(str, Enum):
    none = "none"
    mild = "mild"
    moderate = "moderate"
    severe = "severe"


class VideoRecord(BaseModel):
    video_id: str = Field(..., description="Unique video id.")
    file_path: Path
    athlete_id: str
    session_date: date
    location: str = "Changxing"
    scene: SceneType
    camera_view: CameraView
    fps: float | None = None
    resolution: str | None = None
    duration_sec: float | None = None
    occlusion_level: OcclusionLevel = OcclusionLevel.none
    lighting: str | None = None
    coach_note: str | None = None
    consent_scope: str = "internal_research"
    tags: list[str] = Field(default_factory=list)

