from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field


class InferenceRun(BaseModel):
    run_id: str
    video_id: str
    model_name: str
    model_version: str | None = None
    config_path: Path | None = None
    started_at: datetime = Field(default_factory=datetime.now)
    output_dir: Path
    notes: str | None = None

