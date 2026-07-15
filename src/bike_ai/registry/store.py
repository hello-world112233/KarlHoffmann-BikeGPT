from __future__ import annotations

import json
from pathlib import Path

from bike_ai.common.config import ensure_dir
from bike_ai.registry.schema import VideoRecord


class JsonRegistry:
    """Simple file-based registry for early field work."""

    def __init__(self, registry_dir: str | Path):
        self.registry_dir = ensure_dir(registry_dir)

    def path_for(self, video_id: str) -> Path:
        return self.registry_dir / f"{video_id}.json"

    def save(self, record: VideoRecord) -> Path:
        path = self.path_for(record.video_id)
        with path.open("w", encoding="utf-8") as f:
            json.dump(record.model_dump(mode="json"), f, ensure_ascii=False, indent=2)
        return path

    def load(self, video_id: str) -> VideoRecord:
        with self.path_for(video_id).open("r", encoding="utf-8") as f:
            return VideoRecord.model_validate(json.load(f))

