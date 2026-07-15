from __future__ import annotations

from datetime import date

from bike_ai.registry.schema import CameraView, SceneType, VideoRecord


def test_video_record_minimal() -> None:
    record = VideoRecord(
        video_id="CX20260715_A001_START_SIDE_001",
        file_path="sample.mp4",
        athlete_id="A001",
        session_date=date(2026, 7, 15),
        scene=SceneType.standing_start,
        camera_view=CameraView.side,
    )
    assert record.video_id.startswith("CX")
    assert record.consent_scope == "internal_research"

