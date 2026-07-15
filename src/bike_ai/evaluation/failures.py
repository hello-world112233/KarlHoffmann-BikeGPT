from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FailureCase:
    video_id: str
    frame_id: str
    failure_type: str
    severity: int
    note: str = ""

