"""Offline athlete selection / tracking utilities.

Step 0 of the bike-AI pipeline: given multi-person pose predictions (e.g. from
Sapiens2 + a generic person detector), link detections into tracks and pick the
single track that corresponds to the athlete on the trainer/bike.
"""

from __future__ import annotations

from .athlete import (
    AthleteSelection,
    Track,
    build_tracks,
    score_tracks,
    select_athlete,
)

__all__ = [
    "AthleteSelection",
    "Track",
    "build_tracks",
    "score_tracks",
    "select_athlete",
]
