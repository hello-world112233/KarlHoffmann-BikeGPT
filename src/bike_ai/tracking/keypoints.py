"""Keypoint index constants for the Sapiens2 308-keypoint (goliath) format.

The first 23 keypoints follow the COCO-WholeBody body+foot ordering, which is
what we rely on for the athlete-selection heuristics (center, lower-limb motion).
Face/hand keypoints (indices 23+) are not needed here.
"""

from __future__ import annotations

NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16
LEFT_BIG_TOE = 17
LEFT_SMALL_TOE = 18
LEFT_HEEL = 19
RIGHT_BIG_TOE = 20
RIGHT_SMALL_TOE = 21
RIGHT_HEEL = 22

# Joints whose vertical position oscillates strongly while pedaling.
PEDAL_JOINTS = (LEFT_ANKLE, RIGHT_ANKLE, LEFT_KNEE, RIGHT_KNEE)

# A minimal skeleton (index pairs) for drawing an athlete-only overlay.
SKELETON_LINKS: tuple[tuple[int, int], ...] = (
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE),
    (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE),
    (RIGHT_KNEE, RIGHT_ANKLE),
    (LEFT_ANKLE, LEFT_BIG_TOE),
    (LEFT_ANKLE, LEFT_HEEL),
    (RIGHT_ANKLE, RIGHT_BIG_TOE),
    (RIGHT_ANKLE, RIGHT_HEEL),
)
