from __future__ import annotations

import cv2
import numpy as np

from fallguard.pose.yolo_pose import PoseDetection


COCO_SKELETON = [
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def draw_overlay(
    frame: np.ndarray,
    detection: PoseDetection | None,
    state: str,
    min_confidence: float = 0.35,
) -> np.ndarray:
    annotated = frame.copy()
    if detection is None:
        cv2.putText(annotated, state, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return annotated

    x1, y1, x2, y2 = [int(value) for value in detection.box]
    color = (0, 0, 255) if state == "fall" else (0, 180, 0)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
    cv2.putText(annotated, state, (x1, max(y1 - 8, 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    keypoints = detection.keypoints
    for start, end in COCO_SKELETON:
        if _visible(keypoints, start, min_confidence) and _visible(keypoints, end, min_confidence):
            p1 = tuple(keypoints[start, :2].astype(int))
            p2 = tuple(keypoints[end, :2].astype(int))
            cv2.line(annotated, p1, p2, (255, 180, 0), 2)

    for point in keypoints:
        if point[2] >= min_confidence:
            cv2.circle(annotated, tuple(point[:2].astype(int)), 3, (0, 255, 255), -1)

    return annotated


def _visible(keypoints: np.ndarray, index: int, min_confidence: float) -> bool:
    return index < len(keypoints) and keypoints[index, 2] >= min_confidence
