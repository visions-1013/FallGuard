from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees

import numpy as np

from fallguard.pose.yolo_pose import PoseDetection


@dataclass(frozen=True)
class PoseFeatures:
    valid: bool
    box: tuple[float, float, float, float]
    confidence: float
    keypoint_confidence: float
    aspect_ratio: float
    center_x: float
    center_y: float
    body_angle_degrees: float
    horizontal_score: float
    frame_index: int = 0
    timestamp: float = 0.0


class PoseFeatureExtractor:
    def __init__(self, min_keypoint_confidence: float = 0.35) -> None:
        self.min_keypoint_confidence = min_keypoint_confidence

    def extract(
        self,
        detection: PoseDetection,
        frame_index: int = 0,
        timestamp: float = 0.0,
    ) -> PoseFeatures:
        x1, y1, x2, y2 = detection.box
        width = max(x2 - x1, 1.0)
        height = max(y2 - y1, 1.0)
        aspect_ratio = width / height
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        keypoints = detection.keypoints
        visible = keypoints[:, 2] >= self.min_keypoint_confidence
        positive_confidences = keypoints[keypoints[:, 2] > 0, 2] if keypoints.size else np.array([])
        keypoint_confidence = float(np.mean(positive_confidences)) if positive_confidences.size else 0.0
        valid = bool(np.count_nonzero(visible) >= 4 and keypoint_confidence >= self.min_keypoint_confidence)

        shoulder_center = self._mean_point(keypoints, [5, 6])
        hip_center = self._mean_point(keypoints, [11, 12])
        body_angle = self._body_angle(shoulder_center, hip_center)
        horizontal_score = self._horizontal_score(aspect_ratio, body_angle)

        return PoseFeatures(
            valid=valid,
            box=detection.box,
            confidence=detection.confidence,
            keypoint_confidence=keypoint_confidence,
            aspect_ratio=aspect_ratio,
            center_x=center_x,
            center_y=center_y,
            body_angle_degrees=body_angle,
            horizontal_score=horizontal_score,
            frame_index=frame_index,
            timestamp=timestamp,
        )

    @staticmethod
    def _mean_point(keypoints: np.ndarray, indices: list[int]) -> tuple[float, float] | None:
        points = [keypoints[index] for index in indices if index < len(keypoints) and keypoints[index, 2] > 0]
        if not points:
            return None
        arr = np.array(points, dtype=float)
        return float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1]))

    @staticmethod
    def _body_angle(
        shoulder_center: tuple[float, float] | None,
        hip_center: tuple[float, float] | None,
    ) -> float:
        if shoulder_center is None or hip_center is None:
            return 0.0
        dx = hip_center[0] - shoulder_center[0]
        dy = hip_center[1] - shoulder_center[1]
        angle_from_vertical = abs(degrees(atan2(dx, dy)))
        return min(angle_from_vertical, 180.0 - angle_from_vertical)

    @staticmethod
    def _horizontal_score(aspect_ratio: float, body_angle_degrees: float) -> float:
        ratio_score = min(max((aspect_ratio - 0.8) / 1.8, 0.0), 1.0)
        angle_score = min(max(body_angle_degrees / 80.0, 0.0), 1.0)
        return max(ratio_score, angle_score)
