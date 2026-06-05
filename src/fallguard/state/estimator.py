from __future__ import annotations

from fallguard.features.history import HistorySummary
from fallguard.features.pose_features import PoseFeatures


class PoseStateEstimator:
    def __init__(
        self,
        fall_velocity_threshold: float = 120.0,
        fall_angle_velocity_threshold: float = 120.0,
        lying_velocity_threshold: float = 35.0,
    ) -> None:
        self.fall_velocity_threshold = fall_velocity_threshold
        self.fall_angle_velocity_threshold = fall_angle_velocity_threshold
        self.lying_velocity_threshold = lying_velocity_threshold

    def estimate(self, features: PoseFeatures, history: HistorySummary) -> str:
        if not features.valid:
            return "unknown"

        if self._is_fall(features, history):
            return "fall"

        if self._is_lying(features, history):
            return "lying"

        if features.body_angle_degrees < 15 and features.horizontal_score < 0.3:
            return "standing"

        if history.has_motion:
            return "moving"

        if 15 <= features.body_angle_degrees < 55 and features.horizontal_score < 0.6:
            return "sitting"

        return "unknown"

    def _is_fall(self, features: PoseFeatures, history: HistorySummary) -> bool:
        fast_drop = history.center_velocity >= self.fall_velocity_threshold
        fast_angle_change = history.angle_velocity >= self.fall_angle_velocity_threshold
        sustained_low_pose = features.horizontal_score >= 0.7 and history.low_pose_duration >= 0.2
        return sustained_low_pose and (fast_drop or fast_angle_change)

    def _is_lying(self, features: PoseFeatures, history: HistorySummary) -> bool:
        slow_entry = history.center_velocity <= self.lying_velocity_threshold
        return features.horizontal_score >= 0.7 and slow_entry
