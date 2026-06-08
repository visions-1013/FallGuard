from __future__ import annotations

from fallguard.features.history import HistorySummary
from fallguard.features.pose_features import PoseFeatures


class PoseStateEstimator:
    def __init__(
        self,
        fall_velocity_threshold: float = 120.0,
        fall_angle_velocity_threshold: float = 120.0,
        lying_velocity_threshold: float = 35.0,
        fall_recovery_seconds: float = 1.0,
        low_pose_threshold: float = 0.7,
    ) -> None:
        self.fall_velocity_threshold = fall_velocity_threshold
        self.fall_angle_velocity_threshold = fall_angle_velocity_threshold
        self.lying_velocity_threshold = lying_velocity_threshold
        self.fall_recovery_seconds = fall_recovery_seconds
        self.low_pose_threshold = low_pose_threshold
        self._fall_active = False
        self._recovery_started_at: float | None = None

    def reset(self) -> None:
        self._fall_active = False
        self._recovery_started_at = None

    def estimate(self, features: PoseFeatures, history: HistorySummary) -> str:
        if not features.valid:
            self._recovery_started_at = None
            return "unknown"

        if self._is_fall(features, history):
            self._fall_active = True
            self._recovery_started_at = None
            return "fall"

        base_state = self._base_state(features, history)
        if self._fall_active:
            return self._estimate_active_fall(features, history, base_state)

        return base_state

    def _base_state(self, features: PoseFeatures, history: HistorySummary) -> str:
        if self._is_lying(features, history):
            return "lying"

        if features.body_angle_degrees < 15 and features.horizontal_score < 0.3:
            return "standing"

        if history.has_motion:
            return "moving"

        if 15 <= features.body_angle_degrees < 55 and features.horizontal_score < 0.6:
            return "sitting"

        return "unknown"

    def _estimate_active_fall(
        self,
        features: PoseFeatures,
        history: HistorySummary,
        base_state: str,
    ) -> str:
        if not self._is_recovery_candidate(features, history, base_state):
            self._recovery_started_at = None
            return "fall"

        if self._recovery_started_at is None:
            self._recovery_started_at = features.timestamp
            return "fall"

        if features.timestamp - self._recovery_started_at >= self.fall_recovery_seconds:
            self.reset()
            return base_state

        return "fall"

    def _is_recovery_candidate(
        self,
        features: PoseFeatures,
        history: HistorySummary,
        base_state: str,
    ) -> bool:
        recovered_state = base_state in {"standing", "sitting", "moving"}
        current_low_pose = features.horizontal_score >= self.low_pose_threshold or history.is_low_pose
        return recovered_state and not current_low_pose

    def _is_fall(self, features: PoseFeatures, history: HistorySummary) -> bool:
        fast_drop = history.center_velocity >= self.fall_velocity_threshold
        fast_angle_change = history.angle_velocity >= self.fall_angle_velocity_threshold
        sustained_low_pose = (
            features.horizontal_score >= self.low_pose_threshold and history.low_pose_duration >= 0.2
        )
        return sustained_low_pose and (fast_drop or fast_angle_change)

    def _is_lying(self, features: PoseFeatures, history: HistorySummary) -> bool:
        slow_entry = history.center_velocity <= self.lying_velocity_threshold
        return features.horizontal_score >= self.low_pose_threshold and slow_entry
