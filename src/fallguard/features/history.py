from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from fallguard.features.pose_features import PoseFeatures


@dataclass(frozen=True)
class HistorySummary:
    frame_count: int
    center_drop: float
    center_velocity: float
    angle_velocity: float
    low_pose_duration: float
    is_low_pose: bool
    has_motion: bool


class PoseHistory:
    def __init__(self, max_frames: int = 12, low_pose_threshold: float = 0.7) -> None:
        self.max_frames = max_frames
        self.low_pose_threshold = low_pose_threshold
        self._frames: deque[PoseFeatures] = deque(maxlen=max_frames)

    def add(self, features: PoseFeatures) -> None:
        if features.valid:
            self._frames.append(features)

    def reset(self) -> None:
        self._frames.clear()

    def summary(self) -> HistorySummary:
        if len(self._frames) < 2:
            return HistorySummary(
                frame_count=len(self._frames),
                center_drop=0.0,
                center_velocity=0.0,
                angle_velocity=0.0,
                low_pose_duration=0.0,
                is_low_pose=False,
                has_motion=False,
            )

        first = self._frames[0]
        last = self._frames[-1]
        elapsed = max(last.timestamp - first.timestamp, 1e-6)
        center_drop = max(last.center_y - first.center_y, 0.0)
        center_velocity = center_drop / elapsed
        angle_velocity = abs(last.body_angle_degrees - first.body_angle_degrees) / elapsed
        low_pose_duration = self._low_pose_duration()
        has_motion = center_velocity > 10.0 or angle_velocity > 10.0

        return HistorySummary(
            frame_count=len(self._frames),
            center_drop=center_drop,
            center_velocity=center_velocity,
            angle_velocity=angle_velocity,
            low_pose_duration=low_pose_duration,
            is_low_pose=last.horizontal_score >= self.low_pose_threshold,
            has_motion=has_motion,
        )

    def _low_pose_duration(self) -> float:
        low_frames = [frame for frame in self._frames if frame.horizontal_score >= self.low_pose_threshold]
        if len(low_frames) < 2:
            return 0.0
        return round(max(low_frames[-1].timestamp - low_frames[0].timestamp, 0.0), 10)
