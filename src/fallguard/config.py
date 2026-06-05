from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FallGuardConfig:
    pose_model_name: str = "yolo11n-pose.pt"
    min_keypoint_confidence: float = 0.35
    history_frames: int = 12
    low_pose_threshold: float = 0.7
    fall_velocity_threshold: float = 120.0
    fall_angle_velocity_threshold: float = 120.0
    lying_velocity_threshold: float = 35.0
