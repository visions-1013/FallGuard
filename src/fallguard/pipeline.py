from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fallguard.config import FallGuardConfig
from fallguard.features.history import PoseHistory
from fallguard.features.pose_features import PoseFeatureExtractor
from fallguard.pose.yolo_pose import PoseDetection, PoseDetector
from fallguard.state.estimator import PoseStateEstimator
from fallguard.visualization.draw import draw_overlay


@dataclass(frozen=True)
class PipelineResult:
    annotated_frame: np.ndarray
    state: str
    summary: str
    detection: PoseDetection | None


class FallGuardPipeline:
    def __init__(
        self,
        config: FallGuardConfig | None = None,
        detector: PoseDetector | None = None,
    ) -> None:
        self.config = config or FallGuardConfig()
        self.detector = detector or PoseDetector(self.config.pose_model_name)
        self.extractor = PoseFeatureExtractor(self.config.min_keypoint_confidence)
        self.history = PoseHistory(self.config.history_frames, self.config.low_pose_threshold)
        self.estimator = PoseStateEstimator(
            fall_velocity_threshold=self.config.fall_velocity_threshold,
            fall_angle_velocity_threshold=self.config.fall_angle_velocity_threshold,
            lying_velocity_threshold=self.config.lying_velocity_threshold,
        )

    def process_frame(self, frame: np.ndarray, frame_index: int = 0, fps: float = 30.0) -> PipelineResult:
        detections = self.detector.detect(frame)
        detection = max(detections, key=lambda item: item.confidence, default=None)
        if detection is None:
            annotated = draw_overlay(frame, None, "unknown", self.config.min_keypoint_confidence)
            return PipelineResult(annotated, "unknown", "no person detected", None)

        features = self.extractor.extract(detection, frame_index=frame_index, timestamp=frame_index / fps)
        self.history.add(features)
        history = self.history.summary()
        state = self.estimator.estimate(features, history)
        annotated = draw_overlay(frame, detection, state, self.config.min_keypoint_confidence)
        summary = (
            f"angle={features.body_angle_degrees:.1f}, "
            f"horizontal={features.horizontal_score:.2f}, "
            f"velocity={history.center_velocity:.1f}"
        )
        return PipelineResult(annotated, state, summary, detection)
