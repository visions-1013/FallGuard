import numpy as np

from fallguard.features.pose_features import PoseFeatures
from fallguard.pipeline import FallGuardPipeline
from fallguard.pose.yolo_pose import PoseDetection


class FakeDetector:
    def detect(self, frame):
        keypoints = np.zeros((17, 3), dtype=float)
        return [PoseDetection(box=(0, 0, 100, 200), confidence=0.9, keypoints=keypoints)]


class FakeExtractor:
    def extract(self, detection, frame_index=0, timestamp=0.0):
        return PoseFeatures(
            valid=True,
            box=detection.box,
            confidence=detection.confidence,
            keypoint_confidence=0.9,
            aspect_ratio=0.5,
            center_x=50,
            center_y=120,
            body_angle_degrees=4,
            horizontal_score=0.1,
            frame_index=frame_index,
            timestamp=timestamp,
        )


class FakeEstimator:
    def __init__(self):
        self.reset_called = False

    def reset(self):
        self.reset_called = True

    def estimate(self, features, history):
        return "standing"


def test_pipeline_resets_history_and_estimator_when_new_source_starts_at_first_frame():
    pipeline = FallGuardPipeline(detector=FakeDetector())
    pipeline.extractor = FakeExtractor()
    pipeline.estimator = FakeEstimator()
    pipeline.history.add(
        PoseFeatures(
            valid=True,
            box=(0, 0, 100, 200),
            confidence=0.9,
            keypoint_confidence=0.9,
            aspect_ratio=0.5,
            center_x=50,
            center_y=220,
            body_angle_degrees=80,
            horizontal_score=0.9,
            frame_index=50,
            timestamp=5.0,
        )
    )

    pipeline.process_frame(np.zeros((10, 10, 3), dtype=np.uint8), frame_index=0, fps=30.0)

    assert pipeline.estimator.reset_called is True
    assert pipeline.history.summary().frame_count == 1
