from dataclasses import replace

import pytest

from fallguard.features.history import HistorySummary
from fallguard.features.pose_features import PoseFeatures
from fallguard.state.estimator import PoseStateEstimator


def features(angle=5, horizontal_score=0.1, valid=True, center_y=120, timestamp=0.0):
    return PoseFeatures(
        valid=valid,
        box=(0, 0, 100, 200),
        confidence=0.9,
        keypoint_confidence=0.9,
        aspect_ratio=0.5,
        center_x=50,
        center_y=center_y,
        body_angle_degrees=angle,
        horizontal_score=horizontal_score,
        frame_index=0,
        timestamp=timestamp,
    )


def summary(
    drop=0,
    center_velocity=0,
    angle_velocity=0,
    low_pose_duration=0,
    has_motion=False,
    is_low_pose=None,
):
    return HistorySummary(
        frame_count=4,
        center_drop=drop,
        center_velocity=center_velocity,
        angle_velocity=angle_velocity,
        low_pose_duration=low_pose_duration,
        is_low_pose=low_pose_duration > 0 if is_low_pose is None else is_low_pose,
        has_motion=has_motion,
    )


def test_low_confidence_outputs_unknown():
    state = PoseStateEstimator().estimate(features(valid=False), summary())

    assert state == "unknown"


def test_vertical_pose_outputs_standing():
    state = PoseStateEstimator().estimate(features(angle=4, horizontal_score=0.1), summary())

    assert state == "standing"


def test_sitting_pose_outputs_sitting():
    state = PoseStateEstimator().estimate(features(angle=32, horizontal_score=0.35), summary())

    assert state == "sitting"


def test_normal_motion_outputs_moving():
    state = PoseStateEstimator().estimate(
        features(angle=15, horizontal_score=0.25),
        summary(center_velocity=18, angle_velocity=12, has_motion=True),
    )

    assert state == "moving"


def test_slow_low_pose_outputs_lying():
    state = PoseStateEstimator().estimate(
        features(angle=82, horizontal_score=0.9),
        summary(center_velocity=12, angle_velocity=8, low_pose_duration=0.5),
    )

    assert state == "lying"


def test_fast_drop_to_low_pose_outputs_fall():
    state = PoseStateEstimator().estimate(
        features(angle=84, horizontal_score=0.9),
        summary(center_velocity=180, angle_velocity=220, low_pose_duration=0.4, has_motion=True),
    )

    assert state == "fall"


def trigger_fall(estimator):
    return estimator.estimate(
        features(angle=84, horizontal_score=0.9, timestamp=0.0),
        summary(center_velocity=180, angle_velocity=220, low_pose_duration=0.4, has_motion=True),
    )


def test_confirmed_fall_continues_when_body_remains_low_and_still():
    estimator = PoseStateEstimator()
    assert trigger_fall(estimator) == "fall"

    state = estimator.estimate(
        features(angle=82, horizontal_score=0.9, timestamp=0.5),
        summary(center_velocity=0, angle_velocity=0, low_pose_duration=1.5, is_low_pose=True),
    )

    assert state == "fall"


def test_invalid_frame_reports_unknown_without_clearing_confirmed_fall():
    estimator = PoseStateEstimator()
    assert trigger_fall(estimator) == "fall"

    assert estimator.estimate(features(valid=False, timestamp=0.2), summary()) == "unknown"

    state = estimator.estimate(
        features(angle=80, horizontal_score=0.9, timestamp=0.4),
        summary(center_velocity=0, angle_velocity=0, low_pose_duration=1.0, is_low_pose=True),
    )

    assert state == "fall"


@pytest.mark.parametrize(
    ("recovered_features", "recovered_summary", "expected_state"),
    [
        (features(angle=4, horizontal_score=0.1), summary(), "standing"),
        (features(angle=32, horizontal_score=0.35), summary(), "sitting"),
        (
            features(angle=15, horizontal_score=0.25),
            summary(center_velocity=18, angle_velocity=12, has_motion=True),
            "moving",
        ),
    ],
)
def test_confirmed_fall_clears_after_stable_recovery(
    recovered_features, recovered_summary, expected_state
):
    estimator = PoseStateEstimator(fall_recovery_seconds=1.0)
    assert trigger_fall(estimator) == "fall"

    early_features = replace(recovered_features, timestamp=0.2)
    settled_features = replace(recovered_features, timestamp=1.3)

    assert estimator.estimate(early_features, recovered_summary) == "fall"
    assert estimator.estimate(settled_features, recovered_summary) == expected_state
