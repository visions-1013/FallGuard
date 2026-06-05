from fallguard.features.history import PoseHistory
from fallguard.features.pose_features import PoseFeatures


def make_features(frame_index, center_y, angle, horizontal_score=0.1, valid=True):
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
        frame_index=frame_index,
        timestamp=frame_index / 10,
    )


def test_history_calculates_drop_and_angle_velocity():
    history = PoseHistory(max_frames=5)
    history.add(make_features(0, 100, 5))
    history.add(make_features(1, 112, 15))
    history.add(make_features(2, 135, 50))

    summary = history.summary()

    assert summary.frame_count == 3
    assert summary.center_drop == 35
    assert summary.center_velocity > 100
    assert summary.angle_velocity > 100
    assert summary.low_pose_duration == 0


def test_history_tracks_low_pose_duration():
    history = PoseHistory(max_frames=5, low_pose_threshold=0.7)
    history.add(make_features(0, 100, 10, horizontal_score=0.2))
    history.add(make_features(1, 130, 80, horizontal_score=0.8))
    history.add(make_features(2, 132, 82, horizontal_score=0.9))
    history.add(make_features(3, 134, 83, horizontal_score=0.85))

    summary = history.summary()

    assert summary.low_pose_duration == 0.2
    assert summary.is_low_pose is True


def test_history_ignores_invalid_features():
    history = PoseHistory(max_frames=5)
    history.add(make_features(0, 100, 5, valid=False))

    summary = history.summary()

    assert summary.frame_count == 0
    assert summary.has_motion is False
