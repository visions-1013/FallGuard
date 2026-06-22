from __future__ import annotations

import numpy as np

from fallguard.inference.pose_buffer import PoseBuffer


def _pose(value: float) -> np.ndarray:
    result = np.zeros((17, 3), dtype=np.float32)
    result[:, 0] = value
    result[:, 2] = 1.0
    return result


def test_pose_buffer_resamples_to_target_timeline() -> None:
    buffer = PoseBuffer(target_fps=20, window_frames=4, stride=2)
    for index in range(6):
        buffer.add(index / 25, _pose(float(index)))

    window = buffer.take_window_if_ready()

    assert window is not None
    assert window.shape == (4, 17, 3)


def test_pose_buffer_resets_after_long_missing_period() -> None:
    buffer = PoseBuffer(target_fps=20, window_frames=4, stride=2, reset_after_seconds=0.5)
    buffer.add(0.0, _pose(0))
    buffer.add(0.6, None)

    assert buffer.valid_frames == 0
    assert buffer.take_window_if_ready() is None


def test_pose_buffer_does_not_interpolate_more_than_three_missing_frames() -> None:
    buffer = PoseBuffer(
        target_fps=20,
        window_frames=8,
        stride=2,
        max_interpolation_frames=3,
        reset_after_seconds=0.5,
    )
    buffer.add(0.0, _pose(0))
    buffer.add(0.25, _pose(5))

    assert buffer.valid_frames == 1
    assert buffer.take_window_if_ready() is None
