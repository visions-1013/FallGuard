from __future__ import annotations

import numpy as np

from fallguard.data.preprocessing import make_windows, pre_normalize_2d, resample_pose_sequence


def test_pre_normalize_2d_matches_mmaction_formula() -> None:
    keypoints = np.zeros((1, 17, 3), dtype=np.float32)
    keypoints[0, 0] = [0.0, 0.0, 0.2]
    keypoints[0, 1] = [320.0, 240.0, 0.8]

    normalized = pre_normalize_2d(keypoints, width=320, height=240)

    np.testing.assert_allclose(normalized[0, 0], [-1.0, -1.0, 0.2])
    np.testing.assert_allclose(normalized[0, 1], [1.0, 1.0, 0.8])
    np.testing.assert_allclose(keypoints[0, 0], [0.0, 0.0, 0.2])


def test_resample_pose_sequence_uses_time_axis() -> None:
    sequence = np.zeros((3, 17, 3), dtype=np.float32)
    sequence[:, :, 0] = np.array([0.0, 1.0, 2.0])[:, None]

    sampled = resample_pose_sequence(sequence, source_fps=2.0, target_fps=4.0)

    assert sampled.shape == (5, 17, 3)
    np.testing.assert_allclose(sampled[:, 0, 0], [0.0, 0.5, 1.0, 1.5, 2.0])


def test_make_windows_returns_fixed_windows_and_start_indices() -> None:
    sequence = np.zeros((40, 17, 3), dtype=np.float32)

    windows, starts = make_windows(sequence, window_frames=32, stride=4)

    assert windows.shape == (3, 32, 17, 3)
    assert starts.tolist() == [0, 4, 8]
