from __future__ import annotations

import numpy as np

from fallguard.data.windows import build_labeled_windows


def test_non_fall_sequence_produces_only_negative_windows() -> None:
    poses = np.zeros((80, 17, 3), dtype=np.float32)

    samples = build_labeled_windows(poses, pose_fps=20, window_frames=32, stride=4)

    assert samples.inputs.shape == (13, 1, 32, 17, 3)
    assert set(samples.labels.tolist()) == {0}


def test_fall_sequence_labels_transition_and_excludes_boundary_windows() -> None:
    poses = np.zeros((100, 17, 3), dtype=np.float32)

    samples = build_labeled_windows(
        poses,
        pose_fps=20,
        window_frames=32,
        stride=4,
        fall_start_seconds=2.0,
        fall_end_seconds=3.0,
    )

    assert 1 in samples.labels
    assert 0 in samples.labels
    assert -1 not in samples.labels
    assert len(samples.labels) < 18
