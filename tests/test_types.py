from __future__ import annotations

import numpy as np
import pytest

from fallguard.types import PoseFrame, PoseSequence


def test_pose_frame_requires_coco17_shape() -> None:
    with pytest.raises(ValueError, match="17, 3"):
        PoseFrame(frame_index=0, timestamp=0.0, keypoints=np.zeros((16, 3)))


def test_pose_sequence_exposes_single_person_model_input() -> None:
    sequence = PoseSequence(keypoints=np.zeros((32, 17, 3)), fps=20.0)

    assert sequence.as_model_input().shape == (1, 1, 32, 17, 3)
