from __future__ import annotations

import numpy as np

from fallguard.pose.yolo_pose import select_primary_person


def test_select_primary_person_prefers_large_confident_detection() -> None:
    boxes = np.array([[0, 0, 20, 20], [0, 0, 80, 80]], dtype=np.float32)
    scores = np.array([0.95, 0.8], dtype=np.float32)

    assert select_primary_person(boxes, scores) == 1


def test_select_primary_person_keeps_previous_target_by_iou() -> None:
    boxes = np.array([[10, 10, 50, 90], [100, 10, 190, 190]], dtype=np.float32)
    scores = np.array([0.75, 0.9], dtype=np.float32)
    previous = np.array([8, 8, 52, 92], dtype=np.float32)

    assert select_primary_person(boxes, scores, previous_box=previous) == 0


def test_select_primary_person_handles_empty_detections() -> None:
    assert select_primary_person(np.empty((0, 4)), np.empty((0,))) is None
