import math

import numpy as np

from fallguard.features.pose_features import PoseFeatureExtractor
from fallguard.pose.yolo_pose import PoseDetection


def make_keypoints(points):
    keypoints = np.zeros((17, 3), dtype=float)
    for index, xy in points.items():
        keypoints[index, 0] = xy[0]
        keypoints[index, 1] = xy[1]
        keypoints[index, 2] = 0.95
    return keypoints


def test_extracts_vertical_pose_features():
    keypoints = make_keypoints(
        {
            5: (90, 80),
            6: (110, 80),
            11: (92, 160),
            12: (108, 160),
            15: (92, 240),
            16: (108, 240),
        }
    )
    detection = PoseDetection(box=(80, 50, 120, 260), confidence=0.9, keypoints=keypoints)

    features = PoseFeatureExtractor().extract(detection)

    assert features.valid is True
    assert features.aspect_ratio == 40 / 210
    assert features.center_y == 155
    assert features.body_angle_degrees < 10
    assert features.horizontal_score < 0.3


def test_extracts_horizontal_pose_features():
    keypoints = make_keypoints(
        {
            5: (80, 120),
            6: (80, 140),
            11: (170, 122),
            12: (170, 142),
            15: (240, 126),
            16: (240, 146),
        }
    )
    detection = PoseDetection(box=(70, 110, 250, 160), confidence=0.9, keypoints=keypoints)

    features = PoseFeatureExtractor().extract(detection)

    assert features.valid is True
    assert math.isclose(features.aspect_ratio, 180 / 50)
    assert features.body_angle_degrees > 70
    assert features.horizontal_score > 0.7


def test_marks_low_confidence_pose_invalid():
    keypoints = make_keypoints({5: (90, 80), 6: (110, 80), 11: (92, 160), 12: (108, 160)})
    keypoints[:, 2] = 0.1
    detection = PoseDetection(box=(80, 50, 120, 260), confidence=0.9, keypoints=keypoints)

    features = PoseFeatureExtractor(min_keypoint_confidence=0.4).extract(detection)

    assert features.valid is False
