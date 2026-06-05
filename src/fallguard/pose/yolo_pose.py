from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PoseDetection:
    box: tuple[float, float, float, float]
    confidence: float
    keypoints: np.ndarray


class PoseDetector:
    """Thin wrapper around Ultralytics YOLO-Pose."""

    def __init__(self, model_name: str = "yolo11n-pose.pt") -> None:
        config_root = Path.cwd() / ".ultralytics"
        config_root.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(config_root))
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is required for PoseDetector. Install the conda environment first."
            ) from exc

        self._model = YOLO(model_name)

    def detect(self, frame: np.ndarray) -> list[PoseDetection]:
        results = self._model(frame, verbose=False)
        detections: list[PoseDetection] = []
        for result in results:
            if result.boxes is None or result.keypoints is None:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            keypoints_xy = result.keypoints.xy.cpu().numpy()
            keypoints_conf = self._keypoint_confidences(result.keypoints, keypoints_xy)
            for box, confidence, points_xy, points_conf in zip(
                boxes, confidences, keypoints_xy, keypoints_conf
            ):
                points = np.concatenate([points_xy, points_conf[:, None]], axis=1)
                detections.append(
                    PoseDetection(
                        box=tuple(float(value) for value in box),
                        confidence=float(confidence),
                        keypoints=points,
                    )
                )
        return detections

    @staticmethod
    def _keypoint_confidences(keypoints: object, points_xy: np.ndarray) -> Iterable[np.ndarray]:
        conf = getattr(keypoints, "conf", None)
        if conf is None:
            return np.ones(points_xy.shape[:2], dtype=float)
        return conf.cpu().numpy()
