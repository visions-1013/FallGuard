from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np

from fallguard.types import PoseFrame


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    return cast(np.ndarray, value.detach().cpu().numpy())


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    x1, y1 = np.maximum(left[:2], right[:2])
    x2, y2 = np.minimum(left[2:], right[2:])
    intersection = max(float(x2 - x1), 0.0) * max(float(y2 - y1), 0.0)
    left_area = max(float(left[2] - left[0]), 0.0) * max(float(left[3] - left[1]), 0.0)
    right_area = max(float(right[2] - right[0]), 0.0) * max(float(right[3] - right[1]), 0.0)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def select_primary_person(
    boxes: np.ndarray,
    scores: np.ndarray,
    previous_box: np.ndarray | None = None,
) -> int | None:
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if not len(boxes):
        return None
    if boxes.shape != (len(boxes), 4) or scores.shape != (len(boxes),):
        raise ValueError("boxes and scores have incompatible shapes")
    areas = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0)
    normalized_area = areas / max(float(areas.max()), 1.0)
    ranking = scores + 0.5 * normalized_area
    if previous_box is not None:
        ranking += np.array([2.0 * _iou(box, previous_box) for box in boxes])
    return int(np.argmax(ranking))


class YoloPoseExtractor:
    def __init__(
        self, model: str | Path | Any = "yolo26n-pose.pt", device: str | None = None
    ) -> None:
        if isinstance(model, (str, Path)):
            from ultralytics import YOLO  # type: ignore[attr-defined]

            self.model = YOLO(str(model))
        else:
            self.model = model
        self.device = device
        self.previous_box: np.ndarray | None = None

    def reset(self) -> None:
        self.previous_box = None

    def extract(self, frame: np.ndarray, frame_index: int, timestamp: float) -> PoseFrame | None:
        result = self.model.predict(source=frame, device=self.device, verbose=False, conf=0.25)[0]
        if result.boxes is None or result.keypoints is None:
            return None
        boxes = _to_numpy(result.boxes.xyxy)
        scores = _to_numpy(result.boxes.conf)
        keypoints = _to_numpy(result.keypoints.data)
        index = select_primary_person(boxes, scores, self.previous_box)
        if index is None:
            return None
        self.previous_box = boxes[index].astype(np.float32)
        return PoseFrame(
            frame_index=frame_index,
            timestamp=timestamp,
            keypoints=keypoints[index],
            box=self.previous_box,
        )
