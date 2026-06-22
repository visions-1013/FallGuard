from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PoseFrame:
    frame_index: int
    timestamp: float
    keypoints: np.ndarray
    box: np.ndarray | None = None

    def __post_init__(self) -> None:
        keypoints = np.asarray(self.keypoints, dtype=np.float32)
        if keypoints.shape != (17, 3):
            raise ValueError(f"keypoints must have shape (17, 3), got {keypoints.shape}")
        object.__setattr__(self, "keypoints", keypoints)
        if self.box is not None:
            box = np.asarray(self.box, dtype=np.float32)
            if box.shape != (4,):
                raise ValueError(f"box must have shape (4,), got {box.shape}")
            object.__setattr__(self, "box", box)


@dataclass(frozen=True)
class PoseSequence:
    keypoints: np.ndarray
    fps: float

    def __post_init__(self) -> None:
        keypoints = np.asarray(self.keypoints, dtype=np.float32)
        if keypoints.ndim != 3 or keypoints.shape[1:] != (17, 3):
            raise ValueError(f"keypoints must have shape (T, 17, 3), got {keypoints.shape}")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        object.__setattr__(self, "keypoints", keypoints)

    def as_model_input(self) -> np.ndarray:
        return self.keypoints[None, None, ...]


@dataclass(frozen=True)
class FallPrediction:
    timestamp: float
    probability: float
    label: str


@dataclass
class FallEvent:
    start_time: float
    end_time: float | None = None
    trigger_time: float | None = None
    max_probability: float = 0.0


@dataclass(frozen=True)
class FrameResult:
    frame_index: int
    timestamp: float
    state: str
    fall_probability: float | None
    pose: PoseFrame | None
    events: tuple[FallEvent, ...] = field(default_factory=tuple)
