from __future__ import annotations

from collections import deque

import numpy as np


class PoseBuffer:
    def __init__(
        self,
        target_fps: float = 20.0,
        window_frames: int = 32,
        stride: int = 4,
        max_interpolation_frames: int = 3,
        reset_after_seconds: float = 0.5,
    ) -> None:
        self.target_fps = target_fps
        self.window_frames = window_frames
        self.stride = stride
        self.max_interpolation_frames = max_interpolation_frames
        self.reset_after_seconds = reset_after_seconds
        self._frames: deque[np.ndarray] = deque(maxlen=window_frames)
        self._last_pose: np.ndarray | None = None
        self._last_timestamp: float | None = None
        self._next_timestamp: float | None = None
        self._new_since_inference = 0

    @property
    def valid_frames(self) -> int:
        return len(self._frames)

    def reset(self) -> None:
        self._frames.clear()
        self._last_pose = None
        self._last_timestamp = None
        self._next_timestamp = None
        self._new_since_inference = 0

    def _append(self, pose: np.ndarray) -> None:
        self._frames.append(np.asarray(pose, dtype=np.float32).copy())
        self._new_since_inference += 1

    def add(self, timestamp: float, pose: np.ndarray | None) -> None:
        if pose is None:
            if (
                self._last_timestamp is not None
                and timestamp - self._last_timestamp > self.reset_after_seconds
            ):
                self.reset()
            return
        pose = np.asarray(pose, dtype=np.float32)
        if pose.shape != (17, 3):
            raise ValueError(f"pose must have shape (17, 3), got {pose.shape}")
        interval = 1.0 / self.target_fps
        if self._last_pose is None or self._last_timestamp is None:
            self._append(pose)
            self._last_pose = pose
            self._last_timestamp = timestamp
            self._next_timestamp = timestamp + interval
            return
        gap = timestamp - self._last_timestamp
        if gap > self.reset_after_seconds:
            self.reset()
            self.add(timestamp, pose)
            return
        missing_frames = max(0, int(np.floor((gap + 1e-9) / interval)) - 1)
        if missing_frames > self.max_interpolation_frames:
            self.reset()
            self.add(timestamp, pose)
            return
        if self._next_timestamp is None:
            self._next_timestamp = self._last_timestamp + interval
        while self._next_timestamp <= timestamp + 1e-9:
            alpha = min(max((self._next_timestamp - self._last_timestamp) / max(gap, 1e-9), 0), 1)
            interpolated = self._last_pose + (pose - self._last_pose) * alpha
            self._append(interpolated)
            self._next_timestamp += interval
        self._last_pose = pose
        self._last_timestamp = timestamp

    def take_window_if_ready(self) -> np.ndarray | None:
        if len(self._frames) < self.window_frames or self._new_since_inference < self.stride:
            return None
        self._new_since_inference = 0
        return np.stack(self._frames)
