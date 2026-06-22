from __future__ import annotations

from typing import Any

import numpy as np
import torch

from fallguard.data.preprocessing import pre_normalize_2d
from fallguard.types import FallEvent, FrameResult

from .events import FallEventStateMachine
from .pose_buffer import PoseBuffer


class FallGuardPipeline:
    def __init__(
        self,
        extractor: Any,
        model: torch.nn.Module,
        device: str | None = None,
        pose_fps: float = 20.0,
        window_frames: int = 32,
        window_stride: int = 4,
        fall_threshold: float = 0.5,
        recovery_threshold: float = 0.35,
        trigger_windows: int = 2,
        recovery_seconds: float = 2.0,
        cooldown_seconds: float = 10.0,
        model_version: str = "unknown",
    ) -> None:
        self.extractor = extractor
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.model_version = model_version
        self.buffer = PoseBuffer(pose_fps, window_frames, window_stride)
        self.events = FallEventStateMachine(
            fall_threshold,
            recovery_threshold,
            trigger_windows,
            recovery_seconds,
            cooldown_seconds,
        )
        self._frame_index = 0
        self._last_probability: float | None = None

    def reset(self) -> None:
        self.extractor.reset()
        self.buffer.reset()
        self.events.reset()
        self._frame_index = 0
        self._last_probability = None

    def process(self, frame: np.ndarray, timestamp: float) -> FrameResult:
        pose = self.extractor.extract(frame, self._frame_index, timestamp)
        self.buffer.add(timestamp, None if pose is None else pose.keypoints)
        window = self.buffer.take_window_if_ready()
        completed: tuple[FallEvent, ...] = ()
        if window is not None:
            height, width = frame.shape[:2]
            normalized = pre_normalize_2d(window, width=width, height=height)
            inputs = torch.from_numpy(normalized[None, None]).to(self.device)
            with torch.no_grad():
                logits = self.model(inputs)
                self._last_probability = float(torch.softmax(logits, dim=1)[0, 1].item())
            update = self.events.update(timestamp, self._last_probability)
            completed = update.completed_events
        result = FrameResult(
            frame_index=self._frame_index,
            timestamp=timestamp,
            state=self.events.state,
            fall_probability=self._last_probability,
            pose=pose,
            events=completed,
        )
        self._frame_index += 1
        return result

    def finish(self, timestamp: float) -> tuple[FallEvent, ...]:
        return self.events.finish(timestamp)
