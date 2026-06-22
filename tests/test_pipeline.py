from __future__ import annotations

import numpy as np
import torch
from torch import nn

from fallguard.inference.pipeline import FallGuardPipeline
from fallguard.types import PoseFrame


class FakeExtractor:
    def reset(self) -> None:
        pass

    def extract(self, frame: np.ndarray, frame_index: int, timestamp: float) -> PoseFrame:
        keypoints = np.zeros((17, 3), dtype=np.float32)
        keypoints[:, :2] = [8, 8]
        keypoints[:, 2] = 1.0
        return PoseFrame(frame_index, timestamp, keypoints, np.array([2, 2, 14, 14]))


class AlwaysFall(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.tensor([[0.0, 5.0]], device=inputs.device).repeat(len(inputs), 1)


def test_pipeline_uses_shared_window_and_triggers_fall() -> None:
    pipeline = FallGuardPipeline(
        extractor=FakeExtractor(),
        model=AlwaysFall(),
        device="cpu",
        pose_fps=20,
        window_frames=4,
        window_stride=1,
        trigger_windows=2,
    )
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    results = [pipeline.process(frame, index / 20) for index in range(5)]

    assert results[-1].state == "fall"
    assert results[-1].fall_probability is not None
    assert results[-1].fall_probability > 0.9
