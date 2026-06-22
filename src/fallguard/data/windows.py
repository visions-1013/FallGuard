from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .preprocessing import make_windows


@dataclass(frozen=True)
class LabeledWindows:
    inputs: np.ndarray
    labels: np.ndarray
    starts: np.ndarray


def build_labeled_windows(
    keypoints: np.ndarray,
    pose_fps: float,
    window_frames: int = 32,
    stride: int = 4,
    fall_start_seconds: float | None = None,
    fall_end_seconds: float | None = None,
) -> LabeledWindows:
    windows, starts = make_windows(keypoints, window_frames, stride)
    if fall_start_seconds is None or fall_end_seconds is None:
        labels = np.zeros(len(starts), dtype=np.int64)
        return LabeledWindows(windows[:, None], labels, starts)
    if fall_end_seconds < fall_start_seconds:
        raise ValueError("fall_end_seconds must not precede fall_start_seconds")
    labels = np.full(len(starts), -1, dtype=np.int64)
    start_times = starts / pose_fps
    end_times = (starts + window_frames - 1) / pose_fps
    positive = (end_times >= fall_start_seconds + 0.4) & (end_times <= fall_end_seconds + 0.4)
    negative = (end_times < fall_start_seconds) | (start_times > fall_end_seconds + 1.0)
    labels[positive] = 1
    labels[negative] = 0
    keep = labels >= 0
    return LabeledWindows(windows[keep, None], labels[keep], starts[keep])
