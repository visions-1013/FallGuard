from __future__ import annotations

from typing import cast

import numpy as np


def pre_normalize_2d(keypoints: np.ndarray, width: int, height: int) -> np.ndarray:
    """Match MMAction2 PreNormalize2D while preserving confidence."""
    if keypoints.shape[-2:] != (17, 3):
        raise ValueError(f"expected (..., 17, 3), got {keypoints.shape}")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    normalized = np.asarray(keypoints, dtype=np.float32).copy()
    normalized[..., 0] = (normalized[..., 0] - width / 2) / (width / 2)
    normalized[..., 1] = (normalized[..., 1] - height / 2) / (height / 2)
    return cast(np.ndarray, normalized)


def resample_pose_sequence(
    keypoints: np.ndarray, source_fps: float, target_fps: float
) -> np.ndarray:
    if keypoints.ndim != 3 or keypoints.shape[1:] != (17, 3):
        raise ValueError(f"expected (T, 17, 3), got {keypoints.shape}")
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("fps must be positive")
    if len(keypoints) <= 1 or source_fps == target_fps:
        return cast(np.ndarray, np.asarray(keypoints, dtype=np.float32).copy())
    duration = (len(keypoints) - 1) / source_fps
    output_frames = int(round(duration * target_fps)) + 1
    source_time = np.arange(len(keypoints), dtype=np.float64) / source_fps
    target_time = np.arange(output_frames, dtype=np.float64) / target_fps
    target_time[-1] = duration
    flat = np.asarray(keypoints, dtype=np.float32).reshape(len(keypoints), -1)
    result = np.empty((output_frames, flat.shape[1]), dtype=np.float32)
    for column in range(flat.shape[1]):
        result[:, column] = np.interp(target_time, source_time, flat[:, column])
    return result.reshape(output_frames, 17, 3)


def make_windows(
    keypoints: np.ndarray, window_frames: int = 32, stride: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    if keypoints.ndim != 3 or keypoints.shape[1:] != (17, 3):
        raise ValueError(f"expected (T, 17, 3), got {keypoints.shape}")
    if window_frames <= 0 or stride <= 0:
        raise ValueError("window_frames and stride must be positive")
    starts = np.arange(0, max(len(keypoints) - window_frames + 1, 0), stride, dtype=np.int64)
    if not len(starts):
        return np.empty((0, window_frames, 17, 3), dtype=np.float32), starts
    windows = np.stack([keypoints[start : start + window_frames] for start in starts])
    return np.asarray(windows, dtype=np.float32), starts
