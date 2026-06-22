from __future__ import annotations

import pickle
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MmactionSample:
    sample_id: str
    scene: str
    keypoints: np.ndarray
    label: int
    image_shape: tuple[int, int]

    def __post_init__(self) -> None:
        keypoints = np.asarray(self.keypoints, dtype=np.float32)
        if keypoints.ndim != 3 or keypoints.shape[1:] != (17, 3):
            raise ValueError(f"keypoints must have shape (T, 17, 3), got {keypoints.shape}")
        if self.label not in (0, 1):
            raise ValueError("label must be 0 or 1")
        object.__setattr__(self, "keypoints", keypoints)


def export_pose_dataset(
    samples: Iterable[MmactionSample], output_path: Path, fold_scenes: list[str]
) -> None:
    sample_list = list(samples)
    annotations = []
    for sample in sample_list:
        annotations.append(
            {
                "frame_dir": sample.sample_id,
                "label": sample.label,
                "img_shape": sample.image_shape,
                "original_shape": sample.image_shape,
                "total_frames": len(sample.keypoints),
                "keypoint": sample.keypoints[None, ..., :2],
                "keypoint_score": sample.keypoints[None, ..., 2],
                "scene": sample.scene,
            }
        )
    split: dict[str, list[str]] = {}
    for fold, scene in enumerate(fold_scenes):
        split[f"fold_{fold}_val"] = [
            sample.sample_id for sample in sample_list if sample.scene == scene
        ]
        split[f"fold_{fold}_train"] = [
            sample.sample_id for sample in sample_list if sample.scene != scene
        ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump({"split": split, "annotations": annotations}, handle)
