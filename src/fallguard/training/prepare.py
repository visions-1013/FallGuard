from __future__ import annotations

import csv
from pathlib import Path

from fallguard.data.pose_cache import load_pose_cache
from fallguard.data.preprocessing import resample_pose_sequence
from fallguard.data.windows import build_labeled_windows

from .mmaction_data import MmactionSample, export_pose_dataset

DEFAULT_FOLD_SCENES = ["Coffee_room_01", "Coffee_room_02", "Home_01", "Home_02"]


def prepare_mmaction_dataset(
    manifest_csv: Path,
    pose_cache_dir: Path,
    output_path: Path,
    pose_fps: float = 20.0,
    window_frames: int = 32,
    stride: int = 4,
    fold_scenes: list[str] | None = None,
) -> int:
    samples: list[MmactionSample] = []
    with manifest_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row["scene"] not in (fold_scenes or DEFAULT_FOLD_SCENES):
            continue
        cache_path = pose_cache_dir / row["scene"] / f"{row['video_id']}.npz"
        cache = load_pose_cache(cache_path)
        poses = resample_pose_sequence(cache.keypoints, cache.metadata.fps, pose_fps)
        fall_start = float(row["fall_start"]) / float(row["fps"]) if row["fall_start"] else None
        fall_end = float(row["fall_end"]) / float(row["fps"]) if row["fall_end"] else None
        windows = build_labeled_windows(
            poses,
            pose_fps=pose_fps,
            window_frames=window_frames,
            stride=stride,
            fall_start_seconds=fall_start,
            fall_end_seconds=fall_end,
        )
        for window, label, start in zip(
            windows.inputs[:, 0], windows.labels, windows.starts, strict=True
        ):
            samples.append(
                MmactionSample(
                    sample_id=f"{row['scene']}/{row['video_id']}/{int(start)}",
                    scene=row["scene"],
                    keypoints=window,
                    label=int(label),
                    image_shape=(int(row["height"]), int(row["width"])),
                )
            )
    export_pose_dataset(samples, output_path, fold_scenes or DEFAULT_FOLD_SCENES)
    return len(samples)
