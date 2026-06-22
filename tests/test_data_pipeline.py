from __future__ import annotations

import csv
import pickle
from pathlib import Path

import cv2
import numpy as np

from fallguard.data.pose_cache import PoseCache, PoseCacheMetadata, save_pose_cache
from fallguard.pose.extract_video import extract_video_to_cache
from fallguard.training.prepare import prepare_mmaction_dataset
from fallguard.types import PoseFrame


class FakeExtractor:
    def reset(self) -> None:
        pass

    def extract(self, frame: np.ndarray, frame_index: int, timestamp: float) -> PoseFrame:
        keypoints = np.zeros((17, 3), dtype=np.float32)
        keypoints[:, 0] = frame_index
        keypoints[:, 2] = 1.0
        return PoseFrame(frame_index, timestamp, keypoints, np.array([0, 0, 8, 8]))


def _write_video(path: Path, frames: int = 3) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (16, 16))
    assert writer.isOpened()
    for index in range(frames):
        writer.write(np.full((16, 16, 3), index, dtype=np.uint8))
    writer.release()


def test_extract_video_to_cache_writes_one_pose_per_frame(tmp_path: Path) -> None:
    video = tmp_path / "input.avi"
    output = tmp_path / "pose.npz"
    _write_video(video)

    cache = extract_video_to_cache(video, output, FakeExtractor(), "fake-pose", "fake-hash")

    assert output.is_file()
    assert cache.keypoints.shape == (3, 17, 3)
    assert cache.metadata.frames == 3
    assert cache.metadata.fps == 25.0


def test_prepare_mmaction_dataset_builds_windows_from_manifest_and_cache(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scene",
                "video_id",
                "video_path",
                "annotation_path",
                "label",
                "fall_start",
                "fall_end",
                "width",
                "height",
                "fps",
                "frames",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "scene": "Coffee_room_01",
                "video_id": "video (1)",
                "video_path": "input.avi",
                "annotation_path": "input.txt",
                "label": "fall",
                "fall_start": 50,
                "fall_end": 75,
                "width": 320,
                "height": 240,
                "fps": 25,
                "frames": 125,
            }
        )
    pose_root = tmp_path / "poses"
    save_pose_cache(
        pose_root / "Coffee_room_01" / "video (1).npz",
        PoseCache(
            keypoints=np.zeros((125, 17, 3), dtype=np.float32),
            boxes=np.zeros((125, 4), dtype=np.float32),
            metadata=PoseCacheMetadata("input.avi", "", "fake", "", 320, 240, 25.0, 125),
        ),
    )
    output = tmp_path / "mmaction.pkl"

    count = prepare_mmaction_dataset(manifest, pose_root, output)

    assert count > 0
    with output.open("rb") as handle:
        payload = pickle.load(handle)
    assert "fold_0_train" in payload["split"]
    assert {item["label"] for item in payload["annotations"]} == {0, 1}
