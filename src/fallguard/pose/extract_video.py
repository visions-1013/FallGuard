from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from fallguard.data.pose_cache import PoseCache, PoseCacheMetadata, save_pose_cache
from fallguard.types import PoseFrame


class PoseExtractor(Protocol):
    def reset(self) -> None: ...

    def extract(
        self, frame: np.ndarray, frame_index: int, timestamp: float
    ) -> PoseFrame | None: ...


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def extract_video_to_cache(
    video_path: Path,
    output_path: Path,
    extractor: PoseExtractor,
    model_name: str,
    model_sha256: str,
) -> PoseCache:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    extractor.reset()
    keypoints: list[np.ndarray] = []
    boxes: list[np.ndarray] = []
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            pose = extractor.extract(frame, frame_index, frame_index / fps)
            if pose is None:
                keypoints.append(np.zeros((17, 3), dtype=np.float32))
                boxes.append(np.zeros(4, dtype=np.float32))
            else:
                keypoints.append(pose.keypoints)
                boxes.append(pose.box if pose.box is not None else np.zeros(4, dtype=np.float32))
            frame_index += 1
    finally:
        capture.release()
    cache = PoseCache(
        keypoints=np.asarray(keypoints, dtype=np.float32),
        boxes=np.asarray(boxes, dtype=np.float32),
        metadata=PoseCacheMetadata(
            source_path=str(video_path.resolve()),
            source_sha256=sha256_file(video_path),
            model_name=model_name,
            model_sha256=model_sha256,
            width=width,
            height=height,
            fps=fps,
            frames=frame_index,
        ),
    )
    save_pose_cache(output_path, cache)
    return cache
