from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PoseCacheMetadata:
    source_path: str
    source_sha256: str
    model_name: str
    model_sha256: str
    width: int
    height: int
    fps: float
    frames: int


@dataclass(frozen=True)
class PoseCache:
    keypoints: np.ndarray
    boxes: np.ndarray
    metadata: PoseCacheMetadata

    def __post_init__(self) -> None:
        keypoints = np.asarray(self.keypoints, dtype=np.float32)
        boxes = np.asarray(self.boxes, dtype=np.float32)
        if keypoints.ndim != 3 or keypoints.shape[1:] != (17, 3):
            raise ValueError(f"keypoints must have shape (T, 17, 3), got {keypoints.shape}")
        if boxes.shape != (len(keypoints), 4):
            raise ValueError(f"boxes must have shape (T, 4), got {boxes.shape}")
        object.__setattr__(self, "keypoints", keypoints)
        object.__setattr__(self, "boxes", boxes)


def save_pose_cache(path: Path, cache: PoseCache) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps(asdict(cache.metadata), ensure_ascii=False, sort_keys=True)
    np.savez_compressed(
        path,
        keypoints=cache.keypoints,
        boxes=cache.boxes,
        metadata=np.array(metadata),
    )


def load_pose_cache(path: Path) -> PoseCache:
    with np.load(path, allow_pickle=False) as payload:
        metadata = PoseCacheMetadata(**json.loads(str(payload["metadata"].item())))
        return PoseCache(
            keypoints=payload["keypoints"],
            boxes=payload["boxes"],
            metadata=metadata,
        )
