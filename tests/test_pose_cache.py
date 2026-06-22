from __future__ import annotations

from pathlib import Path

import numpy as np

from fallguard.data.pose_cache import PoseCache, PoseCacheMetadata, load_pose_cache, save_pose_cache


def test_pose_cache_round_trip_preserves_metadata(tmp_path: Path) -> None:
    cache = PoseCache(
        keypoints=np.zeros((5, 17, 3), dtype=np.float32),
        boxes=np.ones((5, 4), dtype=np.float32),
        metadata=PoseCacheMetadata(
            source_path="video.avi",
            source_sha256="source-hash",
            model_name="yolo26n-pose.pt",
            model_sha256="model-hash",
            width=320,
            height=240,
            fps=25.0,
            frames=5,
        ),
    )
    output = tmp_path / "pose.npz"

    save_pose_cache(output, cache)
    restored = load_pose_cache(output)

    np.testing.assert_array_equal(restored.keypoints, cache.keypoints)
    np.testing.assert_array_equal(restored.boxes, cache.boxes)
    assert restored.metadata == cache.metadata
