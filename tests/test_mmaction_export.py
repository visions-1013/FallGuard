from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from fallguard.training.mmaction_data import MmactionSample, export_pose_dataset


def test_export_pose_dataset_writes_annotations_and_scene_folds(tmp_path: Path) -> None:
    samples = [
        MmactionSample(
            sample_id="Coffee_room_01/video (1)/0",
            scene="Coffee_room_01",
            keypoints=np.zeros((32, 17, 3), dtype=np.float32),
            label=1,
            image_shape=(240, 320),
        ),
        MmactionSample(
            sample_id="Home_01/video (1)/0",
            scene="Home_01",
            keypoints=np.zeros((32, 17, 3), dtype=np.float32),
            label=0,
            image_shape=(240, 320),
        ),
    ]
    output = tmp_path / "le2i_pose.pkl"

    export_pose_dataset(samples, output, fold_scenes=["Coffee_room_01", "Home_01"])

    with output.open("rb") as handle:
        payload = pickle.load(handle)
    assert len(payload["annotations"]) == 2
    assert payload["annotations"][0]["keypoint"].shape == (1, 32, 17, 2)
    assert payload["annotations"][0]["keypoint_score"].shape == (1, 32, 17)
    assert payload["split"]["fold_0_val"] == ["Coffee_room_01/video (1)/0"]
    assert payload["split"]["fold_0_train"] == ["Home_01/video (1)/0"]
