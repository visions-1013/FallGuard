from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from fallguard.models.stgcn import STGCNClassifier
from fallguard.training.evaluate import evaluate_pose_dataset
from fallguard.training.mmaction_data import MmactionSample, export_pose_dataset


def test_evaluate_pose_dataset_loads_mmaction_checkpoint_and_writes_report(tmp_path: Path) -> None:
    samples = [
        MmactionSample(
            sample_id=f"scene/sample-{index}",
            scene="scene",
            keypoints=np.zeros((32, 17, 3), dtype=np.float32),
            label=index % 2,
            image_shape=(240, 320),
        )
        for index in range(4)
    ]
    dataset = tmp_path / "data.pkl"
    export_pose_dataset(samples, dataset, fold_scenes=["scene"])
    model = STGCNClassifier(num_classes=2)
    state = {f"backbone.{key}": value for key, value in model.backbone.state_dict().items()}
    state["cls_head.fc_cls.weight"] = model.head.weight.detach().clone()
    state["cls_head.fc_cls.bias"] = model.head.bias.detach().clone()
    checkpoint = tmp_path / "model.pth"
    torch.save({"state_dict": state}, checkpoint)
    history_path = tmp_path / "scalars.json"
    history_path.write_text(
        "\n".join(
            [
                json.dumps({"step": 20, "epoch": 1, "train/loss": 0.8, "train/lr": 0.001}),
                json.dumps(
                    {
                        "step": 0,
                        "binary/precision": 0.5,
                        "binary/recall": 0.5,
                        "binary/f1": 0.5,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    metrics = evaluate_pose_dataset(
        checkpoint,
        dataset,
        "fold_0_val",
        tmp_path / "report",
        device="cpu",
        history_path=history_path,
    )

    assert metrics["samples"] == 4
    assert (tmp_path / "report" / "predictions.csv").is_file()
    assert (tmp_path / "report" / "metrics.json").is_file()
    assert (tmp_path / "report" / "errors.csv").is_file()
    assert (tmp_path / "report" / "scene_metrics.csv").is_file()
    assert (tmp_path / "report" / "scene_metrics.json").is_file()
    assert (tmp_path / "report" / "training_curves.png").is_file()
    assert (tmp_path / "report" / "history.csv").is_file()
