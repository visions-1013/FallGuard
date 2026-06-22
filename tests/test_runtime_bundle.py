from __future__ import annotations

import json
from pathlib import Path

import torch

from fallguard.models.runtime import export_runtime_bundle, load_runtime_bundle
from fallguard.models.stgcn import STGCNClassifier


def _mmaction_checkpoint(path: Path) -> None:
    model = STGCNClassifier(num_classes=2)
    state = {f"backbone.{key}": value for key, value in model.backbone.state_dict().items()}
    state["cls_head.fc_cls.weight"] = model.head.weight.detach().clone()
    state["cls_head.fc_cls.bias"] = model.head.bias.detach().clone()
    torch.save({"state_dict": state}, path)


def test_runtime_bundle_round_trip_includes_threshold_and_hash(tmp_path: Path) -> None:
    checkpoint = tmp_path / "mmaction.pth"
    _mmaction_checkpoint(checkpoint)

    export_runtime_bundle(checkpoint, tmp_path / "bundle", fall_threshold=0.65)
    model, metadata = load_runtime_bundle(tmp_path / "bundle", device="cpu")

    assert isinstance(model, STGCNClassifier)
    assert metadata["fall_threshold"] == 0.65
    assert len(metadata["weights_sha256"]) == 64
    assert json.loads((tmp_path / "bundle" / "model_meta.json").read_text())["classes"] == [
        "non_fall",
        "fall",
    ]
