from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from fallguard.models.stgcn import STGCNClassifier, load_mmaction_backbone


def test_lightweight_backbone_matches_official_mmaction2() -> None:
    checkpoint_value = os.environ.get("FALLGUARD_OFFICIAL_STGCN_CHECKPOINT")
    if not checkpoint_value:
        pytest.skip("set FALLGUARD_OFFICIAL_STGCN_CHECKPOINT for official parity test")
    checkpoint = Path(checkpoint_value)
    if not checkpoint.is_file():
        pytest.fail(f"official checkpoint does not exist: {checkpoint}")

    from mmaction.registry import MODELS
    from mmaction.utils import register_all_modules
    from mmengine.runner.checkpoint import load_checkpoint

    register_all_modules(init_default_scope=True)
    official = MODELS.build(dict(type="STGCN", graph_cfg=dict(layout="coco", mode="stgcn_spatial")))
    load_checkpoint(
        official,
        str(checkpoint),
        map_location="cpu",
        strict=False,
        revise_keys=[(r"^backbone\.", "")],
    )
    local = STGCNClassifier(num_classes=2)
    report = load_mmaction_backbone(local, checkpoint)
    official.eval()
    local.backbone.eval()
    torch.manual_seed(7)
    inputs = torch.randn(2, 1, 32, 17, 3)

    with torch.no_grad():
        official_output = official(inputs)
        local_output = local.backbone(inputs)

    assert report.missing == []
    torch.testing.assert_close(local_output, official_output, rtol=0, atol=0)
