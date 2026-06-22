from __future__ import annotations

import torch

from fallguard.models.graph import CocoGraph
from fallguard.models.stgcn import (
    STGCNClassifier,
    load_mmaction_backbone,
    load_mmaction_classifier,
)


def test_coco_graph_matches_stgcn_spatial_shape() -> None:
    graph = CocoGraph()

    assert graph.A.shape == (3, 17, 17)
    assert graph.center == 0
    assert torch.isfinite(torch.from_numpy(graph.A)).all()


def test_stgcn_returns_two_logits_for_single_person_pose() -> None:
    model = STGCNClassifier(num_classes=2)
    inputs = torch.randn(2, 1, 32, 17, 3)

    logits = model(inputs)

    assert logits.shape == (2, 2)
    assert logits.requires_grad


def test_load_mmaction_backbone_ignores_sixty_class_head() -> None:
    source = STGCNClassifier(num_classes=60)
    target = STGCNClassifier(num_classes=2)
    first_key = next(iter(source.backbone.state_dict()))
    source_state = {
        f"backbone.{key}": value.clone() for key, value in source.backbone.state_dict().items()
    }
    source_state["cls_head.fc_cls.weight"] = source.head.weight.detach().clone()
    source_state["cls_head.fc_cls.bias"] = source.head.bias.detach().clone()

    report = load_mmaction_backbone(target, {"state_dict": source_state})

    assert report.loaded == len(source.backbone.state_dict())
    assert report.skipped == ["cls_head.fc_cls.bias", "cls_head.fc_cls.weight"]
    torch.testing.assert_close(
        target.backbone.state_dict()[first_key], source.backbone.state_dict()[first_key]
    )


def test_load_mmaction_classifier_maps_binary_head() -> None:
    source = STGCNClassifier(num_classes=2)
    target = STGCNClassifier(num_classes=2)
    state = {f"backbone.{key}": value for key, value in source.backbone.state_dict().items()}
    state["cls_head.fc_cls.weight"] = source.head.weight.detach().clone()
    state["cls_head.fc_cls.bias"] = source.head.bias.detach().clone()

    load_mmaction_classifier(target, {"state_dict": state})

    torch.testing.assert_close(target.head.weight, source.head.weight)
    torch.testing.assert_close(target.head.bias, source.head.bias)


def test_load_mmaction_classifier_accepts_gcn_head_fc_keys() -> None:
    source = STGCNClassifier(num_classes=2)
    target = STGCNClassifier(num_classes=2)
    state = {f"backbone.{key}": value for key, value in source.backbone.state_dict().items()}
    state["cls_head.fc.weight"] = source.head.weight.detach().clone()
    state["cls_head.fc.bias"] = source.head.bias.detach().clone()

    report = load_mmaction_classifier(target, {"state_dict": state})

    assert report.skipped == []
    torch.testing.assert_close(target.head.weight, source.head.weight)
    torch.testing.assert_close(target.head.bias, source.head.bias)
