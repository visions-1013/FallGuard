from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from .graph import CocoGraph


class UnitGCN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, adjacency: torch.Tensor) -> None:
        super().__init__()
        self.num_subsets = adjacency.size(0)
        self.register_buffer("A", adjacency)
        self.A: torch.Tensor
        self.PA = nn.Parameter(torch.ones_like(adjacency))
        self.conv = nn.Conv2d(in_channels, out_channels * self.num_subsets, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, _, frames, vertices = inputs.shape
        adjacency = self.A * self.PA
        features = self.conv(inputs).view(batch, self.num_subsets, -1, frames, vertices)
        features = torch.einsum("nkctv,kvw->nctw", features, adjacency).contiguous()
        return cast(torch.Tensor, self.act(self.bn(features)))


class UnitTCN(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 9, stride: int = 1
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
            stride=(stride, 1),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.drop(self.bn(self.conv(inputs))))


class STGCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adjacency: torch.Tensor,
        stride: int = 1,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.gcn = UnitGCN(in_channels, out_channels, adjacency)
        self.tcn = UnitTCN(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual: nn.Module | None = None
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = UnitTCN(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = 0 if self.residual is None else self.residual(inputs)
        return cast(torch.Tensor, self.relu(self.tcn(self.gcn(inputs)) + residual))


class STGCNBackbone(nn.Module):
    def __init__(self, in_channels: int = 3, num_person: int = 1) -> None:
        super().__init__()
        adjacency = torch.tensor(CocoGraph().A, dtype=torch.float32)
        self.data_bn = nn.BatchNorm1d(in_channels * 17)
        channels = [64, 64, 64, 64, 128, 128, 128, 256, 256, 256]
        blocks: list[nn.Module] = []
        current = in_channels
        for stage, output in enumerate(channels, start=1):
            stride = 2 if stage in (5, 8) else 1
            blocks.append(
                STGCNBlock(
                    current,
                    output,
                    adjacency.clone(),
                    stride=stride,
                    residual=stage != 1,
                )
            )
            current = output
        self.gcn = nn.ModuleList(blocks)
        self.num_person = num_person
        self.out_channels = channels[-1]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 5 or inputs.shape[-2:] != (17, 3):
            raise ValueError(f"expected (N, M, T, 17, 3), got {tuple(inputs.shape)}")
        batch, people, frames, vertices, channels = inputs.shape
        features = inputs.permute(0, 1, 3, 4, 2).contiguous()
        features = self.data_bn(features.view(batch * people, vertices * channels, frames))
        features = (
            features.view(batch, people, vertices, channels, frames)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
            .view(batch * people, channels, frames, vertices)
        )
        for block in self.gcn:
            features = cast(torch.Tensor, block(features))
        return features.reshape((batch, people) + features.shape[1:])


class STGCNClassifier(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.backbone = STGCNBackbone()
        self.head = nn.Linear(self.backbone.out_channels, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.backbone(inputs)
        pooled = features.mean(dim=(1, 3, 4))
        return cast(torch.Tensor, self.head(pooled))


@dataclass(frozen=True)
class CheckpointLoadReport:
    loaded: int
    missing: list[str]
    skipped: list[str]


def load_mmaction_backbone(
    model: STGCNClassifier, checkpoint: str | Path | Mapping[str, Any]
) -> CheckpointLoadReport:
    if isinstance(checkpoint, (str, Path)):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    else:
        payload = checkpoint
    raw_state = payload.get("state_dict", payload)
    backbone_state: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, value in raw_state.items():
        if key.startswith("backbone."):
            backbone_state[key.removeprefix("backbone.")] = value
        else:
            skipped.append(key)
    result = model.backbone.load_state_dict(backbone_state, strict=False)
    return CheckpointLoadReport(
        loaded=len(backbone_state) - len(result.unexpected_keys),
        missing=sorted(result.missing_keys),
        skipped=sorted(skipped + list(result.unexpected_keys)),
    )


def load_mmaction_classifier(
    model: STGCNClassifier, checkpoint: str | Path | Mapping[str, Any]
) -> CheckpointLoadReport:
    if isinstance(checkpoint, (str, Path)):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    else:
        payload = checkpoint
    report = load_mmaction_backbone(model, payload)
    state = payload.get("state_dict", payload)
    head_prefix = next(
        (
            prefix
            for prefix in ("cls_head.fc", "cls_head.fc_cls")
            if f"{prefix}.weight" in state and f"{prefix}.bias" in state
        ),
        None,
    )
    if head_prefix is None:
        raise KeyError("checkpoint does not contain a supported MMAction2 GCNHead")
    head_state = {
        "weight": state[f"{head_prefix}.weight"],
        "bias": state[f"{head_prefix}.bias"],
    }
    model.head.load_state_dict(head_state)
    skipped = [
        key for key in report.skipped if key not in (f"{head_prefix}.weight", f"{head_prefix}.bias")
    ]
    return CheckpointLoadReport(
        loaded=report.loaded + 2,
        missing=report.missing,
        skipped=skipped,
    )
