from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TypedDict, cast

import torch

from .stgcn import STGCNClassifier, load_mmaction_classifier


class RuntimeMetadata(TypedDict):
    architecture: str
    classes: list[str]
    fall_threshold: float
    recovery_threshold: float
    pose_fps: float
    window_frames: int
    window_stride: int
    trigger_windows: int
    recovery_seconds: float
    cooldown_seconds: float
    passed_deployment_gate: bool
    source_checkpoint: str
    weights_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def export_runtime_bundle(
    mmaction_checkpoint: Path,
    output_dir: Path,
    fall_threshold: float,
    recovery_threshold: float = 0.35,
    pose_fps: float = 20.0,
    window_frames: int = 32,
    window_stride: int = 4,
    passed_deployment_gate: bool = False,
) -> tuple[Path, Path]:
    model = STGCNClassifier(num_classes=2)
    load_mmaction_classifier(model, mmaction_checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "best.pt"
    torch.save({"state_dict": model.state_dict()}, weights_path)
    metadata = {
        "architecture": "stgcn-coco17",
        "classes": ["non_fall", "fall"],
        "fall_threshold": fall_threshold,
        "recovery_threshold": recovery_threshold,
        "pose_fps": pose_fps,
        "window_frames": window_frames,
        "window_stride": window_stride,
        "trigger_windows": 2,
        "recovery_seconds": 2.0,
        "cooldown_seconds": 10.0,
        "passed_deployment_gate": passed_deployment_gate,
        "source_checkpoint": str(mmaction_checkpoint),
        "weights_sha256": _sha256(weights_path),
    }
    metadata_path = output_dir / "model_meta.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return weights_path, metadata_path


def load_runtime_bundle(
    bundle_dir: Path, device: str | None = None
) -> tuple[STGCNClassifier, RuntimeMetadata]:
    metadata = cast(
        RuntimeMetadata,
        json.loads((bundle_dir / "model_meta.json").read_text(encoding="utf-8")),
    )
    weights_path = bundle_dir / "best.pt"
    if _sha256(weights_path) != metadata["weights_sha256"]:
        raise ValueError("runtime weights SHA256 does not match model_meta.json")
    runtime_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model = STGCNClassifier(num_classes=2)
    payload = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["state_dict"])
    return model.to(runtime_device).eval(), metadata
