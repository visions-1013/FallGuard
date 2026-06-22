from __future__ import annotations

import json
import locale
import os
import pickle
import re
from pathlib import Path
from typing import Any

OFFICIAL_STGCN_2D_JOINT = (
    "https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/"
    "stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/"
    "stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221129-484a394a.pth"
)


def parse_windows_code_page(encoding: str) -> int | None:
    normalized = encoding.lower().replace("-", "")
    if normalized == "utf8":
        return 65001
    match = re.fullmatch(r"cp(\d+)", normalized)
    return int(match.group(1)) if match else None


def _align_windows_console_encoding() -> None:
    if os.name != "nt":
        return
    code_page = parse_windows_code_page(locale.getpreferredencoding(False))
    if code_page is None:
        return
    import ctypes

    ctypes.windll.kernel32.SetConsoleOutputCP(code_page)


def build_training_overrides(
    annotation_file: Path,
    work_dir: Path,
    fold: int,
    pretrained: bool,
    class_weights: list[float] | None = None,
) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "work_dir": str(work_dir),
        "train_dataloader.dataset.ann_file": str(annotation_file),
        "train_dataloader.dataset.split": f"fold_{fold}_train",
        "val_dataloader.dataset.ann_file": str(annotation_file),
        "val_dataloader.dataset.split": f"fold_{fold}_val",
        "test_dataloader.dataset.ann_file": str(annotation_file),
        "test_dataloader.dataset.split": f"fold_{fold}_val",
        "custom_hooks.0.freeze_epochs": 5 if pretrained else 0,
    }
    overrides["model.backbone.init_cfg"] = (
        {
            "type": "Pretrained",
            "checkpoint": OFFICIAL_STGCN_2D_JOINT,
            "prefix": "backbone.",
        }
        if pretrained
        else None
    )
    if class_weights is not None:
        overrides["model.cls_head.loss_cls.class_weight"] = class_weights
    if (work_dir / "last_checkpoint").is_file():
        overrides["resume"] = True
    return overrides


def compute_class_weights(annotation_file: Path, split: str | None = None) -> list[float]:
    with annotation_file.open("rb") as handle:
        payload = pickle.load(handle)
    selected = set(payload["split"][split]) if split is not None else None
    counts = [0, 0]
    for annotation in payload["annotations"]:
        if selected is not None and annotation["frame_dir"] not in selected:
            continue
        counts[int(annotation["label"])] += 1
    if not all(counts):
        raise ValueError(f"both classes are required, got counts={counts}")
    total = sum(counts)
    return [total / (2 * count) for count in counts]


def run_mmaction_training(
    config_path: Path,
    annotation_file: Path,
    work_dir: Path,
    fold: int,
    pretrained: bool,
    max_epochs: int | None = None,
) -> None:
    _align_windows_console_encoding()
    try:
        from mmaction.utils import register_all_modules
        from mmengine.config import Config
        from mmengine.runner import Runner
    except ImportError as exc:
        raise RuntimeError(
            "MMAction2 training dependencies are missing; install fallguard[train]"
        ) from exc
    register_all_modules(init_default_scope=True)
    config = Config.fromfile(str(config_path))
    overrides = build_training_overrides(
        annotation_file,
        work_dir,
        fold,
        pretrained,
        class_weights=compute_class_weights(annotation_file, f"fold_{fold}_train"),
    )
    if max_epochs is not None:
        overrides["train_cfg.max_epochs"] = max_epochs
        overrides["param_scheduler.0.T_max"] = max_epochs
    config.merge_from_dict(overrides)
    Runner.from_cfg(config).train()


def read_mmengine_history(path: Path) -> list[dict[str, float]]:
    epochs: dict[int, dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        validation_line = any(key.endswith(("/precision", "/recall", "/f1")) for key in item)
        if "epoch" in item:
            epoch = int(item["epoch"])
        else:
            epoch = int(item.get("step", 0)) + (1 if validation_line else 0)
        row = epochs.setdefault(epoch, {"epoch": float(epoch)})
        mappings = {
            "train/loss": "train_loss",
            "loss": "train_loss",
            "train/lr": "lr",
            "lr": "lr",
        }
        for source, target in mappings.items():
            if source in item:
                row[target] = float(item[source])
        for metric in ("precision", "recall", "f1"):
            metric_source = next((key for key in item if key.endswith(f"/{metric}")), None)
            if metric_source is not None:
                row[metric] = float(item[metric_source])
    required = ("train_loss", "precision", "recall", "f1", "lr")
    return [row for _, row in sorted(epochs.items()) if all(field in row for field in required)]
