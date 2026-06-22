from __future__ import annotations

import json
import pickle
from pathlib import Path

from fallguard.training.mmaction_runner import (
    build_training_overrides,
    compute_class_weights,
    parse_windows_code_page,
    read_mmengine_history,
)


def test_build_training_overrides_sets_fold_and_pretrained_backbone(tmp_path: Path) -> None:
    overrides = build_training_overrides(
        annotation_file=tmp_path / "poses.pkl",
        work_dir=tmp_path / "work",
        fold=2,
        pretrained=True,
    )

    assert overrides["train_dataloader.dataset.split"] == "fold_2_train"
    assert overrides["val_dataloader.dataset.split"] == "fold_2_val"
    assert overrides["model.backbone.init_cfg"]["prefix"] == "backbone."
    assert "download.openmmlab.com" in overrides["model.backbone.init_cfg"]["checkpoint"]
    assert overrides["custom_hooks.0.freeze_epochs"] == 5


def test_build_training_overrides_does_not_freeze_random_backbone(tmp_path: Path) -> None:
    overrides = build_training_overrides(
        annotation_file=tmp_path / "poses.pkl",
        work_dir=tmp_path / "work",
        fold=0,
        pretrained=False,
    )

    assert overrides["custom_hooks.0.freeze_epochs"] == 0


def test_build_training_overrides_resumes_existing_work_dir(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "last_checkpoint").write_text("epoch_3.pth", encoding="utf-8")

    overrides = build_training_overrides(
        annotation_file=tmp_path / "poses.pkl",
        work_dir=work_dir,
        fold=0,
        pretrained=True,
    )

    assert overrides["resume"] is True


def test_compute_class_weights_uses_inverse_frequency(tmp_path: Path) -> None:
    annotations = tmp_path / "poses.pkl"
    with annotations.open("wb") as handle:
        pickle.dump(
            {"annotations": [{"label": 0}, {"label": 0}, {"label": 0}, {"label": 1}]},
            handle,
        )

    weights = compute_class_weights(annotations)

    assert weights == [2 / 3, 2.0]


def test_parse_windows_code_page() -> None:
    assert parse_windows_code_page("cp936") == 936
    assert parse_windows_code_page("UTF-8") == 65001
    assert parse_windows_code_page("unknown") is None


def test_read_mmengine_history_converts_json_lines(tmp_path: Path) -> None:
    log = tmp_path / "scalars.json"
    log.write_text(
        "\n".join(
            [
                json.dumps({"step": 20, "epoch": 1, "train/loss": 0.8, "train/lr": 0.001}),
                json.dumps(
                    {
                        "step": 0,
                        "binary/precision": 0.9,
                        "binary/recall": 0.8,
                        "binary/f1": 0.85,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    history = read_mmengine_history(log)

    assert history == [
        {
            "epoch": 1.0,
            "train_loss": 0.8,
            "precision": 0.9,
            "recall": 0.8,
            "f1": 0.85,
            "lr": 0.001,
        }
    ]
