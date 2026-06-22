from __future__ import annotations

from pathlib import Path

import numpy as np

from fallguard.training.reporting import (
    export_run_provenance,
    generate_binary_report,
    select_threshold,
)


def test_select_threshold_enforces_minimum_precision() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])

    selection = select_threshold(labels, probabilities, min_precision=0.85)

    assert selection.threshold == 0.6
    assert selection.precision == 1.0
    assert selection.recall == 1.0


def test_generate_binary_report_creates_yolo_style_artifacts(tmp_path: Path) -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])

    history = [
        {
            "epoch": 1,
            "train_loss": 0.8,
            "val_loss": 0.7,
            "precision": 0.7,
            "recall": 0.8,
            "f1": 0.75,
            "lr": 0.001,
        },
        {
            "epoch": 2,
            "train_loss": 0.5,
            "val_loss": 0.4,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "lr": 0.0005,
        },
    ]

    metrics = generate_binary_report(
        labels, probabilities, tmp_path, threshold=0.6, history=history
    )

    assert metrics["recall"] == 1.0
    for name in [
        "metrics.json",
        "metrics.csv",
        "confusion_matrix.png",
        "pr_curve.png",
        "roc_curve.png",
        "threshold_curve.png",
        "training_curves.png",
        "history.csv",
    ]:
        assert (tmp_path / name).is_file(), name


def test_generate_binary_report_accepts_history_without_validation_loss(tmp_path: Path) -> None:
    history = [
        {
            "epoch": 1,
            "train_loss": 0.8,
            "precision": 0.9,
            "recall": 0.8,
            "f1": 0.85,
            "lr": 0.001,
        }
    ]

    generate_binary_report(
        np.asarray([0, 1]),
        np.asarray([0.1, 0.9]),
        tmp_path,
        threshold=0.5,
        history=history,
    )

    assert (tmp_path / "training_curves.png").is_file()


def test_export_run_provenance_records_hashes_and_best_epoch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best_binary_f1_epoch_12.pth"
    checkpoint.write_bytes(b"checkpoint")
    config = tmp_path / "stgcn.py"
    config.write_text("model = {}\n", encoding="utf-8")

    payload = export_run_provenance(checkpoint, config, tmp_path / "report")

    assert payload["best_epoch"] == 12
    assert len(payload["checkpoint_sha256"]) == 64
    assert len(payload["config_sha256"]) == 64
    assert (tmp_path / "report" / "provenance.json").is_file()
    assert (tmp_path / "report" / "training_config.py").read_text(
        encoding="utf-8"
    ) == "model = {}\n"
