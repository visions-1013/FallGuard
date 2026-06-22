from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    precision: float
    recall: float
    f1: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def export_run_provenance(
    checkpoint_path: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    epoch_match = re.search(r"epoch[_-]?(\d+)", checkpoint_path.stem)
    versions = {}
    for package in ("torch", "ultralytics", "mmaction2", "mmengine", "mmcv-lite"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    payload: dict[str, object] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "best_epoch": int(epoch_match.group(1)) if epoch_match else None,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
    }
    shutil.copy2(config_path, output_dir / "training_config.py")
    (output_dir / "provenance.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def _threshold_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> ThresholdSelection:
    predictions = (probabilities >= threshold).astype(np.int64)
    return ThresholdSelection(
        threshold=float(threshold),
        precision=float(precision_score(labels, predictions, zero_division=0)),
        recall=float(recall_score(labels, predictions, zero_division=0)),
        f1=float(f1_score(labels, predictions, zero_division=0)),
    )


def select_threshold(
    labels: np.ndarray, probabilities: np.ndarray, min_precision: float = 0.85
) -> ThresholdSelection:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    candidates = [
        _threshold_metrics(labels, probabilities, value) for value in np.unique(probabilities)
    ]
    eligible = [item for item in candidates if item.precision >= min_precision]
    pool = eligible or candidates
    return max(pool, key=lambda item: (item.recall, item.precision, item.f1, -item.threshold))


def generate_binary_report(
    labels: np.ndarray,
    probabilities: np.ndarray,
    output_dir: Path,
    threshold: float,
    history: list[dict[str, float]] | None = None,
    predictions: np.ndarray | None = None,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if predictions is None:
        predictions = (probabilities >= threshold).astype(np.int64)
    else:
        predictions = np.asarray(predictions, dtype=np.int64)
        if predictions.shape != labels.shape:
            raise ValueError("predictions must match labels")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)

    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    display = ConfusionMatrixDisplay(matrix, display_labels=["non_fall", "fall"])
    display.plot(cmap="Blues", colorbar=False)
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close()

    precision, recall, _ = precision_recall_curve(labels, probabilities)
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(output_dir / "pr_curve.png", dpi=160)
    plt.close()

    false_positive_rate, true_positive_rate, _ = roc_curve(labels, probabilities)
    plt.plot(false_positive_rate, true_positive_rate)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=160)
    plt.close()

    selections = [
        _threshold_metrics(labels, probabilities, value) for value in np.linspace(0.0, 1.0, 101)
    ]
    values = [item.threshold for item in selections]
    plt.plot(values, [item.precision for item in selections], label="Precision")
    plt.plot(values, [item.recall for item in selections], label="Recall")
    plt.plot(values, [item.f1 for item in selections], label="F1")
    plt.axvline(threshold, linestyle="--", color="black", label="Selected")
    plt.xlabel("Fall threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "threshold_curve.png", dpi=160)
    plt.close()

    if history:
        fields = ["epoch"] + sorted({field for row in history for field in row if field != "epoch"})
        with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(history)
        epochs = [row["epoch"] for row in history]
        figure, axes = plt.subplots(1, 3, figsize=(14, 4))
        if any("train_loss" in row for row in history):
            axes[0].plot(
                epochs,
                [row.get("train_loss", np.nan) for row in history],
                label="Train",
            )
        if any("val_loss" in row for row in history):
            axes[0].plot(
                epochs,
                [row.get("val_loss", np.nan) for row in history],
                label="Validation",
            )
        axes[0].set_title("Loss")
        if axes[0].lines:
            axes[0].legend()
        for metric in ("precision", "recall", "f1"):
            if any(metric in row for row in history):
                axes[1].plot(
                    epochs,
                    [row.get(metric, np.nan) for row in history],
                    label=metric.title(),
                )
        axes[1].set_title("Validation metrics")
        if axes[1].lines:
            axes[1].legend()
        if any("lr" in row for row in history):
            axes[2].plot(epochs, [row.get("lr", np.nan) for row in history])
        axes[2].set_title("Learning rate")
        figure.tight_layout()
        figure.savefig(output_dir / "training_curves.png", dpi=160)
        plt.close(figure)
    return metrics
