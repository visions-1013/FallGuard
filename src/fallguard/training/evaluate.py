from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from fallguard.data.preprocessing import pre_normalize_2d
from fallguard.models.stgcn import STGCNClassifier, load_mmaction_classifier

from .mmaction_runner import read_mmengine_history
from .reporting import generate_binary_report, select_threshold


def evaluate_pose_dataset(
    checkpoint_path: Path,
    dataset_path: Path,
    split: str,
    output_dir: Path,
    device: str | None = None,
    batch_size: int = 64,
    history_path: Path | None = None,
) -> dict[str, float | int]:
    with dataset_path.open("rb") as handle:
        payload = pickle.load(handle)
    selected_ids = set(payload["split"][split])
    annotations = [item for item in payload["annotations"] if item["frame_dir"] in selected_ids]
    if not annotations:
        raise ValueError(f"split {split} contains no samples")
    labels = np.asarray([item["label"] for item in annotations], dtype=np.int64)
    model_inputs = []
    for item in annotations:
        keypoints = np.concatenate(
            [item["keypoint"], item["keypoint_score"][..., None]], axis=-1
        ).astype(np.float32)
        height, width = item["img_shape"]
        model_inputs.append(pre_normalize_2d(keypoints, width=width, height=height))
    inputs = np.stack(model_inputs)
    runtime_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model = STGCNClassifier(num_classes=2).to(runtime_device)
    load_mmaction_classifier(model, checkpoint_path)
    model.eval()
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            batch = torch.from_numpy(inputs[start : start + batch_size]).to(runtime_device)
            probabilities.append(torch.softmax(model(batch), dim=1)[:, 1].cpu().numpy())
    fall_probabilities = np.concatenate(probabilities)
    threshold = select_threshold(labels, fall_probabilities).threshold
    predictions = (fall_probabilities >= threshold).astype(np.int64)
    history = (
        read_mmengine_history(history_path)
        if history_path is not None and history_path.is_file()
        else None
    )
    metrics: dict[str, float | int] = generate_binary_report(
        labels, fall_probabilities, output_dir, threshold, history=history
    )
    metrics["samples"] = len(labels)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["sample_id", "label", "fall_probability", "prediction"]
        )
        writer.writeheader()
        for item, label, probability, prediction in zip(
            annotations, labels, fall_probabilities, predictions, strict=True
        ):
            writer.writerow(
                {
                    "sample_id": item["frame_dir"],
                    "label": int(label),
                    "fall_probability": float(probability),
                    "prediction": int(prediction),
                }
            )

    error_fields = ["sample_id", "scene", "error_type", "fall_probability"]
    with (output_dir / "errors.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=error_fields)
        writer.writeheader()
        for item, label, probability, prediction in zip(
            annotations, labels, fall_probabilities, predictions, strict=True
        ):
            if label == prediction:
                continue
            writer.writerow(
                {
                    "sample_id": item["frame_dir"],
                    "scene": item.get("scene", "unknown"),
                    "error_type": "false_positive" if prediction else "false_negative",
                    "fall_probability": float(probability),
                }
            )

    scene_rows = []
    for scene in sorted({item.get("scene", "unknown") for item in annotations}):
        indices = [
            index for index, item in enumerate(annotations) if item.get("scene", "unknown") == scene
        ]
        scene_labels = labels[indices]
        scene_predictions = predictions[indices]
        scene_rows.append(
            {
                "scene": scene,
                "samples": len(indices),
                "accuracy": float(accuracy_score(scene_labels, scene_predictions)),
                "precision": float(
                    precision_score(scene_labels, scene_predictions, zero_division=0)
                ),
                "recall": float(recall_score(scene_labels, scene_predictions, zero_division=0)),
                "f1": float(f1_score(scene_labels, scene_predictions, zero_division=0)),
            }
        )
    (output_dir / "scene_metrics.json").write_text(
        json.dumps(scene_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output_dir / "scene_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scene_rows[0]))
        writer.writeheader()
        writer.writerows(scene_rows)
    return metrics
