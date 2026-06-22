"""MMAction2 metrics imported only in the training environment."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from mmaction.registry import METRICS
from mmengine.evaluator import BaseMetric


@METRICS.register_module()
class BinaryClassificationMetric(BaseMetric):  # type: ignore[misc]
    default_prefix = "binary"

    def __init__(
        self,
        threshold: float = 0.5,
        collect_device: str = "cpu",
        prefix: str | None = None,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.threshold = threshold

    def process(
        self,
        data_batch: Any,
        data_samples: Sequence[dict[str, Any]],
    ) -> None:
        del data_batch
        for sample in data_samples:
            score = sample["pred_score"].detach().cpu().numpy()
            label = int(sample["gt_label"].item())
            self.results.append({"fall_probability": float(score[1]), "label": label})

    def compute_metrics(self, results: list[dict[str, Any]]) -> dict[str, float]:
        labels = np.asarray([item["label"] for item in results], dtype=np.int64)
        predictions = np.asarray(
            [item["fall_probability"] >= self.threshold for item in results], dtype=np.int64
        )
        true_positive = int(np.sum((labels == 1) & (predictions == 1)))
        false_positive = int(np.sum((labels == 0) & (predictions == 1)))
        false_negative = int(np.sum((labels == 1) & (predictions == 0)))
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        return {"precision": precision, "recall": recall, "f1": f1}
