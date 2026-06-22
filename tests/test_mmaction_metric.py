from __future__ import annotations

import pytest
import torch

pytest.importorskip("mmengine")
pytest.importorskip("mmaction.registry")

from fallguard.training.mmaction_metrics import BinaryClassificationMetric


def test_binary_classification_metric_reports_validation_scores() -> None:
    metric = BinaryClassificationMetric(threshold=0.5)
    metric.process(
        {},
        [
            {"pred_score": torch.tensor([0.9, 0.1]), "gt_label": torch.tensor([0])},
            {"pred_score": torch.tensor([0.1, 0.9]), "gt_label": torch.tensor([1])},
        ],
    )

    result = metric.compute_metrics(metric.results)

    assert result == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
