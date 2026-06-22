from __future__ import annotations

from pathlib import Path

import yaml


def test_le2i_config_marks_unlabeled_scenes_as_excluded() -> None:
    config = yaml.safe_load(Path("configs/data/le2i.yaml").read_text(encoding="utf-8"))

    assert config["expected_videos"] == 190
    assert config["expected_supervised_videos"] == 130
    assert config["expected_excluded_videos"] == 60
    assert config["unlabeled_scenes"] == ["Lecture_room", "Office"]
    assert "external_test_scenes" not in config
