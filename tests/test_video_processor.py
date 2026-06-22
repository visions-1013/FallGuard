from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from fallguard.inference.video import process_video
from fallguard.types import FallEvent, FrameResult


class FakePipeline:
    device = "cpu"
    model_version = "fixture-sha256"

    def reset(self) -> None:
        pass

    def process(self, frame: np.ndarray, timestamp: float) -> FrameResult:
        return FrameResult(0, timestamp, "non_fall", 0.1, None)

    def finish(self, timestamp: float):
        return (
            FallEvent(
                start_time=0.1,
                end_time=timestamp,
                trigger_time=0.3,
                max_probability=0.9,
            ),
        )


class FailingPipeline(FakePipeline):
    def process(self, frame: np.ndarray, timestamp: float) -> FrameResult:
        raise RuntimeError("synthetic inference failure")


def _write_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 32))
    assert writer.isOpened()
    for _ in range(3):
        writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    writer.release()


def test_process_video_writes_video_events_and_summary(tmp_path: Path) -> None:
    input_path = tmp_path / "input.avi"
    _write_video(input_path)

    result = process_video(input_path, tmp_path / "output", FakePipeline())

    assert result.video_path.is_file()
    assert result.events_path.is_file()
    assert result.summary_path.is_file()
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    events = json.loads(result.events_path.read_text(encoding="utf-8"))
    assert summary["status"] == "completed"
    assert summary["processed_frames"] == 3
    assert summary["device"] == "cpu"
    assert summary["model_version"] == "fixture-sha256"
    assert events[0]["trigger_delay_seconds"] == 0.2


def test_process_video_preserves_partial_file_and_failure_summary(tmp_path: Path) -> None:
    input_path = tmp_path / "input.avi"
    _write_video(input_path)

    result = process_video(input_path, tmp_path / "output", FailingPipeline())

    assert result.status == "failed"
    assert result.video_path.name.endswith(".partial.mp4")
    assert result.video_path.is_file()
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["error"] == "synthetic inference failure"
