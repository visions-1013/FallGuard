from __future__ import annotations

from pathlib import Path

import numpy as np

from fallguard.types import FrameResult
from fallguard.ui.worker import VideoWorker


def test_video_worker_reports_result_without_touching_tkinter(tmp_path: Path) -> None:
    received = []

    def processor(input_path, output_dir, pipeline, cancel_event, progress, preview):
        received.append((input_path, output_dir, cancel_event.is_set()))
        return "finished"

    worker = VideoWorker(
        input_path=tmp_path / "input.avi",
        output_dir=tmp_path,
        pipeline=object(),
        processor=processor,
    )
    worker.start()
    worker.join(timeout=2)

    assert worker.result == "finished"
    assert worker.error is None
    assert received[0][2] is False


def test_video_worker_keeps_latest_preview_and_counts_fall_transitions(tmp_path: Path) -> None:
    worker = VideoWorker(tmp_path / "input.avi", tmp_path, object())
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    worker._preview(1, 3, FrameResult(0, 0.0, "non_fall", 0.1, None), frame)
    worker._preview(2, 3, FrameResult(1, 0.1, "fall", 0.9, None), frame)
    worker._preview(3, 3, FrameResult(2, 0.2, "fall", 0.8, None), frame)

    assert worker.preview_queue.qsize() == 1
    latest = worker.preview_queue.get_nowait()
    assert latest.processed == 3
    assert latest.result.frame_index == 2
    assert worker.event_count == 1
