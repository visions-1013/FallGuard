from __future__ import annotations

from pathlib import Path

from fallguard.ui.worker import VideoWorker


def test_video_worker_reports_result_without_touching_tkinter(tmp_path: Path) -> None:
    received = []

    def processor(input_path, output_dir, pipeline, cancel_event, progress):
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
