from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from typing import Any

from fallguard.inference.video import process_video


class VideoWorker(Thread):
    def __init__(
        self,
        input_path: Path,
        output_dir: Path,
        pipeline: Any,
        processor: Callable[..., Any] = process_video,
    ) -> None:
        super().__init__(daemon=True)
        self.input_path = input_path
        self.output_dir = output_dir
        self.pipeline = pipeline
        self.processor = processor
        self.cancel_event = Event()
        self.progress_queue: Queue[tuple[int, int, Any]] = Queue(maxsize=1)
        self.result: Any = None
        self.error: Exception | None = None

    def _progress(self, processed: int, total: int, result: Any) -> None:
        while not self.progress_queue.empty():
            try:
                self.progress_queue.get_nowait()
            except Exception:
                break
        self.progress_queue.put_nowait((processed, total, result))

    def cancel(self) -> None:
        self.cancel_event.set()

    def run(self) -> None:
        try:
            self.result = self.processor(
                self.input_path,
                self.output_dir,
                self.pipeline,
                self.cancel_event,
                self._progress,
            )
        except Exception as exc:
            self.error = exc
