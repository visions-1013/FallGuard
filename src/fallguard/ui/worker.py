from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from typing import Any

import numpy as np

from fallguard.inference.video import process_video
from fallguard.types import FrameResult


@dataclass(frozen=True)
class PreviewUpdate:
    processed: int
    total: int
    result: FrameResult
    frame: np.ndarray


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
        self.preview_queue: Queue[PreviewUpdate] = Queue(maxsize=1)
        self.result: Any = None
        self.error: Exception | None = None
        self.event_count = 0
        self._last_state = "non_fall"

    def _progress(self, processed: int, total: int, result: Any) -> None:
        while not self.progress_queue.empty():
            try:
                self.progress_queue.get_nowait()
            except Exception:
                break
        self.progress_queue.put_nowait((processed, total, result))

    def cancel(self) -> None:
        self.cancel_event.set()

    def _preview(
        self,
        processed: int,
        total: int,
        result: FrameResult,
        frame: np.ndarray,
    ) -> None:
        if result.state == "fall" and self._last_state != "fall":
            self.event_count += 1
        self._last_state = result.state
        while not self.preview_queue.empty():
            try:
                self.preview_queue.get_nowait()
            except Exception:
                break
        self.preview_queue.put_nowait(PreviewUpdate(processed, total, result, frame.copy()))

    def run(self) -> None:
        try:
            self.result = self.processor(
                self.input_path,
                self.output_dir,
                self.pipeline,
                self.cancel_event,
                self._progress,
                self._preview,
            )
        except Exception as exc:
            self.error = exc
