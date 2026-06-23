from __future__ import annotations

import tkinter as tk
from collections.abc import Iterable
from typing import Any


def intervals_to_pixels(
    intervals: Iterable[tuple[float, float]], duration: float, width: int
) -> list[tuple[int, int]]:
    if duration <= 0 or width <= 0:
        return []
    pixels: list[tuple[int, int]] = []
    for first, second in intervals:
        start, end = sorted((first, second))
        start = min(max(start, 0.0), duration)
        end = min(max(end, 0.0), duration)
        pixels.append((round(start / duration * width), round(end / duration * width)))
    return pixels


class EventTimeline(tk.Canvas):
    def __init__(self, master: tk.Misc, **kwargs: Any) -> None:
        super().__init__(master, height=22, highlightthickness=0, **kwargs)
        self._duration = 0.0
        self._position = 0.0
        self._intervals: list[tuple[float, float]] = []
        self.bind("<Configure>", lambda _event: self._redraw())

    def set_data(
        self,
        duration: float,
        position: float,
        intervals: Iterable[tuple[float, float]],
    ) -> None:
        self._duration = max(duration, 0.0)
        self._position = max(position, 0.0)
        self._intervals = list(intervals)
        self._redraw()

    def _redraw(self) -> None:
        self.delete("all")
        width = max(self.winfo_width(), 1)
        height = max(self.winfo_height(), 1)
        self.create_rectangle(0, 6, width, height - 6, fill="#25324a", outline="")
        if self._duration <= 0:
            return
        progress_x = min(self._position / self._duration, 1.0) * width
        self.create_rectangle(0, 6, progress_x, height - 6, fill="#2f80ed", outline="")
        for start_x, end_x in intervals_to_pixels(self._intervals, self._duration, width):
            self.create_rectangle(start_x, 3, end_x, height - 3, fill="#ef4444", outline="")
        self.create_line(progress_x, 1, progress_x, height - 1, fill="#ffffff", width=2)
