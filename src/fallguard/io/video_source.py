from __future__ import annotations

from pathlib import Path

import cv2


def open_video_source(source: int | str | Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(source) if isinstance(source, Path) else source)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video source: {source}")
    return capture
