from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Event
from typing import Any

import cv2
import numpy as np

from fallguard.models.graph import CocoGraph
from fallguard.types import FallEvent, FrameResult

PreviewCallback = Callable[[int, int, FrameResult, np.ndarray], None]


@dataclass(frozen=True)
class VideoProcessingResult:
    video_path: Path
    events_path: Path
    summary_path: Path
    status: str


def _event_payload(event: FallEvent) -> dict[str, object]:
    payload: dict[str, object] = asdict(event)
    payload["trigger_delay_seconds"] = (
        round(event.trigger_time - event.start_time, 6) if event.trigger_time is not None else None
    )
    return payload


def _annotate(frame, result: FrameResult):  # type: ignore[no-untyped-def]
    color = (0, 0, 255) if result.state == "fall" else (0, 200, 0)
    if result.pose is not None:
        points = result.pose.keypoints
        if result.pose.box is not None:
            x1, y1, x2, y2 = result.pose.box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        for left, right in CocoGraph.inward:
            if points[left, 2] > 0.2 and points[right, 2] > 0.2:
                cv2.line(
                    frame,
                    tuple(points[left, :2].astype(int)),
                    tuple(points[right, :2].astype(int)),
                    color,
                    2,
                )
        for x, y, confidence in points:
            if confidence > 0.2:
                cv2.circle(frame, (int(x), int(y)), 2, color, -1)
    probability = "--" if result.fall_probability is None else f"{result.fall_probability:.3f}"
    cv2.putText(
        frame,
        f"{result.state}  fall={probability}  t={result.timestamp:.2f}s",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )
    return frame


def process_video(
    input_path: Path,
    output_dir: Path,
    pipeline: Any,
    cancel_event: Event | None = None,
    progress: Callable[[int, int, FrameResult], None] | None = None,
    preview: PreviewCallback | None = None,
) -> VideoProcessingResult:
    input_path = input_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final_video = output_dir / f"{input_path.stem}_fallguard.mp4"
    partial_video = output_dir / f"{input_path.stem}_fallguard.partial.mp4"
    events_path = output_dir / f"{input_path.stem}_events.json"
    summary_path = output_dir / f"{input_path.stem}_summary.json"
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {input_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(
        str(partial_video),
        cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore[attr-defined]
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"cannot create output video: {partial_video}")
    started = time.perf_counter()
    processed = 0
    all_events = []
    status = "completed"
    error: str | None = None
    try:
        pipeline.reset()
        while True:
            if cancel_event is not None and cancel_event.is_set():
                status = "cancelled"
                break
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = processed / fps
            result = pipeline.process(frame, timestamp)
            all_events.extend(result.events)
            annotated = _annotate(frame, result)
            writer.write(annotated)
            processed += 1
            if progress is not None:
                progress(processed, total_frames, result)
            if preview is not None:
                preview(processed, total_frames, result, annotated.copy())
        last_timestamp = max((processed - 1) / fps, 0.0)
        all_events.extend(pipeline.finish(last_timestamp))
    except Exception as exc:  # The summary is the durable error channel for UI/CLI callers.
        status = "failed"
        error = str(exc)
    finally:
        capture.release()
        writer.release()
    if status == "completed":
        partial_video.replace(final_video)
        output_video = final_video
    else:
        output_video = partial_video
    events_path.write_text(
        json.dumps([_event_payload(event) for event in all_events], indent=2),
        encoding="utf-8",
    )
    summary = {
        "status": status,
        "input_path": str(input_path),
        "output_path": str(output_video),
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "processed_frames": processed,
        "processing_seconds": time.perf_counter() - started,
        "event_count": len(all_events),
        "model_version": getattr(pipeline, "model_version", "unknown"),
        "device": getattr(pipeline, "device", "unknown"),
        "audio_preserved": False,
    }
    if error is not None:
        summary["error"] = error
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return VideoProcessingResult(output_video, events_path, summary_path, status)
