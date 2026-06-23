from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HistoryRecord:
    record_dir: Path
    video_path: Path
    events_path: Path
    summary_path: Path
    recorded_at: datetime
    source_name: str
    status: str
    event_count: int
    duration_seconds: float


def create_recording_dir(
    root: Path, input_path: Path, *, now: datetime | None = None
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    safe_stem = re.sub(r"[^\w-]+", "_", input_path.stem, flags=re.UNICODE).strip("_")
    safe_stem = safe_stem or "video"
    base_name = f"{timestamp}_{safe_stem}"
    candidate = root / base_name
    suffix = 2
    while candidate.exists():
        candidate = root / f"{base_name}_{suffix}"
        suffix += 1
    candidate.mkdir()
    return candidate


def _recorded_at(record_dir: Path) -> datetime:
    try:
        return datetime.strptime(record_dir.name[:15], "%Y%m%d_%H%M%S")
    except ValueError:
        return datetime.fromtimestamp(record_dir.stat().st_mtime)


def _load_record(summary_path: Path) -> HistoryRecord | None:
    try:
        payload: dict[str, Any] = json.loads(summary_path.read_text(encoding="utf-8"))
        record_dir = summary_path.parent
        video_path = Path(str(payload["output_path"]))
        if not video_path.is_file():
            video_path = record_dir / video_path.name
        events_name = summary_path.name.replace("_summary.json", "_events.json")
        events_path = summary_path.with_name(events_name)
        fps = float(payload.get("fps", 0.0))
        frames = int(payload.get("processed_frames", 0))
        if not video_path.is_file() or not events_path.is_file():
            return None
        return HistoryRecord(
            record_dir=record_dir,
            video_path=video_path,
            events_path=events_path,
            summary_path=summary_path,
            recorded_at=_recorded_at(record_dir),
            source_name=Path(str(payload.get("input_path", video_path.name))).name,
            status=str(payload.get("status", "unknown")),
            event_count=int(payload.get("event_count", 0)),
            duration_seconds=frames / fps if fps > 0 else 0.0,
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def load_history_records(root: Path) -> list[HistoryRecord]:
    if not root.is_dir():
        return []
    records = [
        record
        for summary_path in root.glob("*/*_summary.json")
        if (record := _load_record(summary_path)) is not None
    ]
    return sorted(records, key=lambda record: record.recorded_at, reverse=True)


def load_event_intervals(events_path: Path, duration: float) -> list[tuple[float, float]]:
    try:
        payload = json.loads(events_path.read_text(encoding="utf-8"))
        intervals = []
        for event in payload:
            start = float(event["start_time"])
            end_value = event.get("end_time")
            end = duration if end_value is None else float(end_value)
            intervals.append((start, end))
        return intervals
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return []
