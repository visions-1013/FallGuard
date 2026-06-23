from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from fallguard.ui.history import create_recording_dir, load_event_intervals, load_history_records


def _write_record(root: Path, folder: str, source: str, event_count: int) -> None:
    record_dir = root / folder
    record_dir.mkdir(parents=True)
    video = record_dir / "result.mp4"
    video.write_bytes(b"video")
    (record_dir / "result_events.json").write_text("[]", encoding="utf-8")
    (record_dir / "result_summary.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "input_path": source,
                "output_path": str(video),
                "fps": 10.0,
                "processed_frames": 50,
                "event_count": event_count,
            }
        ),
        encoding="utf-8",
    )


def test_create_recording_dir_keeps_same_video_runs_separate(tmp_path: Path) -> None:
    now = datetime(2026, 6, 23, 9, 30, 0)

    first = create_recording_dir(tmp_path, Path("video (44).avi"), now=now)
    second = create_recording_dir(tmp_path, Path("video (44).avi"), now=now)

    assert first.name == "20260623_093000_video_44"
    assert second.name == "20260623_093000_video_44_2"


def test_load_history_records_is_newest_first_and_skips_broken_entries(tmp_path: Path) -> None:
    _write_record(tmp_path, "20260623_093000_first", "first.avi", 1)
    _write_record(tmp_path, "20260623_103000_second", "second.avi", 2)
    broken = tmp_path / "20260623_110000_broken"
    broken.mkdir()
    (broken / "broken_summary.json").write_text("not-json", encoding="utf-8")

    records = load_history_records(tmp_path)

    assert [record.source_name for record in records] == ["second.avi", "first.avi"]
    assert records[0].event_count == 2
    assert records[0].duration_seconds == 5.0


def test_load_event_intervals_uses_duration_for_open_event(tmp_path: Path) -> None:
    events_path = tmp_path / "events.json"
    events_path.write_text(
        json.dumps(
            [
                {"start_time": 1.0, "end_time": 2.5},
                {"start_time": 4.0, "end_time": None},
            ]
        ),
        encoding="utf-8",
    )

    assert load_event_intervals(events_path, duration=6.0) == [(1.0, 2.5), (4.0, 6.0)]
