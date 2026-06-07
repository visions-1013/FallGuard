from datetime import datetime
from pathlib import Path

from fallguard.ui import replay
from fallguard.ui.replay import (
    append_recording_fall_point,
    build_fall_intervals,
    build_recording_session,
    build_recording_session_from_file,
    load_recording_markers,
    load_recording_sessions,
    recording_marker_path,
    recording_video_files,
    recording_filename,
    save_recording_markers,
)

from app import streamlit_app
from app.streamlit_app import (
    build_event,
    calculate_processing_fps,
    ensure_pipeline_loaded,
    release_camera_capture,
    reset_detection_state,
    should_record_frame,
    should_show_preview_message,
    source_fps,
)


class ReleasableCapture:
    def __init__(self):
        self.released = False

    def release(self):
        self.released = True


def test_build_event_uses_chinese_columns_and_english_state():
    event = build_event(12, "lying", "angle=80.0")

    assert event == {"帧号": "12", "状态": "lying", "说明": "angle=80.0"}


def test_ensure_pipeline_loaded_shows_message_before_loading():
    calls = []

    class StatusSlot:
        def info(self, message):
            calls.append(("info", message))

    expected_pipeline = object()

    def loader():
        calls.append(("loader", "called"))
        return expected_pipeline

    pipeline = ensure_pipeline_loaded(StatusSlot(), loader=loader)

    assert pipeline is expected_pipeline
    assert calls == [("info", "正在加载 YOLO-Pose，请稍候..."), ("loader", "called")]


def test_should_show_preview_message_only_when_not_running():
    assert should_show_preview_message(running=False) is True
    assert should_show_preview_message(running=True) is False


def test_render_detection_messages_only_keeps_non_status_subheaders(monkeypatch):
    subheaders = []
    empty_calls = []

    class Slot:
        pass

    monkeypatch.setattr(streamlit_app.st, "subheader", lambda label: subheaders.append(label))
    monkeypatch.setattr(streamlit_app.st, "empty", lambda: empty_calls.append("empty") or Slot())

    message_slot = streamlit_app.render_detection_messages()

    assert message_slot is not None
    assert "当前状态" not in subheaders
    assert empty_calls == ["empty"]


def test_calculate_processing_fps_handles_zero_elapsed_time():
    assert calculate_processing_fps(0) == 0.0
    assert calculate_processing_fps(-0.1) == 0.0
    assert calculate_processing_fps(0.25) == 4.0


def test_source_fps_falls_back_when_capture_value_is_invalid():
    class Capture:
        def __init__(self, fps):
            self.fps = fps

        def get(self, _property_id):
            return self.fps

    assert source_fps(Capture(0)) == 30.0
    assert source_fps(Capture(-1)) == 30.0
    assert source_fps(Capture(24)) == 24.0


def test_reset_detection_state_clears_runtime_metrics():
    capture = ReleasableCapture()
    state = {
        "events": [{"帧号": "1", "状态": "standing", "说明": "angle=1.0"}],
        "recorded_sessions": [{"文件名": "fallguard-recording.mp4"}],
        "running": True,
        "frame_index": 12,
        "input_fps": 25.0,
        "processing_fps": 8.5,
        "last_frame_rgb": object(),
        "camera_capture": capture,
        "camera_source_index": 0,
    }

    reset_detection_state(state)

    assert state["events"] == []
    assert state["running"] is False
    assert state["frame_index"] == 0
    assert state["input_fps"] == 0.0
    assert state["processing_fps"] == 0.0
    assert state["last_frame_rgb"] is None
    assert state["recorded_sessions"] == [{"文件名": "fallguard-recording.mp4"}]
    assert capture.released is True
    assert state["camera_capture"] is None
    assert state["camera_source_index"] is None


def test_release_camera_capture_releases_and_clears_state():
    capture = ReleasableCapture()
    state = {"camera_capture": capture, "camera_source_index": 0}

    release_camera_capture(state)

    assert capture.released is True
    assert state["camera_capture"] is None
    assert state["camera_source_index"] is None


def test_should_record_frame_uses_sampling_interval():
    assert should_record_frame(now=10.0, last_recorded_at=0.0, interval_seconds=1.0) is True
    assert should_record_frame(now=10.5, last_recorded_at=10.0, interval_seconds=1.0) is False
    assert should_record_frame(now=11.0, last_recorded_at=10.0, interval_seconds=1.0) is True


def test_recording_filename_uses_stable_mp4_name():
    started_at = datetime(2026, 6, 5, 8, 9, 10)

    assert recording_filename(started_at) == "fallguard_20260605_080910.mp4"


def test_build_recording_session_uses_chinese_fields():
    session = build_recording_session(
        Path("recordings/fallguard_20260605_080910.mp4"),
        datetime(2026, 6, 5, 8, 9, 10),
        datetime(2026, 6, 5, 8, 9, 15),
        frame_count=4,
    )

    assert session == {
        "开始时间": "2026-06-05 08:09:10",
        "结束时间": "2026-06-05 08:09:15",
        "时长(秒)": "5.0",
        "帧数": "4",
        "文件名": "fallguard_20260605_080910.mp4",
        "文件路径": "recordings\\fallguard_20260605_080910.mp4",
    }


def test_recording_video_files_reads_supported_files_in_time_order(tmp_path, monkeypatch):
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    older = recordings / "fallguard_20260605_080910.mp4"
    newer = recordings / "fallguard_20260605_080920.MOV"
    ignored = recordings / "notes.txt"
    older.write_bytes(b"")
    newer.write_bytes(b"")
    ignored.write_text("not a video", encoding="utf-8")

    monkeypatch.setattr(replay, "recordings_dir", lambda: recordings)

    assert recording_video_files() == [older, newer]


def test_build_recording_session_from_file_uses_filename_and_video_metadata(tmp_path, monkeypatch):
    path = tmp_path / "recordings" / "fallguard_20260605_080910.mp4"
    path.parent.mkdir()
    path.write_bytes(b"video")

    class Capture:
        def isOpened(self):
            return True

        def get(self, property_id):
            if property_id == replay.cv2.CAP_PROP_FRAME_COUNT:
                return 25
            if property_id == replay.cv2.CAP_PROP_FPS:
                return 5
            return 0

        def release(self):
            pass

    monkeypatch.setattr(replay.cv2, "VideoCapture", lambda _path: Capture())

    session = build_recording_session_from_file(path)

    assert session == {
        "开始时间": "2026-06-05 08:09:10",
        "结束时间": "2026-06-05 08:09:15",
        "时长(秒)": "5.0",
        "帧数": "25",
        "文件名": "fallguard_20260605_080910.mp4",
        "文件路径": str(path),
    }


def test_load_recording_sessions_merges_disk_and_session_records(tmp_path, monkeypatch):
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    existing_path = recordings / "fallguard_20260605_080910.mp4"
    new_path = recordings / "fallguard_20260605_080920.mp4"
    existing_path.write_bytes(b"video")
    new_path.write_bytes(b"video")

    session_record = {
        "开始时间": "2026-06-05 08:09:10",
        "结束时间": "2026-06-05 08:09:12",
        "时长(秒)": "2.0",
        "帧数": "2",
        "文件名": existing_path.name,
        "文件路径": str(existing_path),
    }

    class Capture:
        def __init__(self, path_text):
            self.path_text = path_text

        def isOpened(self):
            return True

        def get(self, property_id):
            if property_id == replay.cv2.CAP_PROP_FRAME_COUNT:
                return 10
            if property_id == replay.cv2.CAP_PROP_FPS:
                return 5
            return 0

        def release(self):
            pass

    monkeypatch.setattr(replay, "recordings_dir", lambda: recordings)
    monkeypatch.setattr(replay.cv2, "VideoCapture", lambda path_text: Capture(path_text))

    sessions = load_recording_sessions([session_record])

    assert sessions[0] == session_record
    assert sessions[1]["文件名"] == new_path.name
    assert sessions[1]["文件路径"] == str(new_path)
    assert len(sessions) == 2


def test_recording_marker_path_uses_sidecar_json_name():
    video_path = Path("recordings/fallguard_20260605_080910.mp4")

    assert recording_marker_path(video_path) == Path("recordings/fallguard_20260605_080910.json")


def test_save_and_load_recording_markers_round_trip(tmp_path):
    video_path = tmp_path / "recordings" / "fallguard_20260605_080910.mp4"
    markers = {
        "video_file": video_path.name,
        "frame_count": 3,
        "fps": 1.0,
        "duration_seconds": 3.0,
        "fall_points": [{"frame": 1, "time_seconds": 1.0, "summary": "fall"}],
        "fall_intervals": [{"start_frame": 1, "end_frame": 1, "start_seconds": 1.0, "end_seconds": 2.0}],
    }

    save_recording_markers(video_path, markers)

    assert load_recording_markers(video_path) == markers


def test_load_recording_markers_returns_none_when_sidecar_missing(tmp_path):
    video_path = tmp_path / "recordings" / "fallguard_20260605_080910.mp4"

    assert load_recording_markers(video_path) is None


def test_build_fall_intervals_merges_consecutive_fall_points():
    points = [
        {"frame": 1, "time_seconds": 0.5, "summary": "fall a"},
        {"frame": 2, "time_seconds": 1.0, "summary": "fall b"},
        {"frame": 3, "time_seconds": 1.5, "summary": "fall c"},
        {"frame": 7, "time_seconds": 3.5, "summary": "fall d"},
    ]

    assert build_fall_intervals(points, fps=2.0) == [
        {"start_frame": 1, "end_frame": 3, "start_seconds": 0.5, "end_seconds": 2.0},
        {"start_frame": 7, "end_frame": 7, "start_seconds": 3.5, "end_seconds": 4.0},
    ]


def test_append_recording_fall_point_keeps_only_fall_state():
    state = {"recording_fall_points": []}

    append_recording_fall_point(state, 0, "standing", "not fall", fps=1.0)
    append_recording_fall_point(state, 1, "fall", "detected", fps=1.0)

    assert state["recording_fall_points"] == [{"frame": 1, "time_seconds": 1.0, "summary": "detected"}]
