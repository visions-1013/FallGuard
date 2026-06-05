from datetime import datetime
from pathlib import Path

from app.streamlit_app import (
    build_event,
    build_recording_session,
    calculate_processing_fps,
    recording_filename,
    release_camera_capture,
    reset_detection_state,
    should_record_frame,
    source_fps,
    status_notice,
)


class ReleasableCapture:
    def __init__(self):
        self.released = False

    def release(self):
        self.released = True


def test_status_notice_keeps_english_state_and_chinese_message():
    notice = status_notice("standing")

    assert notice["state"] == "standing"
    assert notice["title"] == "当前状态"
    assert "站立" in notice["message"]


def test_status_notice_marks_fall_as_alert():
    notice = status_notice("fall")

    assert notice["state"] == "fall"
    assert notice["level"] == "error"
    assert "检测到摔倒" in notice["message"]


def test_build_event_uses_chinese_columns_and_english_state():
    event = build_event(12, "lying", "angle=80.0")

    assert event == {"帧号": "12", "状态": "lying", "说明": "angle=80.0"}


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
