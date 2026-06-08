from __future__ import annotations

import time
from collections.abc import MutableMapping
from datetime import datetime
import tempfile
from pathlib import Path
from typing import Any

import cv2
import streamlit as st

from fallguard.pipeline import FallGuardPipeline
from fallguard.ui.replay import (
    append_recording_fall_point,
    build_fall_intervals,
    build_recording_session,
    recording_filename,
    recordings_dir,
    save_recording_markers,
)


RECORDING_SAMPLE_INTERVAL_SECONDS = 1.0
BROWSER_FRIENDLY_RECORDING_CODECS = ("avc1", "H264", "X264", "mp4v")


def build_event(frame_index: int, state: str, summary: str) -> dict[str, str]:
    return {"帧号": str(frame_index), "状态": state, "说明": summary}


def should_record_frame(now: float, last_recorded_at: float, interval_seconds: float) -> bool:
    return interval_seconds <= 0 or last_recorded_at <= 0 or now - last_recorded_at >= interval_seconds


@st.cache_resource
def load_pipeline() -> FallGuardPipeline:
    return FallGuardPipeline()


def ensure_pipeline_loaded(message_slot: Any, loader: Any = load_pipeline) -> FallGuardPipeline:
    message_slot.info("正在加载 YOLO-Pose，请稍候...")
    return loader()


def should_show_preview_message(running: bool) -> bool:
    return not running


def ensure_session_state() -> None:
    st.session_state.setdefault("events", [])
    st.session_state.setdefault("recorded_sessions", [])
    st.session_state.setdefault("uploaded_video_path", "")
    st.session_state.setdefault("uploaded_video_key", "")
    st.session_state.setdefault("running", False)
    st.session_state.setdefault("frame_index", 0)
    st.session_state.setdefault("input_fps", 0.0)
    st.session_state.setdefault("processing_fps", 0.0)
    st.session_state.setdefault("last_frame_rgb", None)
    st.session_state.setdefault("source_signature", "")
    st.session_state.setdefault("video_finished", False)
    st.session_state.setdefault("recording_writer", None)
    st.session_state.setdefault("recording_path", "")
    st.session_state.setdefault("recording_started_at", None)
    st.session_state.setdefault("recording_last_saved_at", 0.0)
    st.session_state.setdefault("recorded_frame_count", 0)
    st.session_state.setdefault("recording_output_fps", 0.0)
    st.session_state.setdefault("recording_sample_interval_seconds", RECORDING_SAMPLE_INTERVAL_SECONDS)
    st.session_state.setdefault("recording_fall_points", [])
    st.session_state.setdefault("camera_capture", None)
    st.session_state.setdefault("camera_source_index", None)


def calculate_processing_fps(elapsed_seconds: float) -> float:
    if elapsed_seconds <= 0:
        return 0.0
    return round(1.0 / elapsed_seconds, 2)


def source_fps(capture: Any, fallback: float = 30.0) -> float:
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    return fps if fps > 0 else fallback


def reset_detection_state(state: MutableMapping[str, Any] | None = None) -> None:
    target = st.session_state if state is None else state
    release_recording_writer(target)
    release_camera_capture(target)
    target["events"] = []
    target["running"] = False
    target["frame_index"] = 0
    target["input_fps"] = 0.0
    target["processing_fps"] = 0.0
    target["last_frame_rgb"] = None
    target["video_finished"] = False
    target["recording_writer"] = None
    target["recording_path"] = ""
    target["recording_started_at"] = None
    target["recording_last_saved_at"] = 0.0
    target["recorded_frame_count"] = 0
    target["recording_output_fps"] = 0.0
    target["recording_sample_interval_seconds"] = RECORDING_SAMPLE_INTERVAL_SECONDS
    target["recording_fall_points"] = []
    target["camera_capture"] = None
    target["camera_source_index"] = None


def reset_runtime_for_new_source(source_signature: str) -> None:
    if st.session_state.get("source_signature") == source_signature:
        return
    reset_detection_state()
    st.session_state["source_signature"] = source_signature


def release_recording_writer(state: MutableMapping[str, Any] | None = None) -> None:
    target = st.session_state if state is None else state
    writer = target.get("recording_writer")
    if writer is not None:
        writer.release()
        target["recording_writer"] = None


def release_camera_capture(state: MutableMapping[str, Any] | None = None) -> None:
    target = st.session_state if state is None else state
    capture = target.get("camera_capture")
    if capture is not None:
        capture.release()
    target["camera_capture"] = None
    target["camera_source_index"] = None


def open_camera_capture(camera_index: int) -> Any | None:
    existing = st.session_state.get("camera_capture")
    existing_index = st.session_state.get("camera_source_index")
    if existing is not None and existing_index == camera_index and existing.isOpened():
        return existing

    release_camera_capture()
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        capture.release()
        return None

    st.session_state["camera_capture"] = capture
    st.session_state["camera_source_index"] = camera_index
    return capture


def default_recording_fps() -> float:
    return max(1.0 / RECORDING_SAMPLE_INTERVAL_SECONDS, 1.0)


def create_recording_writer(path_text: str, output_fps: float, size: tuple[int, int]) -> Any | None:
    for codec in BROWSER_FRIENDLY_RECORDING_CODECS:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(path_text, fourcc, output_fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    return None


def start_recording_session(
    output_fps: float | None = None,
    sample_interval_seconds: float = RECORDING_SAMPLE_INTERVAL_SECONDS,
) -> None:
    started_at = datetime.now()
    output_dir = recordings_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / recording_filename(started_at)
    safe_output_fps = float(output_fps or 0.0)
    if safe_output_fps <= 0:
        safe_output_fps = default_recording_fps()
    st.session_state["recording_writer"] = None
    st.session_state["recording_path"] = str(path)
    st.session_state["recording_started_at"] = started_at
    st.session_state["recording_last_saved_at"] = 0.0
    st.session_state["recorded_frame_count"] = 0
    st.session_state["recording_output_fps"] = safe_output_fps
    st.session_state["recording_sample_interval_seconds"] = float(sample_interval_seconds)
    st.session_state["recording_fall_points"] = []


def finalize_recording_session() -> None:
    release_recording_writer()
    path_text = st.session_state.get("recording_path", "")
    started_at = st.session_state.get("recording_started_at")
    frame_count = int(st.session_state.get("recorded_frame_count", 0))
    if path_text and started_at is not None and frame_count > 0:
        path = Path(path_text)
        output_fps = float(st.session_state.get("recording_output_fps") or 0.0)
        if output_fps <= 0:
            output_fps = default_recording_fps()
        fall_points = list(st.session_state.get("recording_fall_points", []))
        markers = {
            "video_file": path.name,
            "frame_count": frame_count,
            "fps": output_fps,
            "duration_seconds": round(frame_count / output_fps, 3),
            "fall_points": fall_points,
            "fall_intervals": build_fall_intervals(fall_points, output_fps),
        }
        save_recording_markers(path, markers)
        session = build_recording_session(path, started_at, datetime.now(), frame_count)
        st.session_state["recorded_sessions"].append(session)
    st.session_state["recording_path"] = ""
    st.session_state["recording_started_at"] = None
    st.session_state["recording_last_saved_at"] = 0.0
    st.session_state["recorded_frame_count"] = 0
    st.session_state["recording_output_fps"] = 0.0
    st.session_state["recording_sample_interval_seconds"] = RECORDING_SAMPLE_INTERVAL_SECONDS
    st.session_state["recording_fall_points"] = []


def record_annotated_frame(frame: Any, state: str = "", summary: str = "") -> None:
    path_text = st.session_state.get("recording_path", "")
    if not path_text:
        return

    now = time.perf_counter()
    last_saved_at = float(st.session_state.get("recording_last_saved_at", 0.0))
    sample_interval_seconds = float(
        st.session_state.get("recording_sample_interval_seconds", RECORDING_SAMPLE_INTERVAL_SECONDS)
    )
    if not should_record_frame(now, last_saved_at, sample_interval_seconds):
        return

    output_fps = float(st.session_state.get("recording_output_fps") or 0.0)
    if output_fps <= 0:
        output_fps = default_recording_fps()
    writer = st.session_state.get("recording_writer")
    if writer is None:
        height, width = frame.shape[:2]
        Path(path_text).parent.mkdir(parents=True, exist_ok=True)
        writer = create_recording_writer(path_text, output_fps, (width, height))
        if writer is None:
            st.session_state["recording_writer"] = None
            return
        st.session_state["recording_writer"] = writer

    recorded_frame_index = int(st.session_state.get("recorded_frame_count", 0))
    writer.write(frame)
    append_recording_fall_point(st.session_state, recorded_frame_index, state, summary, output_fps)
    st.session_state["recording_last_saved_at"] = now
    st.session_state["recorded_frame_count"] = recorded_frame_index + 1


def video_source_recording_fps(source: Path) -> float:
    capture = cv2.VideoCapture(str(source))
    try:
        return source_fps(capture)
    finally:
        capture.release()


def save_uploaded_video(uploaded_file: Any) -> Path:
    content = uploaded_file.getbuffer()
    upload_key = f"{uploaded_file.name}:{len(content)}"
    existing_path = st.session_state.get("uploaded_video_path", "")

    if st.session_state.get("uploaded_video_key") == upload_key and existing_path:
        return Path(existing_path)

    if existing_path:
        Path(existing_path).unlink(missing_ok=True)

    suffix = Path(uploaded_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        saved_path = Path(tmp.name)

    st.session_state["uploaded_video_key"] = upload_key
    st.session_state["uploaded_video_path"] = str(saved_path)
    return saved_path


def read_preview_frame(source: int | str | Path) -> tuple[bool, Any, str]:
    capture = cv2.VideoCapture(str(source) if isinstance(source, Path) else source)
    if not capture.isOpened():
        return False, None, "无法打开输入源，请检查摄像头编号、权限或视频文件。"
    ok, frame = capture.read()
    capture.release()
    if not ok:
        return False, None, "输入源已打开，但没有读取到有效画面。"
    return True, frame, ""


def frame_to_rgb(frame: Any) -> Any:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def render_detection_messages() -> Any:
    return st.empty()


def render_history(events: list[dict[str, str]]) -> None:
    st.subheader("历史事件")
    if events:
        st.dataframe(events, use_container_width=True, hide_index=True)
    else:
        st.caption("暂无事件。开始检测后会记录状态变化。")


def render_metrics(metric_slot: Any) -> None:
    input_fps = st.session_state.get("input_fps", 0.0)
    processing_fps = st.session_state.get("processing_fps", 0.0)
    frame_index = st.session_state.get("frame_index", 0)

    with metric_slot.container():
        col1, col2, col3 = st.columns(3)
        col1.metric("输入 FPS", f"{input_fps:.1f}" if input_fps > 0 else "未知")
        col2.metric("处理 FPS", f"{processing_fps:.1f}" if processing_fps > 0 else "等待中")
        col3.metric("已处理帧数", str(frame_index))


def process_next_frame(
    source: int | str | Path,
    frame_slot: Any,
    message_slot: Any,
    metric_slot: Any,
    history_slot: Any,
) -> None:
    is_video_file = isinstance(source, Path)
    if is_video_file:
        capture = cv2.VideoCapture(str(source))
    else:
        capture = open_camera_capture(int(source))

    if capture is None:
        message_slot.error("无法打开输入源，请先完成预览检查。")
        st.session_state["running"] = False
        finalize_recording_session()
        return

    if not capture.isOpened():
        message_slot.error("无法打开输入源，请先完成预览检查。")
        st.session_state["running"] = False
        finalize_recording_session()
        if not is_video_file:
            release_camera_capture()
        return

    frame_index = int(st.session_state.get("frame_index", 0))
    if is_video_file:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    fps = source_fps(capture)
    st.session_state["input_fps"] = fps

    ok, frame = capture.read()
    if is_video_file:
        capture.release()
    if not ok:
        finalize_recording_session()
        if not is_video_file:
            release_camera_capture()
        st.session_state["running"] = False
        st.session_state["video_finished"] = is_video_file
        message = "视频已播放完。" if is_video_file else "没有读取到有效帧。"
        message_slot.info(message)
        render_metrics(metric_slot)
        return

    pipeline = load_pipeline()

    started_at = time.perf_counter()
    result = pipeline.process_frame(frame, frame_index=frame_index, fps=fps)
    elapsed = time.perf_counter() - started_at

    st.session_state["processing_fps"] = calculate_processing_fps(elapsed)
    annotated_rgb = frame_to_rgb(result.annotated_frame)
    record_annotated_frame(result.annotated_frame, state=result.state, summary=result.summary)
    st.session_state["last_frame_rgb"] = annotated_rgb
    frame_slot.image(annotated_rgb, channels="RGB", use_container_width=True)

    events = st.session_state["events"]
    last_state = events[-1]["状态"] if events else ""
    if result.state != last_state:
        event_summary = result.summary
        if result.state == "fall":
            event_summary = f"检测到摔倒；{result.summary}"
        events.append(build_event(frame_index, result.state, event_summary))

    st.session_state["frame_index"] = frame_index + 1
    with history_slot.container():
        render_history(events)
    render_metrics(metric_slot)

    if st.session_state.get("running", False):
        st.rerun()


def render_sidebar() -> tuple[str, int, Any, bool, bool, bool, bool]:
    st.sidebar.header("控制面板")
    source_mode = st.sidebar.radio("输入源", ["摄像头", "视频文件"], horizontal=True)

    camera_index = 0
    uploaded_file = None
    preview_clicked = False
    start_clicked = False
    pause_clicked = False

    if source_mode == "摄像头":
        camera_index = st.sidebar.number_input("摄像头编号", min_value=0, max_value=10, value=0, step=1)
        preview_clicked = st.sidebar.button("预览摄像头", use_container_width=True)
    else:
        uploaded_file = st.sidebar.file_uploader(
            "上传本地视频",
            type=["mp4", "avi", "mov", "mkv"],
            help="视频只在本地处理，不会上传到外部服务。",
        )

    running = st.session_state.get("running", False)
    start_clicked = st.sidebar.button("开始检测", type="primary", use_container_width=True, disabled=running)
    pause_clicked = st.sidebar.button("暂停检测", use_container_width=True, disabled=not running)
    clear_clicked = st.sidebar.button("清空结果", use_container_width=True)
    return source_mode, int(camera_index), uploaded_file, preview_clicked, start_clicked, pause_clicked, clear_clicked


def main() -> None:
    st.set_page_config(page_title="FallGuard 摔倒检测演示", layout="wide")
    ensure_session_state()

    st.title("FallGuard 摔倒检测演示")
    st.caption("本地视频/摄像头输入，YOLO-Pose 姿态估计，状态标签保持英文，页面交互使用中文。")

    source_mode, camera_index, uploaded_file, preview_clicked, start_clicked, pause_clicked, clear_clicked = render_sidebar()
    running = st.session_state.get("running", False)

    if clear_clicked:
        reset_detection_state()
    if pause_clicked:
        finalize_recording_session()
        release_camera_capture()
        st.session_state["running"] = False

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("监测画面")
        frame_slot = st.empty()
        if st.session_state.get("last_frame_rgb") is not None:
            frame_slot.image(st.session_state["last_frame_rgb"], channels="RGB", use_container_width=True)

    with right:
        message_slot = render_detection_messages()
        metric_slot = st.empty()
        render_metrics(metric_slot)
        history_slot = st.empty()
        with history_slot.container():
            render_history(st.session_state["events"])

    source: int | Path | None = None

    if source_mode == "摄像头":
        source = camera_index
        reset_runtime_for_new_source(f"camera:{camera_index}")
        if preview_clicked and should_show_preview_message(running) and not start_clicked:
            ok, frame, error = read_preview_frame(source)
            if ok:
                frame_slot.image(frame_to_rgb(frame), channels="RGB", use_container_width=True)
                message_slot.success(f"摄像头 {camera_index} 可用。确认画面后可以开始检测。")
            else:
                message_slot.error(error)
    else:
        if uploaded_file is None:
            if should_show_preview_message(running):
                message_slot.info("请选择左侧“上传本地视频”。上传后会先显示首帧预览。")
        else:
            source = save_uploaded_video(uploaded_file)
            reset_runtime_for_new_source(f"video:{st.session_state['uploaded_video_key']}")
            if should_show_preview_message(running) and not start_clicked:
                st.sidebar.success(f"已选择：{uploaded_file.name}")
                st.sidebar.caption(f"文件大小：{uploaded_file.size / 1024 / 1024:.2f} MB")
                ok, frame, error = read_preview_frame(source)
                if ok:
                    frame_slot.image(frame_to_rgb(frame), channels="RGB", use_container_width=True)
                    message_slot.success("视频首帧读取成功。确认画面后可以开始检测。")
                else:
                    message_slot.error(error)

    if start_clicked:
        if source is None:
            message_slot.warning("请先选择并预览输入源。")
            return
        if source_mode == "摄像头":
            if open_camera_capture(camera_index) is None:
                message_slot.error("无法打开摄像头，请检查编号或权限。")
                st.session_state["running"] = False
                return
            start_recording_session()
        else:
            start_recording_session(output_fps=video_source_recording_fps(source), sample_interval_seconds=0.0)
        ensure_pipeline_loaded(message_slot)
        st.session_state["running"] = True
        st.session_state["video_finished"] = False

    if st.session_state.get("video_finished", False):
        message_slot.info("视频已播放完。")

    if st.session_state.get("running", False):
        if source is None:
            st.session_state["running"] = False
            message_slot.warning("请先选择并预览输入源。")
            return
        process_next_frame(source, frame_slot, message_slot, metric_slot, history_slot)


if __name__ == "__main__":
    main()
