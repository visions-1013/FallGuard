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


RECORDING_SAMPLE_INTERVAL_SECONDS = 1.0


def build_event(frame_index: int, state: str, summary: str) -> dict[str, str]:
    return {"帧号": str(frame_index), "状态": state, "说明": summary}


def recordings_dir() -> Path:
    return Path("recordings")


def recording_filename(started_at: datetime) -> str:
    return f"fallguard_{started_at:%Y%m%d_%H%M%S}.mp4"


def should_record_frame(now: float, last_recorded_at: float, interval_seconds: float) -> bool:
    return interval_seconds <= 0 or last_recorded_at <= 0 or now - last_recorded_at >= interval_seconds


def build_recording_session(
    path: Path,
    started_at: datetime,
    ended_at: datetime,
    frame_count: int,
) -> dict[str, str]:
    duration = max((ended_at - started_at).total_seconds(), 0.0)
    return {
        "开始时间": started_at.strftime("%Y-%m-%d %H:%M:%S"),
        "结束时间": ended_at.strftime("%Y-%m-%d %H:%M:%S"),
        "时长(秒)": f"{duration:.1f}",
        "帧数": str(frame_count),
        "文件名": path.name,
        "文件路径": str(path),
    }


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


def start_recording_session() -> None:
    started_at = datetime.now()
    output_dir = recordings_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / recording_filename(started_at)
    st.session_state["recording_writer"] = None
    st.session_state["recording_path"] = str(path)
    st.session_state["recording_started_at"] = started_at
    st.session_state["recording_last_saved_at"] = 0.0
    st.session_state["recorded_frame_count"] = 0


def finalize_recording_session() -> None:
    release_recording_writer()
    path_text = st.session_state.get("recording_path", "")
    started_at = st.session_state.get("recording_started_at")
    frame_count = int(st.session_state.get("recorded_frame_count", 0))
    if path_text and started_at is not None and frame_count > 0:
        session = build_recording_session(Path(path_text), started_at, datetime.now(), frame_count)
        st.session_state["recorded_sessions"].append(session)
    st.session_state["recording_path"] = ""
    st.session_state["recording_started_at"] = None
    st.session_state["recording_last_saved_at"] = 0.0
    st.session_state["recorded_frame_count"] = 0


def record_annotated_frame(frame: Any) -> None:
    path_text = st.session_state.get("recording_path", "")
    if not path_text:
        return

    now = time.perf_counter()
    last_saved_at = float(st.session_state.get("recording_last_saved_at", 0.0))
    if not should_record_frame(now, last_saved_at, RECORDING_SAMPLE_INTERVAL_SECONDS):
        return

    writer = st.session_state.get("recording_writer")
    if writer is None:
        height, width = frame.shape[:2]
        output_fps = max(1.0 / RECORDING_SAMPLE_INTERVAL_SECONDS, 1.0)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path_text, fourcc, output_fps, (width, height))
        if not writer.isOpened():
            st.session_state["recording_writer"] = None
            return
        st.session_state["recording_writer"] = writer

    writer.write(frame)
    st.session_state["recording_last_saved_at"] = now
    st.session_state["recorded_frame_count"] = int(st.session_state.get("recorded_frame_count", 0)) + 1


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


def render_recordings() -> None:
    st.subheader("录像回放")
    sessions = st.session_state.get("recorded_sessions", [])
    if st.button("清空录像历史", use_container_width=True, disabled=not sessions):
        st.session_state["recorded_sessions"] = []
        st.rerun()

    if not sessions:
        st.caption("暂无录像。摄像头开始检测后，暂停时会保存本段标注视频。")
        return

    st.dataframe(sessions, use_container_width=True, hide_index=True)
    for index, session in enumerate(reversed(sessions), start=1):
        title = f"{session['开始时间']} - {session['文件名']}"
        with st.expander(title):
            st.caption(f"时长 {session['时长(秒)']} 秒，采样帧数 {session['帧数']}。")
            path = Path(session["文件路径"])
            if path.exists():
                st.video(path)
            else:
                st.warning("录像文件不存在，可能已被手动删除。")


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
        if not is_video_file:
            finalize_recording_session()
            release_camera_capture()
        return

    frame_index = int(st.session_state.get("frame_index", 0))
    if is_video_file:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    fps = source_fps(capture)
    st.session_state["input_fps"] = fps

    pipeline = load_pipeline()

    ok, frame = capture.read()
    if is_video_file:
        capture.release()
    if not ok:
        if not is_video_file:
            finalize_recording_session()
            release_camera_capture()
        st.session_state["running"] = False
        st.session_state["video_finished"] = is_video_file
        message = "视频已播放完。" if is_video_file else "没有读取到有效帧。"
        message_slot.info(message)
        render_metrics(metric_slot)
        return

    started_at = time.perf_counter()
    result = pipeline.process_frame(frame, frame_index=frame_index, fps=fps)
    elapsed = time.perf_counter() - started_at

    st.session_state["processing_fps"] = calculate_processing_fps(elapsed)
    annotated_rgb = frame_to_rgb(result.annotated_frame)
    if not is_video_file:
        record_annotated_frame(result.annotated_frame)
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
        history_tab, recording_tab = st.tabs(["状态历史", "录像回放"])
        with history_tab:
            history_slot = st.empty()
            with history_slot.container():
                render_history(st.session_state["events"])
        with recording_tab:
            render_recordings()

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
