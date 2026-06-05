from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import cv2
import streamlit as st

from fallguard.pipeline import FallGuardPipeline


STATUS_MESSAGES = {
    "unknown": "未获得可靠人体姿态",
    "standing": "站立状态",
    "sitting": "坐姿状态",
    "moving": "正常移动",
    "lying": "低姿态/主动躺下",
    "fall": "检测到摔倒",
}


def status_notice(state: str) -> dict[str, str]:
    message = STATUS_MESSAGES.get(state, "未知状态")
    level = "error" if state == "fall" else "info"
    return {"title": "当前状态", "state": state, "message": message, "level": level}


def build_event(frame_index: int, state: str, summary: str) -> dict[str, str]:
    return {"帧号": str(frame_index), "状态": state, "说明": summary}


@st.cache_resource
def load_pipeline() -> FallGuardPipeline:
    return FallGuardPipeline()


def ensure_session_state() -> None:
    st.session_state.setdefault("events", [])
    st.session_state.setdefault("uploaded_video_path", "")
    st.session_state.setdefault("uploaded_video_key", "")


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


def render_status(state: str, summary: str = "") -> None:
    notice = status_notice(state)
    label = f"{notice['message']} ({notice['state']})"
    if notice["level"] == "error":
        st.error(label)
    else:
        st.info(label)
    if summary:
        st.caption(summary)


def render_history(events: list[dict[str, str]]) -> None:
    st.subheader("历史事件")
    if events:
        st.dataframe(events, use_container_width=True, hide_index=True)
    else:
        st.caption("暂无事件。开始检测后会记录状态变化。")


def run_batch_detection(
    source: int | str | Path,
    max_frames: int,
    frame_slot: Any,
    status_slot: Any,
    progress_slot: Any,
    history_slot: Any,
) -> None:
    capture = cv2.VideoCapture(str(source) if isinstance(source, Path) else source)
    if not capture.isOpened():
        status_slot.error("无法打开输入源，请先完成预览检查。")
        return

    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 30.0

    with st.spinner("正在加载 YOLO-Pose 并开始本地检测..."):
        pipeline = load_pipeline()

    events = st.session_state["events"]
    last_state = events[-1]["状态"] if events else ""

    frame_index = 0
    while frame_index < max_frames:
        ok, frame = capture.read()
        if not ok:
            break

        result = pipeline.process_frame(frame, frame_index=frame_index, fps=fps)
        annotated_rgb = frame_to_rgb(result.annotated_frame)
        frame_slot.image(annotated_rgb, channels="RGB", use_container_width=True)

        with status_slot.container():
            render_status(result.state, result.summary)

        if result.state != last_state:
            event_summary = result.summary
            if result.state == "fall":
                event_summary = f"检测到摔倒；{result.summary}"
            events.append(build_event(frame_index, result.state, event_summary))
            last_state = result.state

        with history_slot.container():
            render_history(events)

        frame_index += 1
        progress_slot.progress(frame_index / max_frames, text=f"已处理 {frame_index}/{max_frames} 帧")

    capture.release()
    if frame_index == 0:
        status_slot.warning("没有处理到有效帧。")
    else:
        progress_slot.progress(1.0, text=f"批处理完成，共处理 {frame_index} 帧。")


def render_sidebar() -> tuple[str, int, int, Any, bool, bool, bool]:
    st.sidebar.header("控制面板")
    source_mode = st.sidebar.radio("输入源", ["摄像头", "视频文件"], horizontal=True)
    max_frames = st.sidebar.slider("处理帧数", min_value=30, max_value=600, value=180, step=30)

    camera_index = 0
    uploaded_file = None
    preview_clicked = False
    start_clicked = False

    if source_mode == "摄像头":
        camera_index = st.sidebar.number_input("摄像头编号", min_value=0, max_value=10, value=0, step=1)
        preview_clicked = st.sidebar.button("预览摄像头", use_container_width=True)
    else:
        uploaded_file = st.sidebar.file_uploader(
            "上传本地视频",
            type=["mp4", "avi", "mov", "mkv"],
            help="视频只在本地处理，不会上传到外部服务。",
        )

    start_clicked = st.sidebar.button("开始检测", type="primary", use_container_width=True)
    clear_clicked = st.sidebar.button("清空结果", use_container_width=True)
    return source_mode, int(camera_index), max_frames, uploaded_file, preview_clicked, start_clicked, clear_clicked


def main() -> None:
    st.set_page_config(page_title="FallGuard 摔倒检测演示", layout="wide")
    ensure_session_state()

    st.title("FallGuard 摔倒检测演示")
    st.caption("本地视频/摄像头输入，YOLO-Pose 姿态估计，状态标签保持英文，页面交互使用中文。")

    source_mode, camera_index, max_frames, uploaded_file, preview_clicked, start_clicked, clear_clicked = render_sidebar()

    if clear_clicked:
        st.session_state["events"] = []

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("监测画面")
        frame_slot = st.empty()
        progress_slot = st.empty()

    with right:
        st.subheader("当前状态")
        status_slot = st.empty()
        history_slot = st.empty()
        with history_slot.container():
            render_history(st.session_state["events"])

    source: int | Path | None = None

    if source_mode == "摄像头":
        source = camera_index
        if preview_clicked:
            ok, frame, error = read_preview_frame(source)
            if ok:
                frame_slot.image(frame_to_rgb(frame), channels="RGB", use_container_width=True)
                status_slot.success(f"摄像头 {camera_index} 可用。确认画面后可以开始检测。")
            else:
                status_slot.error(error)
    else:
        if uploaded_file is None:
            status_slot.info("请选择左侧“上传本地视频”。上传后会先显示首帧预览。")
        else:
            source = save_uploaded_video(uploaded_file)
            st.sidebar.success(f"已选择：{uploaded_file.name}")
            st.sidebar.caption(f"文件大小：{uploaded_file.size / 1024 / 1024:.2f} MB")
            ok, frame, error = read_preview_frame(source)
            if ok:
                frame_slot.image(frame_to_rgb(frame), channels="RGB", use_container_width=True)
                status_slot.success("视频首帧读取成功。确认画面后可以开始检测。")
            else:
                status_slot.error(error)

    if start_clicked:
        if source is None:
            status_slot.warning("请先选择并预览输入源。")
            return
        progress_slot.progress(0, text="准备开始检测...")
        run_batch_detection(source, max_frames, frame_slot, status_slot, progress_slot, history_slot)


if __name__ == "__main__":
    main()
