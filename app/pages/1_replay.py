from __future__ import annotations

from pathlib import Path

import streamlit as st

from fallguard.ui.replay import (
    load_recording_markers,
    load_recording_sessions,
    render_replay_player,
)


def format_session_label(session: dict[str, str]) -> str:
    return f"{session['开始时间']} - {session['文件名']}"


def main() -> None:
    st.set_page_config(page_title="FallGuard 录像回放", layout="wide")
    st.session_state.setdefault("recorded_sessions", [])

    st.title("录像回放")
    st.caption("播放本地保存的标注录像，并在进度条上只标记摔倒关键帧或连续摔倒区间。")

    sessions = load_recording_sessions(st.session_state.get("recorded_sessions", []))
    if st.button("刷新录像列表", use_container_width=True):
        st.rerun()

    if not sessions:
        st.info("暂无录像。请先在主监测页面使用摄像头检测并暂停保存录像。")
        return

    left, right = st.columns([1, 2], gap="large")
    with left:
        st.subheader("录像列表")
        selected_session = st.radio(
            "点击选择一个录像",
            sessions,
            format_func=format_session_label,
            label_visibility="collapsed",
        )
        st.dataframe(sessions, use_container_width=True, hide_index=True)

    with right:
        st.subheader("回放画面")
        video_path = Path(selected_session["文件路径"])
        if not video_path.exists():
            st.warning("录像文件不存在，可能已被手动删除。")
            return

        markers = load_recording_markers(video_path)
        if markers is None:
            st.info("暂无摔倒标记数据。旧录像可以播放，但不会在进度条上显示摔倒标记。")
        elif not markers.get("fall_points"):
            st.info("已读取标记数据：本段录像未记录到摔倒关键帧。")
        else:
            st.error("已读取到摔倒关键帧，红色标记表示摔倒时间点或持续区间。")

        render_replay_player(video_path, markers)


if __name__ == "__main__":
    main()
