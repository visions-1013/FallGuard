from __future__ import annotations

import base64
import html
import json
from collections.abc import MutableMapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import streamlit as st
import streamlit.components.v1 as components


SUPPORTED_RECORDING_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
BROWSER_PLAYABLE_MP4_CODECS = {"avc1", "h264"}
H264_PLAYBACK_CODECS = ("avc1", "H264", "X264")
WEBM_PLAYBACK_CODECS = ("VP80",)


def recordings_dir() -> Path:
    return Path("recordings")


def recording_filename(started_at: datetime) -> str:
    return f"fallguard_{started_at:%Y%m%d_%H%M%S}.mp4"


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


def recording_video_files() -> list[Path]:
    directory = recordings_dir()
    if not directory.exists():
        return []

    files = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_RECORDING_SUFFIXES
    ]
    return sorted(files, key=recording_file_sort_key)


def recording_file_sort_key(path: Path) -> tuple[datetime, str]:
    return parse_recording_started_at(path) or datetime.fromtimestamp(path.stat().st_mtime), path.name


def parse_recording_started_at(path: Path) -> datetime | None:
    stem = path.stem
    prefix = "fallguard_"
    if not stem.startswith(prefix):
        return None
    try:
        return datetime.strptime(stem[len(prefix) :], "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def read_video_metadata(path: Path) -> tuple[int, float]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return 0, 0.0
        frame_count = int(max(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0, 0))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        return frame_count, fps if fps > 0 else 0.0
    finally:
        capture.release()


def read_video_codec(path: Path) -> str:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return ""
        fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC) or 0)
        return "".join(chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4)).strip()
    finally:
        capture.release()


def build_recording_session_from_file(path: Path) -> dict[str, str]:
    started_at = parse_recording_started_at(path) or datetime.fromtimestamp(path.stat().st_mtime)
    frame_count, fps = read_video_metadata(path)
    duration_seconds = frame_count / fps if frame_count > 0 and fps > 0 else 0.0
    ended_at = started_at + timedelta(seconds=duration_seconds)
    return build_recording_session(path, started_at, ended_at, frame_count)


def load_recording_sessions(session_records: list[dict[str, str]]) -> list[dict[str, str]]:
    sessions_by_path = {
        session["文件路径"]: session
        for session in (build_recording_session_from_file(path) for path in recording_video_files())
    }
    for session in session_records:
        sessions_by_path[session["文件路径"]] = session

    return sorted(sessions_by_path.values(), key=lambda session: (session["开始时间"], session["文件名"]))


def recording_marker_path(video_path: Path) -> Path:
    return video_path.with_suffix(".json")


def load_recording_markers(video_path: Path) -> dict[str, Any] | None:
    marker_path = recording_marker_path(video_path)
    if not marker_path.exists():
        return None
    try:
        return json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def save_recording_markers(video_path: Path, markers: dict[str, Any]) -> None:
    marker_path = recording_marker_path(video_path)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(json.dumps(markers, ensure_ascii=False, indent=2), encoding="utf-8")


def append_recording_fall_point(
    state: MutableMapping[str, Any],
    recorded_frame_index: int,
    state_name: str,
    summary: str,
    fps: float,
) -> None:
    if state_name != "fall":
        return

    safe_fps = fps if fps > 0 else 1.0
    state.setdefault("recording_fall_points", []).append(
        {
            "frame": int(recorded_frame_index),
            "time_seconds": round(recorded_frame_index / safe_fps, 3),
            "summary": summary,
        }
    )


def build_fall_intervals(points: list[dict], fps: float) -> list[dict[str, float | int]]:
    if not points:
        return []

    safe_fps = fps if fps > 0 else 1.0
    sorted_points = sorted(points, key=lambda point: int(point["frame"]))
    intervals: list[dict[str, float | int]] = []
    start_frame = int(sorted_points[0]["frame"])
    end_frame = start_frame

    for point in sorted_points[1:]:
        frame = int(point["frame"])
        if frame <= end_frame + 1:
            end_frame = max(end_frame, frame)
            continue
        intervals.append(_build_fall_interval(start_frame, end_frame, safe_fps))
        start_frame = frame
        end_frame = frame

    intervals.append(_build_fall_interval(start_frame, end_frame, safe_fps))
    return intervals


def _build_fall_interval(start_frame: int, end_frame: int, fps: float) -> dict[str, float | int]:
    return {
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_seconds": round(start_frame / fps, 3),
        "end_seconds": round((end_frame + 1) / fps, 3),
    }


def resolve_replay_video_path(video_path: Path) -> Path:
    if is_browser_playable_video(video_path):
        return video_path

    cached_path = fresh_playback_cache_path(video_path)
    if cached_path is not None:
        return cached_path

    return transcode_for_browser_playback(video_path)


def is_browser_playable_video(video_path: Path) -> bool:
    suffix = video_path.suffix.lower()
    if suffix == ".webm":
        return True
    if suffix != ".mp4":
        return False
    return read_video_codec(video_path).lower() in BROWSER_PLAYABLE_MP4_CODECS


def playback_cache_path(video_path: Path, suffix: str = ".mp4") -> Path:
    return video_path.parent / ".playback" / f"{video_path.stem}{suffix}"


def fresh_playback_cache_path(video_path: Path) -> Path | None:
    try:
        source_mtime = video_path.stat().st_mtime
    except OSError:
        return None

    for suffix in (".mp4", ".webm"):
        candidate = playback_cache_path(video_path, suffix)
        try:
            if candidate.exists() and candidate.stat().st_size > 0 and candidate.stat().st_mtime >= source_mtime:
                return candidate
        except OSError:
            continue
    return None


def transcode_for_browser_playback(video_path: Path) -> Path:
    mp4_path = playback_cache_path(video_path, ".mp4")
    if transcode_video(video_path, mp4_path, H264_PLAYBACK_CODECS):
        return mp4_path

    webm_path = playback_cache_path(video_path, ".webm")
    if transcode_video(video_path, webm_path, WEBM_PLAYBACK_CODECS):
        return webm_path

    return video_path


def transcode_video(source_path: Path, target_path: Path, codecs: tuple[str, ...]) -> bool:
    capture = cv2.VideoCapture(str(source_path))
    try:
        if not capture.isOpened():
            return False

        ok, first_frame = capture.read()
        if not ok:
            return False

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = 30.0
        height, width = first_frame.shape[:2]
        target_path.parent.mkdir(parents=True, exist_ok=True)

        for codec in codecs:
            target_path.unlink(missing_ok=True)
            writer = cv2.VideoWriter(
                str(target_path),
                cv2.VideoWriter_fourcc(*codec),
                fps,
                (width, height),
            )
            if not writer.isOpened():
                writer.release()
                continue

            writer.write(first_frame)
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                writer.write(frame)
            writer.release()
            if target_path.exists() and target_path.stat().st_size > 0:
                return True
    finally:
        capture.release()

    target_path.unlink(missing_ok=True)
    return False


def render_replay_player(video_path: Path, markers: dict | None) -> None:
    if not video_path.exists():
        st.warning("录像文件不存在，可能已被手动删除。")
        return

    replay_video_path = resolve_replay_video_path(video_path)
    video_data = base64.b64encode(replay_video_path.read_bytes()).decode("ascii")
    mime_type = video_mime_type(replay_video_path)
    duration = marker_duration_seconds(video_path, markers)
    intervals = markers.get("fall_intervals", []) if markers else []
    timeline_marks = build_timeline_marks(intervals, duration)
    metadata_notice = "已读取摔倒标记数据。" if markers is not None else "暂无摔倒标记数据。"
    fall_count = len(markers.get("fall_points", [])) if markers else 0
    interval_count = len(intervals)
    safe_title = html.escape(video_path.name)
    safe_notice = html.escape(metadata_notice)

    player_html = f"""
<style>
  .fallguard-player {{
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: #172033;
  }}
  .fallguard-player video {{
    width: 100%;
    max-height: 62vh;
    background: #080b12;
    border-radius: 8px;
  }}
  .replay-meta {{
    display: flex;
    justify-content: space-between;
    gap: 12px;
    margin: 10px 0 8px;
    font-size: 14px;
    color: #475569;
  }}
  .timeline {{
    position: relative;
    height: 28px;
    border-radius: 999px;
    background: #d8dee9;
    cursor: pointer;
    overflow: hidden;
  }}
  .timeline-progress {{
    position: absolute;
    inset: 0 auto 0 0;
    width: 0%;
    background: #334155;
    opacity: 0.38;
  }}
  .fall-interval {{
    position: absolute;
    top: 0;
    bottom: 0;
    background: rgba(220, 38, 38, 0.82);
    box-shadow: 0 0 0 1px rgba(127, 29, 29, 0.2);
  }}
  .fall-point {{
    position: absolute;
    top: 0;
    bottom: 0;
    width: 4px;
    transform: translateX(-2px);
    border-radius: 999px;
    background: #dc2626;
  }}
  .time-row {{
    display: flex;
    justify-content: space-between;
    margin-top: 6px;
    font-size: 13px;
    color: #64748b;
  }}
</style>
<div class="fallguard-player">
  <video id="replay-video" controls preload="metadata">
    <source src="data:{mime_type};base64,{video_data}" type="{mime_type}">
  </video>
  <div class="replay-meta">
    <strong>{safe_title}</strong>
    <span>{safe_notice} 摔倒点 {fall_count} 个，区间 {interval_count} 段。</span>
  </div>
  <div id="timeline" class="timeline" title="点击跳转回放位置">
    <div id="timeline-progress" class="timeline-progress"></div>
    {timeline_marks}
  </div>
  <div class="time-row">
    <span id="current-time">0.0s</span>
    <span>{duration:.1f}s</span>
  </div>
</div>
<script>
  const video = document.getElementById("replay-video");
  const timeline = document.getElementById("timeline");
  const progress = document.getElementById("timeline-progress");
  const currentTime = document.getElementById("current-time");
  const fallbackDuration = {duration:.6f};

  function durationSeconds() {{
    return Number.isFinite(video.duration) && video.duration > 0 ? video.duration : fallbackDuration;
  }}

  function updateTimeline() {{
    const duration = durationSeconds();
    const percent = duration > 0 ? Math.min((video.currentTime / duration) * 100, 100) : 0;
    progress.style.width = `${{percent}}%`;
    currentTime.textContent = `${{video.currentTime.toFixed(1)}}s`;
  }}

  video.addEventListener("timeupdate", updateTimeline);
  video.addEventListener("loadedmetadata", updateTimeline);
  timeline.addEventListener("click", (event) => {{
    const rect = timeline.getBoundingClientRect();
    const ratio = Math.min(Math.max((event.clientX - rect.left) / rect.width, 0), 1);
    video.currentTime = ratio * durationSeconds();
    updateTimeline();
  }});
</script>
"""
    components.html(player_html, height=620, scrolling=False)


def video_mime_type(video_path: Path) -> str:
    suffix = video_path.suffix.lower()
    if suffix == ".webm":
        return "video/webm"
    if suffix == ".mov":
        return "video/quicktime"
    if suffix == ".mkv":
        return "video/x-matroska"
    if suffix == ".avi":
        return "video/x-msvideo"
    return "video/mp4"


def marker_duration_seconds(video_path: Path, markers: dict | None) -> float:
    if markers is not None:
        duration = float(markers.get("duration_seconds") or 0.0)
        if duration > 0:
            return duration

    frame_count, fps = read_video_metadata(video_path)
    if frame_count > 0 and fps > 0:
        return frame_count / fps
    return 0.0


def build_timeline_marks(intervals: list[dict], duration_seconds: float) -> str:
    if duration_seconds <= 0:
        return ""

    marks = []
    for interval in intervals:
        start_seconds = float(interval.get("start_seconds", 0.0))
        end_seconds = float(interval.get("end_seconds", start_seconds))
        start_percent = max(min((start_seconds / duration_seconds) * 100, 100), 0)
        width_percent = max(min(((end_seconds - start_seconds) / duration_seconds) * 100, 100), 0)
        if int(interval.get("start_frame", -1)) == int(interval.get("end_frame", -2)):
            marks.append(f'<div class="fall-point" style="left:{start_percent:.4f}%"></div>')
        else:
            marks.append(
                '<div class="fall-interval" '
                f'style="left:{start_percent:.4f}%;width:{max(width_percent, 0.35):.4f}%"></div>'
            )
    return "\n    ".join(marks)
