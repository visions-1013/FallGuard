from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import cv2
from PIL import Image, ImageTk

from fallguard.inference.pipeline import FallGuardPipeline
from fallguard.models.runtime import load_runtime_bundle
from fallguard.pose.yolo_pose import YoloPoseExtractor

from .history import (
    HistoryRecord,
    create_recording_dir,
    load_event_intervals,
    load_history_records,
)
from .timeline import EventTimeline
from .worker import PreviewUpdate, VideoWorker

BG = "#0b1020"
PANEL = "#11182a"
SURFACE = "#080d18"
TEXT = "#f4f7fb"
MUTED = "#8e9bb2"
BLUE = "#2f80ed"
GREEN = "#38d39f"
RED = "#ef4444"
DIVIDER = "#243047"


class FixedNavItem(tk.Frame):
    def __init__(
        self,
        master: tk.Misc,
        text: str,
        command: Any,
    ) -> None:
        super().__init__(
            master,
            width=112,
            height=44,
            bg=BG,
            cursor="hand2",
            highlightthickness=0,
        )
        self.pack_propagate(False)
        self.label = tk.Label(
            self,
            text=text,
            bg=BG,
            fg=MUTED,
            font=("Microsoft YaHei UI", 10),
            cursor="hand2",
        )
        self.label.pack(fill="both", expand=True)
        self.indicator = tk.Frame(self, bg=BG, height=2)
        self.indicator.place(x=0, rely=1.0, anchor="sw", relwidth=1.0)
        self.bind("<Button-1>", lambda _event: command())
        self.label.bind("<Button-1>", lambda _event: command())

    def set_selected(self, selected: bool) -> None:
        self.label.configure(fg=TEXT if selected else MUTED)
        self.indicator.configure(bg=BLUE if selected else BG)


class FallGuardApp:
    def __init__(
        self,
        root: tk.Tk,
        *,
        pipeline: Any,
        recordings_dir: Path,
        model_status: str,
        experimental_model: bool,
    ) -> None:
        self.root = root
        self.pipeline = pipeline
        self.recordings_dir = recordings_dir
        self.worker: VideoWorker | None = None
        self.history_records: dict[str, HistoryRecord] = {}
        self._live_photo: ImageTk.PhotoImage | None = None
        self._history_photo: ImageTk.PhotoImage | None = None
        self._live_duration = 0.0
        self._live_intervals: list[tuple[float, float]] = []
        self._active_fall_start: float | None = None
        self._history_capture: cv2.VideoCapture | None = None
        self._history_playing = False
        self._history_after_id: str | None = None
        self._history_fps = 25.0
        self._history_duration = 0.0
        self._history_intervals: list[tuple[float, float]] = []
        self.current_page = "monitor"

        self.root.title("FallGuard 跌倒监测后台")
        self.root.geometry("1120x720")
        self.root.minsize(980, 640)
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

        self.input_var = tk.StringVar()
        self.status_var = tk.StringVar(value="等待选择监控视频")
        self.history_info_var = tk.StringVar(value="选择一条历史记录进行回放")
        self.stat_vars = {
            "state": tk.StringVar(value="待机"),
            "probability": tk.StringVar(value="--"),
            "progress": tk.StringVar(value="0%"),
            "events": tk.StringVar(value="0"),
        }
        self._configure_styles()
        self._build(model_status, experimental_model)
        self.refresh_history()

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(
            "History.Treeview",
            background=PANEL,
            fieldbackground=PANEL,
            foreground=TEXT,
            rowheight=30,
            borderwidth=0,
        )
        style.layout("History.Treeview", [("Treeview.treearea", {"sticky": "nswe"})])
        style.configure(
            "History.Treeview.Heading",
            background="#18223a",
            foreground=MUTED,
            relief="flat",
            padding=8,
        )
        style.map("History.Treeview", background=[("selected", "#1c4f91")])

    def _build(self, model_status: str, experimental_model: bool) -> None:
        header = tk.Frame(self.root, bg=BG, height=64, padx=24)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(
            header,
            text="FallGuard",
            bg=BG,
            fg=TEXT,
            font=("Microsoft YaHei UI", 20, "bold"),
        ).pack(side="left", fill="y")
        tk.Label(
            header,
            text="跌倒监测后台",
            bg=BG,
            fg=MUTED,
            font=("Microsoft YaHei UI", 10),
        ).pack(side="left", fill="y", padx=(12, 0), pady=(5, 0))
        tk.Label(
            header,
            text=f"●  {model_status}",
            bg=BG,
            fg=GREEN,
            anchor="e",
            font=("Microsoft YaHei UI", 10),
        ).pack(side="right", fill="y", pady=(4, 0))

        nav_bar = tk.Frame(self.root, bg=BG, height=44, padx=24)
        nav_bar.pack(fill="x")
        nav_bar.pack_propagate(False)
        self.nav_items = {
            "monitor": FixedNavItem(nav_bar, "实时检测", lambda: self.show_page("monitor")),
            "history": FixedNavItem(nav_bar, "历史记录", lambda: self.show_page("history")),
        }
        for item in self.nav_items.values():
            item.pack(side="left")
        tk.Frame(self.root, bg=DIVIDER, height=1).pack(fill="x", padx=24)

        self.content_container = tk.Frame(self.root, bg=BG)
        self.content_container.pack(fill="both", expand=True, padx=24, pady=(0, 8))
        self.content_container.grid_columnconfigure(0, weight=1)
        self.content_container.grid_rowconfigure(0, weight=1)
        self.monitor_page = tk.Frame(self.content_container, bg=BG)
        self.history_page = tk.Frame(self.content_container, bg=BG)
        self.pages = {"monitor": self.monitor_page, "history": self.history_page}
        for page in self.pages.values():
            page.grid(row=0, column=0, sticky="nsew")
        self._build_monitor_page()
        self._build_history_page()
        self.show_page("monitor")

        footer_text = "实验模型 · 结果仅供课程实验" if experimental_model else "模型已通过部署门槛"
        tk.Label(
            self.root,
            text=footer_text,
            bg=BG,
            fg=MUTED,
            font=("Microsoft YaHei UI", 8),
        ).pack(pady=(0, 8))

    def show_page(self, name: str) -> None:
        if name not in self.pages:
            raise ValueError(f"unknown page: {name}")
        self.current_page = name
        self.pages[name].tkraise()
        for key, item in self.nav_items.items():
            item.set_selected(key == name)
        if name == "history":
            self.refresh_history()

    def _build_monitor_page(self) -> None:
        page = self.monitor_page
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(0, weight=1)
        workspace = tk.Frame(page, bg=BG)
        workspace.grid(row=0, column=0, sticky="nsew", pady=(14, 10))
        workspace.grid_columnconfigure(0, weight=1)
        workspace.grid_columnconfigure(1, minsize=245)
        workspace.grid_rowconfigure(0, weight=1)

        self.preview_label = tk.Label(
            workspace,
            text="选择视频后开始检测",
            bg=SURFACE,
            fg=MUTED,
            font=("Microsoft YaHei UI", 13),
        )
        self.preview_label.grid(row=0, column=0, sticky="nsew")

        stats = tk.Frame(workspace, bg=PANEL, padx=20, pady=16)
        stats.grid(row=0, column=1, sticky="ns", padx=(16, 0))
        labels = [
            ("state", "当前状态"),
            ("probability", "跌倒概率"),
            ("progress", "处理进度"),
            ("events", "检测事件"),
        ]
        for index, (key, title) in enumerate(labels):
            block = tk.Frame(stats, bg=PANEL, pady=10)
            block.pack(fill="x")
            tk.Label(
                block,
                text=title,
                bg=PANEL,
                fg=MUTED,
                anchor="w",
                font=("Microsoft YaHei UI", 9),
            ).pack(fill="x")
            value = tk.Label(
                block,
                textvariable=self.stat_vars[key],
                bg=PANEL,
                fg=TEXT,
                anchor="w",
                font=("Microsoft YaHei UI", 20, "bold"),
            )
            value.pack(fill="x", pady=(4, 0))
            if key == "state":
                self.state_value_label = value
            if index < len(labels) - 1:
                tk.Frame(stats, bg="#25304a", height=1).pack(fill="x")

        timeline_area = tk.Frame(page, bg=BG)
        timeline_area.grid(row=1, column=0, sticky="ew")
        tk.Label(
            timeline_area,
            textvariable=self.status_var,
            bg=BG,
            fg=MUTED,
            anchor="w",
            font=("Microsoft YaHei UI", 9),
        ).pack(fill="x")
        self.live_timeline = EventTimeline(timeline_area, bg=BG)
        self.live_timeline.pack(fill="x", pady=(5, 10))

        controls = tk.Frame(page, bg=BG)
        controls.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        controls.grid_columnconfigure(0, weight=1)
        self.input_entry = tk.Entry(
            controls,
            textvariable=self.input_var,
            bg=PANEL,
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            font=("Microsoft YaHei UI", 10),
        )
        self.input_entry.grid(row=0, column=0, sticky="ew", ipady=9)
        self._button(controls, "选择视频", self._select_input, secondary=True).grid(
            row=0, column=1, padx=(10, 0)
        )
        self.start_button = self._button(controls, "开始检测", self._start)
        self.start_button.grid(row=0, column=2, padx=(10, 0))
        self.cancel_button = self._button(controls, "停止检测", self._cancel, secondary=True)
        self.cancel_button.grid(row=0, column=3, padx=(10, 0))
        self.cancel_button.configure(state="disabled")

    def _build_history_page(self) -> None:
        page = self.history_page
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(1, weight=1)
        columns = ("time", "source", "status", "events")
        self.history_tree = ttk.Treeview(
            page,
            columns=columns,
            show="headings",
            height=5,
            style="History.Treeview",
        )
        headings = {"time": "检测时间", "source": "视频", "status": "状态", "events": "跌倒次数"}
        widths = {"time": 160, "source": 470, "status": 120, "events": 100}
        for column in columns:
            self.history_tree.heading(column, text=headings[column])
            self.history_tree.column(column, width=widths[column], anchor="w")
        self.history_tree.grid(row=0, column=0, sticky="ew", pady=(14, 12))
        self.history_tree.bind("<<TreeviewSelect>>", self._select_history_record)

        player = tk.Frame(page, bg=BG)
        player.grid(row=1, column=0, sticky="nsew")
        player.grid_columnconfigure(0, weight=1)
        player.grid_columnconfigure(1, minsize=205)
        player.grid_rowconfigure(0, weight=1)
        self.history_preview_label = tk.Label(
            player,
            text="选择一条历史记录进行回放",
            bg=SURFACE,
            fg=MUTED,
            font=("Microsoft YaHei UI", 12),
        )
        self.history_preview_label.grid(row=0, column=0, sticky="nsew")
        controls = tk.Frame(player, bg=PANEL, padx=18, pady=18)
        controls.grid(row=0, column=1, sticky="ns", padx=(16, 0))
        tk.Label(
            controls,
            text="记录回放",
            bg=PANEL,
            fg=TEXT,
            font=("Microsoft YaHei UI", 13, "bold"),
        ).pack(anchor="w")
        tk.Label(
            controls,
            textvariable=self.history_info_var,
            bg=PANEL,
            fg=MUTED,
            justify="left",
            wraplength=165,
            font=("Microsoft YaHei UI", 9),
        ).pack(anchor="w", pady=(10, 20))
        self.play_button = self._button(controls, "播放", self._toggle_history_playback)
        self.play_button.pack(fill="x")

        self.history_timeline = EventTimeline(page, bg=BG)
        self.history_timeline.grid(row=2, column=0, sticky="ew", pady=(10, 12))

    def _button(
        self, master: tk.Misc, text: str, command: Any, *, secondary: bool = False
    ) -> tk.Button:
        return tk.Button(
            master,
            text=text,
            command=command,
            bg=PANEL if secondary else BLUE,
            fg=TEXT,
            activebackground="#2467bd" if not secondary else "#1b263d",
            activeforeground=TEXT,
            disabledforeground="#65718a",
            relief="flat",
            bd=0,
            padx=18,
            pady=9,
            cursor="hand2",
            font=("Microsoft YaHei UI", 9),
        )

    def _select_input(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("视频", "*.avi *.mp4 *.mov *.mkv")])
        if path:
            self.input_var.set(path)

    def _start(self) -> None:
        input_path = Path(self.input_var.get())
        if not input_path.is_file():
            messagebox.showerror("无法开始", "请选择有效的视频文件")
            return
        record_dir = create_recording_dir(self.recordings_dir, input_path)
        capture = cv2.VideoCapture(str(input_path))
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        self._live_duration = total / fps if total > 0 else 0.0
        self._live_intervals = []
        self._active_fall_start = None
        initial_stats = {
            "state": "检测中",
            "probability": "--",
            "progress": "0%",
            "events": "0",
        }
        for key, value in initial_stats.items():
            self.stat_vars[key].set(value)
        self.status_var.set(f"正在检测 · {input_path.name}")
        self.worker = VideoWorker(input_path, record_dir, self.pipeline)
        self.worker.start()
        self.start_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.root.after(50, self._poll)

    def _cancel(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
            self.status_var.set("正在停止检测…")

    def _poll(self) -> None:
        worker = self.worker
        if worker is None:
            return
        latest: PreviewUpdate | None = None
        while not worker.preview_queue.empty():
            latest = worker.preview_queue.get_nowait()
        if latest is not None:
            self._apply_live_update(latest, worker.event_count)
        if worker.is_alive():
            self.root.after(50, self._poll)
            return
        self.start_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")
        if worker.error is not None:
            self.stat_vars["state"].set("失败")
            self.state_value_label.configure(fg=RED)
            self.status_var.set(f"处理失败 · {worker.error}")
            messagebox.showerror("处理失败", str(worker.error))
        elif worker.result is not None:
            self._finish_live_intervals(worker.result.events_path)
            status_text = {"completed": "检测完成", "cancelled": "已停止"}.get(
                worker.result.status, worker.result.status
            )
            self.stat_vars["state"].set(status_text)
            final_color = GREEN if worker.result.status == "completed" else TEXT
            self.state_value_label.configure(fg=final_color)
            self.status_var.set(f"{status_text} · 已保存到 {worker.result.video_path.parent}")
            self.refresh_history()

    def _apply_live_update(self, update: PreviewUpdate, event_count: int) -> None:
        result = update.result
        self._display_frame(self.preview_label, update.frame, history=False)
        is_fall = result.state == "fall"
        self.stat_vars["state"].set("检测到跌倒" if is_fall else "正常")
        self.state_value_label.configure(fg=RED if is_fall else GREEN)
        probability = "--" if result.fall_probability is None else f"{result.fall_probability:.1%}"
        self.stat_vars["probability"].set(probability)
        self.stat_vars["progress"].set(f"{update.processed / max(update.total, 1):.0%}")
        self.stat_vars["events"].set(str(event_count))
        if is_fall and self._active_fall_start is None:
            self._active_fall_start = result.timestamp
        elif not is_fall and self._active_fall_start is not None:
            self._live_intervals.append((self._active_fall_start, result.timestamp))
            self._active_fall_start = None
        visible_intervals = list(self._live_intervals)
        if self._active_fall_start is not None:
            visible_intervals.append((self._active_fall_start, result.timestamp))
        duration = max(self._live_duration, result.timestamp, 0.001)
        self.live_timeline.set_data(duration, result.timestamp, visible_intervals)

    def _finish_live_intervals(self, events_path: Path) -> None:
        self._live_intervals = load_event_intervals(events_path, self._live_duration)
        self._active_fall_start = None
        self.live_timeline.set_data(
            self._live_duration, self._live_duration, self._live_intervals
        )

    def _display_frame(self, label: tk.Label, frame: Any, *, history: bool) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        max_width = max(label.winfo_width(), 640)
        max_height = max(label.winfo_height(), 340)
        image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        label.configure(image=photo, text="")
        if history:
            self._history_photo = photo
        else:
            self._live_photo = photo

    def refresh_history(self) -> None:
        selected_path = None
        selection = self.history_tree.selection() if hasattr(self, "history_tree") else ()
        if selection:
            record = self.history_records.get(selection[0])
            selected_path = record.record_dir if record else None
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        self.history_records.clear()
        for index, record in enumerate(load_history_records(self.recordings_dir)):
            item_id = f"record-{index}"
            self.history_records[item_id] = record
            status = {"completed": "已完成", "cancelled": "已停止", "failed": "失败"}.get(
                record.status, record.status
            )
            self.history_tree.insert(
                "",
                "end",
                iid=item_id,
                values=(
                    record.recorded_at.strftime("%Y-%m-%d %H:%M"),
                    record.source_name,
                    status,
                    record.event_count,
                ),
            )
            if selected_path == record.record_dir:
                self.history_tree.selection_set(item_id)

    def _select_history_record(self, _event: tk.Event[tk.Misc]) -> None:
        selection = self.history_tree.selection()
        if not selection:
            return
        record = self.history_records.get(selection[0])
        if record is None:
            return
        self._stop_history_playback()
        self._release_history_capture()
        self._history_capture = cv2.VideoCapture(str(record.video_path))
        if not self._history_capture.isOpened():
            self.history_info_var.set("无法打开历史视频")
            return
        self._history_fps = float(self._history_capture.get(cv2.CAP_PROP_FPS)) or 25.0
        frames = int(self._history_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self._history_duration = (
            frames / self._history_fps if frames > 0 else record.duration_seconds
        )
        self._history_intervals = load_event_intervals(record.events_path, self._history_duration)
        ok, frame = self._history_capture.read()
        if ok:
            self._display_frame(self.history_preview_label, frame, history=True)
        self._history_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.history_timeline.set_data(self._history_duration, 0.0, self._history_intervals)
        history_info = (
            f"{record.source_name}\n"
            f"{record.recorded_at:%Y-%m-%d %H:%M}\n"
            f"跌倒事件 {record.event_count} 次"
        )
        self.history_info_var.set(history_info)
        self.play_button.configure(state="normal", text="播放")

    def _toggle_history_playback(self) -> None:
        if self._history_capture is None:
            return
        if self._history_playing:
            self._stop_history_playback()
        else:
            self._history_playing = True
            self.play_button.configure(text="暂停")
            self._play_next_history_frame()

    def _play_next_history_frame(self) -> None:
        capture = self._history_capture
        if not self._history_playing or capture is None:
            return
        ok, frame = capture.read()
        if not ok:
            self._stop_history_playback()
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.history_timeline.set_data(self._history_duration, 0.0, self._history_intervals)
            return
        self._display_frame(self.history_preview_label, frame, history=True)
        position = float(capture.get(cv2.CAP_PROP_POS_FRAMES)) / self._history_fps
        self.history_timeline.set_data(self._history_duration, position, self._history_intervals)
        delay = max(int(1000 / self._history_fps), 15)
        self._history_after_id = self.root.after(delay, self._play_next_history_frame)

    def _stop_history_playback(self) -> None:
        self._history_playing = False
        if hasattr(self, "play_button"):
            self.play_button.configure(text="播放")
        if self._history_after_id is not None:
            self.root.after_cancel(self._history_after_id)
            self._history_after_id = None

    def _release_history_capture(self) -> None:
        if self._history_capture is not None:
            self._history_capture.release()
            self._history_capture = None

    def _close(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            self.worker.cancel()
        self._stop_history_playback()
        self._release_history_capture()
        self.root.destroy()


def launch_app(
    bundle_dir: Path = Path("outputs/autodl_training/runtime_bundle"),
    pose_model: str = "yolo26n-pose.pt",
    device: str | None = None,
    recordings_dir: Path = Path("recordings"),
) -> None:
    model, metadata = load_runtime_bundle(bundle_dir, device)
    extractor = YoloPoseExtractor(pose_model, device)
    pipeline = FallGuardPipeline(
        extractor,
        model,
        device=device,
        pose_fps=float(metadata["pose_fps"]),
        window_frames=int(metadata["window_frames"]),
        window_stride=int(metadata["window_stride"]),
        fall_threshold=float(metadata["fall_threshold"]),
        recovery_threshold=float(metadata["recovery_threshold"]),
        trigger_windows=int(metadata["trigger_windows"]),
        recovery_seconds=float(metadata["recovery_seconds"]),
        cooldown_seconds=float(metadata["cooldown_seconds"]),
        model_version=metadata["weights_sha256"],
    )
    root = tk.Tk()
    runtime_device = str(pipeline.device).upper()
    FallGuardApp(
        root,
        pipeline=pipeline,
        recordings_dir=recordings_dir,
        model_status=f"YOLO Pose · ST-GCN 已加载 · {runtime_device}",
        experimental_model=not bool(metadata["passed_deployment_gate"]),
    )
    root.mainloop()
