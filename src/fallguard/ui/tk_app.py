from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from fallguard.inference.pipeline import FallGuardPipeline
from fallguard.models.runtime import load_runtime_bundle
from fallguard.pose.yolo_pose import YoloPoseExtractor

from .worker import VideoWorker


class FallGuardApp:
    def __init__(
        self,
        root: tk.Tk,
        bundle_dir: Path,
        pose_model: str = "yolo26n-pose.pt",
        device: str | None = None,
    ) -> None:
        self.root = root
        self.root.title("FallGuard 视频跌倒识别")
        self.root.geometry("680x300")
        model, metadata = load_runtime_bundle(bundle_dir, device)
        extractor = YoloPoseExtractor(pose_model, device)
        self.pipeline = FallGuardPipeline(
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
        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar(value=str(Path.cwd() / "outputs"))
        self.status_var = tk.StringVar(value="请选择视频")
        self.result_var = tk.StringVar(value="")
        self.progress_var = tk.DoubleVar(value=0)
        self.worker: VideoWorker | None = None
        self._build()

    def _build(self) -> None:
        frame = ttk.Frame(self.root, padding=16)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="输入视频").grid(row=0, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.input_var, width=64).grid(row=0, column=1, padx=8)
        ttk.Button(frame, text="选择", command=self._select_input).grid(row=0, column=2)
        ttk.Label(frame, text="输出目录").grid(row=1, column=0, sticky="w", pady=10)
        ttk.Entry(frame, textvariable=self.output_var, width=64).grid(row=1, column=1, padx=8)
        ttk.Button(frame, text="选择", command=self._select_output).grid(row=1, column=2)
        self.progress = ttk.Progressbar(frame, variable=self.progress_var, maximum=100)
        self.progress.grid(row=2, column=0, columnspan=3, sticky="ew", pady=12)
        ttk.Label(frame, textvariable=self.status_var).grid(row=3, column=0, columnspan=3)
        ttk.Label(frame, textvariable=self.result_var, foreground="#1b5e20").grid(
            row=4, column=0, columnspan=3, pady=8
        )
        buttons = ttk.Frame(frame)
        buttons.grid(row=5, column=0, columnspan=3)
        self.start_button = ttk.Button(buttons, text="开始处理", command=self._start)
        self.start_button.pack(side="left", padx=6)
        self.cancel_button = ttk.Button(
            buttons, text="取消", command=self._cancel, state="disabled"
        )
        self.cancel_button.pack(side="left", padx=6)

    def _select_input(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("视频", "*.avi *.mp4 *.mov *.mkv")])
        if path:
            self.input_var.set(path)

    def _select_output(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.output_var.set(path)

    def _start(self) -> None:
        input_path = Path(self.input_var.get())
        if not input_path.is_file():
            messagebox.showerror("错误", "请选择有效的视频文件")
            return
        output_dir = Path(self.output_var.get())
        self.worker = VideoWorker(input_path, output_dir, self.pipeline)
        self.worker.start()
        self.start_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.result_var.set("")
        self.root.after(100, self._poll)

    def _cancel(self) -> None:
        if self.worker:
            self.worker.cancel()
            self.status_var.set("正在取消…")

    def _poll(self) -> None:
        if self.worker is None:
            return
        while not self.worker.progress_queue.empty():
            processed, total, result = self.worker.progress_queue.get_nowait()
            self.progress_var.set(processed / max(total, 1) * 100)
            probability = (
                "--" if result.fall_probability is None else f"{result.fall_probability:.3f}"
            )
            self.status_var.set(f"{processed}/{total}  {result.state}  fall={probability}")
        if self.worker.is_alive():
            self.root.after(100, self._poll)
            return
        self.start_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")
        if self.worker.error:
            messagebox.showerror("处理失败", str(self.worker.error))
            self.status_var.set("处理失败")
        else:
            self.status_var.set(self.worker.result.status)
            self.result_var.set(str(self.worker.result.video_path))


def launch_app(
    bundle_dir: Path,
    pose_model: str = "yolo26n-pose.pt",
    device: str | None = None,
) -> None:
    root = tk.Tk()
    FallGuardApp(root, bundle_dir, pose_model, device)
    root.mainloop()
