from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk

import pytest

from fallguard.ui.tk_app import FallGuardApp


class FakePipeline:
    device = "cpu"


def test_app_builds_fixed_monitor_navigation(tmp_path: Path) -> None:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    root.withdraw()
    try:
        app = FallGuardApp(
            root,
            pipeline=FakePipeline(),
            recordings_dir=tmp_path,
            model_status="YOLO Pose · ST-GCN 已加载",
            experimental_model=True,
        )

        root.update_idletasks()
        assert set(app.nav_items) == {"monitor", "history"}
        before = {
            name: (item.winfo_reqwidth(), item.winfo_reqheight())
            for name, item in app.nav_items.items()
        }
        assert before == {"monitor": (112, 44), "history": (112, 44)}

        app.show_page("history")
        root.update_idletasks()
        after = {
            name: (item.winfo_reqwidth(), item.winfo_reqheight())
            for name, item in app.nav_items.items()
        }

        assert after == before
        assert app.current_page == "history"
        assert ttk.Style(root).layout("History.Treeview") == [
            ("Treeview.treearea", {"sticky": "nswe"})
        ]
        assert set(app.stat_vars) == {"state", "probability", "progress", "events"}
        assert app.history_tree.get_children() == ()
    finally:
        root.destroy()
