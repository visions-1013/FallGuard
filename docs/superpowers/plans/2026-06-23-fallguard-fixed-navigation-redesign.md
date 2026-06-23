# FallGuard Fixed Navigation Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the size-shifting native notebook tabs with a fixed top navigation and refine both monitoring pages into one consistent, restrained workspace.

**Architecture:** Keep the existing `FallGuardApp` and inference flow. Replace `ttk.Notebook` with a fixed navigation frame and a stacked content container; switch pages with `tkraise()` so navigation geometry never depends on page content or selected-state theme metrics.

**Tech Stack:** Python 3.11, Tkinter/ttk, pytest, Pillow, OpenCV

---

### Task 1: Lock Fixed Navigation Behavior with a Regression Test

**Files:**
- Modify: `tests/test_tk_app.py`
- Modify: `src/fallguard/ui/tk_app.py`

- [ ] **Step 1: Replace the notebook assertion with a fixed-navigation regression test**

```python
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
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `python -m pytest tests/test_tk_app.py::test_app_builds_fixed_monitor_navigation -q`

Expected: FAIL because `nav_items`, `show_page`, and `current_page` do not exist.

- [ ] **Step 3: Implement fixed navigation and stacked pages**

Create two `112×44` navigation frames with propagation disabled, a fixed 2px indicator, and labels bound to `show_page()`. Place `monitor_page` and `history_page` in the same content-container grid cell and switch them with `tkraise()`.

```python
def show_page(self, name: str) -> None:
    self.current_page = name
    self.pages[name].tkraise()
    for key, item in self.nav_items.items():
        selected = key == name
        item.label.configure(fg=TEXT if selected else MUTED)
        item.indicator.configure(bg=BLUE if selected else BG)
    if name == "history":
        self.refresh_history()
```

- [ ] **Step 4: Run the focused test and verify GREEN**

Run: `python -m pytest tests/test_tk_app.py -q`

Expected: all GUI tests pass.

### Task 2: Apply the Approved Visual Hierarchy

**Files:**
- Modify: `src/fallguard/ui/tk_app.py`
- Test: `tests/test_tk_app.py`

- [ ] **Step 1: Refine the top bar and page surfaces**

Keep a 64px header, a 44px navigation row, and a single content surface. Remove notebook styling and thick native borders; retain only `#243047` 1px dividers. Keep model status right-aligned and ensure the content container owns all vertical resizing.

- [ ] **Step 2: Align both page workspaces**

Use the same video background, right-side rail width, horizontal padding, and timeline spacing on real-time and history pages. Keep four monitoring values and one history playback action; do not add cards, icons, filters, or configuration controls.

- [ ] **Step 3: Run GUI and static checks**

Run:

```powershell
python -m pytest tests/test_tk_app.py tests/test_timeline.py tests/test_history.py -q
python -m ruff check src/fallguard/ui tests/test_tk_app.py
python -m mypy src/fallguard
```

Expected: all tests and checks pass.

### Task 3: Verify the Complete Application and Refresh Evidence

**Files:**
- Update generated screenshots: `pictures/23_FallGuard实时检测后台.png`
- Update generated screenshots: `pictures/24_FallGuard历史记录页面.png`

- [ ] **Step 1: Run the complete regression suite**

Run: `python -m pytest`

Expected: all existing tests pass with only the existing official-checkpoint skip.

- [ ] **Step 2: Run real GUI inference**

Launch the GUI with the existing YOLO Pose and `outputs/autodl_training/runtime_bundle` weights, process `datasets/Coffee_room_01/Coffee_room_01/Videos/video (44).avi`, and verify the red event interval remains visible.

- [ ] **Step 3: Capture both pages**

Replace the two screenshot files with the redesigned real-time detection and history playback views. Visually confirm that both navigation items remain identical in size across the two images.

- [ ] **Step 4: Restart the final GUI**

Close the previous FallGuard GUI process, start `python -m fallguard.cli app`, and verify its main window title is `FallGuard 跌倒监测后台` and the process is responding.
