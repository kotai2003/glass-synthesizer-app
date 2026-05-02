"""
Manual-screenshot helper.

Workspace-only utility: launches the synthesizer GUI, drives it
programmatically, and saves PNGs of the running window into
../../../00.docs/manual_screens/ for the user manual.

This script ships in the standalone repo too (it lives under
synthesizer_app/tests/, which is the subtree-split prefix), but it
only works when run from the GLASS workspace where 00.docs/ exists
and synthesizer_app is importable as a package. In the standalone
repo, gui_main.py is at the repo root (no `synthesizer_app` package),
and there is no 00.docs/ — so this script aborts early with a clear
message.

Run from GLASS/ with the glass_env interpreter:

    "C:/Users/seong/anaconda3/envs/glass_env/python.exe" \
        synthesizer_app/tests/_capture_manual_screens.py

Not part of the user-facing test suite; the underscore prefix keeps
unittest discover from picking it up.
"""
from __future__ import annotations

import os
import sys
import time
import tkinter as tk

# Make `synthesizer_app...` importable when launched as a plain script
# (only meaningful in the workspace; in the standalone repo there is no
# `synthesizer_app` package directory at this level).
_HERE = os.path.dirname(os.path.abspath(__file__))
_GLASS_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
if _GLASS_ROOT not in sys.path:
    sys.path.insert(0, _GLASS_ROOT)

# Workspace layout sentinel: the manual lives at <workspace>/00.docs/.
_REPO_ROOT = os.path.abspath(os.path.join(_GLASS_ROOT, os.pardir))
_DOCS_DIR = os.path.join(_REPO_ROOT, "00.docs")


def _abort_if_not_workspace():
    """Refuse to run outside the GLASS workspace.

    Prevents the standalone-repo case where _REPO_ROOT resolves to some
    unrelated parent dir and we would (a) fail on `import synthesizer_app`
    and (b) write screenshots to a surprising location.
    """
    if os.path.isdir(_DOCS_DIR) and os.path.isdir(
            os.path.join(_GLASS_ROOT, "synthesizer_app")):
        return
    sys.stderr.write(
        "[capture] This utility is workspace-only. It expects to run from\n"
        "          <workspace>/GLASS/synthesizer_app/tests/_capture_manual_screens.py\n"
        "          where <workspace>/00.docs/ exists. Detected layout:\n"
        f"            _GLASS_ROOT = {_GLASS_ROOT}\n"
        f"            _REPO_ROOT  = {_REPO_ROOT}\n"
        "          The standalone glass-synthesizer-app repo does not\n"
        "          contain 00.docs/ and lays out modules differently, so\n"
        "          screenshots are managed from the parent workspace there.\n"
        "          Aborting.\n"
    )
    sys.exit(2)


_abort_if_not_workspace()

from PIL import ImageGrab  # noqa: E402

from synthesizer_app.gui_main import GuiSynth  # noqa: E402


# Output directory: <repo-root>/00.docs/manual_screens
OUT_DIR = os.path.join(_DOCS_DIR, "manual_screens")
os.makedirs(OUT_DIR, exist_ok=True)

# Demo data on this workstation (see CLAUDE.md).
OK_DIR = r"C:/Datasets-rev002/01.MVTEC_Anomaly_Detection/bottle/train/good"
TEX_DIR = r"C:/Datasets-rev002/DTD/images"
OUT_DIR_DEMO = os.path.join(_REPO_ROOT, "synthetic_dump", "_manual_demo")


def _grab(root: tk.Tk, name: str, settle_ms: int = 250):
    """Capture the window region into OUT_DIR/<name>.png."""
    # Re-assert top-most on every grab in case another app stole focus.
    root.attributes("-topmost", True)
    root.lift()
    root.update_idletasks()
    root.update()
    # Tk needs a tick or two for repainting after state changes.
    time.sleep(settle_ms / 1000.0)
    root.update()
    x = root.winfo_rootx()
    y = root.winfo_rooty()
    w = root.winfo_width()
    h = root.winfo_height()
    bbox = (x, y, x + w, y + h)
    img = ImageGrab.grab(bbox=bbox)
    path = os.path.join(OUT_DIR, name + ".png")
    img.save(path)
    print(f"[capture] {path}  ({w}x{h})")


def _find_notebook(widget):
    from tkinter import ttk
    if isinstance(widget, ttk.Notebook):
        return widget
    for child in widget.winfo_children():
        found = _find_notebook(child)
        if found is not None:
            return found
    return None


def main():
    root = tk.Tk()
    app = GuiSynth(root)
    # Move to a predictable spot; default placement varies per session.
    root.geometry("1600x900+30+20")
    # Keep window on top so ImageGrab.grab(bbox=...) actually captures
    # the Tk window pixels and not whatever else happens to be at those
    # screen coordinates. Without this the window can sit behind the
    # active app and the "screenshot" will leak unrelated content.
    root.attributes("-topmost", True)
    root.lift()
    root.focus_force()
    root.update_idletasks()
    root.update()

    notebook = _find_notebook(root)
    if notebook is None:
        raise RuntimeError("Notebook widget not found")

    steps = []

    # 01: initial state, Control tab, no inputs filled.
    def step_01():
        notebook.select(0)
        _grab(root, "01_initial_control_tab", settle_ms=400)

    # 02: Control tab with paths + class name + N filled in.
    def step_02():
        app.ok_dir_var_ui.set(OK_DIR)
        app.tex_dir_var_ui.set(TEX_DIR)
        app.out_dir_var_ui.set(OUT_DIR_DEMO)
        app.classname_var_ui.set("bottle_demo")
        app.n_per_ok_var_ui.set(5)
        _grab(root, "02_control_tab_filled", settle_ms=300)

    # 03: Configure tab default state.
    def step_03():
        notebook.select(1)
        _grab(root, "03_configure_tab_default", settle_ms=400)

    # 04: Configure tab with custom synth params (knob examples).
    def step_04():
        app.working_size_var_ui.set(384)
        app.perlin_min_var_ui.set(1)
        app.perlin_max_var_ui.set(5)
        app.beta_mean_var_ui.set(0.45)
        app.beta_mean_label_var_ui.set("0.45")
        app.beta_std_var_ui.set(0.08)
        app.beta_std_label_var_ui.set("0.08")
        app.seed_var_ui.set("42")
        _grab(root, "04_configure_tab_tweaked", settle_ms=300)

    # 05: Back to Control tab and run a real Preview synthesis.
    def step_05():
        notebook.select(0)
        # Reset to safe knobs for a clean preview
        app.working_size_var_ui.set(288)
        app.perlin_min_var_ui.set(0)
        app.perlin_max_var_ui.set(6)
        app.beta_mean_var_ui.set(0.5)
        app.beta_mean_label_var_ui.set("0.50")
        app.beta_std_var_ui.set(0.1)
        app.beta_std_label_var_ui.set("0.10")
        app.seed_var_ui.set("7")
        app.on_preview()
        _grab(root, "05_after_preview", settle_ms=600)

    # 06: Configure tab again, with the live-preview rendered next to it
    # (Configure is on the right; preview area is on the left so both
    # can be visible).
    def step_06():
        notebook.select(1)
        # bump beta to show the live-preview re-render
        app.beta_mean_var_ui.set(0.7)
        app.beta_mean_label_var_ui.set("0.70")
        # The slider command would normally be triggered by user drag;
        # call it manually so the debounce path runs.
        app.on_param_change()
        # Wait out the 200ms debounce + ~400ms synth + paint
        _grab(root, "06_live_preview_after_change", settle_ms=900)

    # 07: Back to Control, close cleanly.
    def step_07():
        notebook.select(0)
        _grab(root, "07_final", settle_ms=300)
        root.after(200, root.destroy)

    steps = [step_01, step_02, step_03, step_04, step_05, step_06, step_07]

    # Chain via after(); each step waits for paint settle internally.
    delay = 800
    cursor = [0]

    def runner():
        i = cursor[0]
        if i >= len(steps):
            return
        try:
            steps[i]()
        except Exception as e:
            print(f"[capture] step {i} failed: {e!r}")
            root.destroy()
            return
        cursor[0] += 1
        if cursor[0] < len(steps):
            root.after(delay, runner)

    root.after(delay, runner)
    root.mainloop()


if __name__ == "__main__":
    main()
