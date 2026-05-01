"""
UI layer for the GLASS Synthesizer app.

Pure widget definitions. No business logic, no synthesis calls. Logic lives in
`gui_main.GuiSynth` which subclasses GuiSynthUI and binds the callback methods
referenced here as `command=self.<name>` placeholders.

Layout follows skills/tomomi-gui-style/SKILL.md:
- Toplevel 1920x1080
- PanedWindow split: left preview (weight=5) | right control (weight=1)
- Right Notebook: Control tab (only one for Phase 2)
- Logo on top, copyright on bottom of Control tab
- LabelFrames group "Inputs" / "Run"
- primary.TButton for main actions, secondary.TButton for ancillary
"""
from __future__ import annotations

import datetime
import os
import pathlib
import tkinter as tk
import tkinter.ttk as ttk

from . import custom_styles_jp


_HERE = pathlib.Path(__file__).parent
LOGO_PATH = _HERE / "TR_inc_logo.png"


class GuiSynthUI:
    APP_NAME = "GLASS Synthesizer"

    def __init__(self, master: tk.Tk):
        # Important: setup styles BEFORE creating any ttk widgets.
        custom_styles_jp.setup_ttk_styles(master)
        self.root = master
        self.root.title(f"{self.APP_NAME} by TOMOMI RESEARCH, INC.")
        self.root.geometry("1280x800")
        self.root.minsize(1024, 640)

        # ---- Tk variables (UI layer suffix _var_ui per TR convention) -----
        self.ok_dir_var_ui = tk.StringVar(value="")
        self.tex_dir_var_ui = tk.StringVar(value="")
        self.out_dir_var_ui = tk.StringVar(value="")
        self.classname_var_ui = tk.StringVar(value="my_class")
        self.n_per_ok_var_ui = tk.IntVar(value=10)
        self.status_var_ui = tk.StringVar(value="Ready.")

        # Configure-tab synthesis params (Phase 3)
        self.working_size_var_ui = tk.IntVar(value=288)
        self.perlin_min_var_ui = tk.IntVar(value=0)
        self.perlin_max_var_ui = tk.IntVar(value=6)
        self.beta_mean_var_ui = tk.DoubleVar(value=0.5)
        self.beta_std_var_ui = tk.DoubleVar(value=0.1)
        self.rand_aug_var_ui = tk.BooleanVar(value=True)
        self.seed_var_ui = tk.StringVar(value="")  # empty = random per call
        self.fg_dir_var_ui = tk.StringVar(value="")
        self.use_fg_var_ui = tk.BooleanVar(value=False)
        # Live readouts for ttk.Scale (which is continuous; we display rounded text)
        self.beta_mean_label_var_ui = tk.StringVar(value="0.50")
        self.beta_std_label_var_ui = tk.StringVar(value="0.10")

        # Photo references must live as instance attributes (skill §15.3).
        self._photo_logo = None
        self._photo_canvas_left = None
        self._photo_canvas_mid = None
        self._photo_canvas_right = None

        self._build_layout()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_layout(self):
        paned = ttk.Panedwindow(self.root, orient="horizontal")
        paned.pack(expand=True, fill="both")

        left = ttk.Frame(paned, style="custom.TFrame")
        right = ttk.Frame(paned, style="custom.TFrame")
        paned.add(left, weight=5)
        paned.add(right, weight=2)

        self._build_preview(left)
        self._build_sidepanel(right)

    # ---------------- Preview area (left) -----------------------------
    def _build_preview(self, parent):
        wrap = ttk.LabelFrame(parent, text="Preview (Original | Synthetic | Mask)",
                              style="custom.TLabelframe")
        wrap.pack(expand=True, fill="both", padx=10, pady=10, side="top")

        canvas_row = ttk.Frame(wrap, style="custom.TFrame")
        canvas_row.pack(expand=True, fill="both", padx=4, pady=4)

        self.canvas_left = tk.Canvas(canvas_row, bg="#222", highlightthickness=0)
        self.canvas_mid = tk.Canvas(canvas_row, bg="#222", highlightthickness=0)
        self.canvas_right = tk.Canvas(canvas_row, bg="#222", highlightthickness=0)
        self.canvas_left.pack(side="left", expand=True, fill="both", padx=2)
        self.canvas_mid.pack(side="left", expand=True, fill="both", padx=2)
        self.canvas_right.pack(side="left", expand=True, fill="both", padx=2)

        # Caption row under each canvas
        cap_row = ttk.Frame(wrap, style="custom.TFrame")
        cap_row.pack(fill="x", padx=4, pady=(0, 4))
        ttk.Label(cap_row, text="Original", style="custom.TLabel",
                  anchor="center").pack(side="left", expand=True, fill="x")
        ttk.Label(cap_row, text="Synthetic NG", style="custom.TLabel",
                  anchor="center").pack(side="left", expand=True, fill="x")
        ttk.Label(cap_row, text="Mask", style="custom.TLabel",
                  anchor="center").pack(side="left", expand=True, fill="x")

        # Thumbnail strip of generated samples (Phase 5)
        self._build_thumbnail_strip(parent)

        # Status label sits below preview
        status = ttk.LabelFrame(parent, text="Status", style="custom.TLabelframe")
        status.pack(fill="x", padx=10, pady=(0, 10), side="top")
        self.label_status = ttk.Label(status, textvariable=self.status_var_ui,
                                      style="custom.TLabel", anchor="w")
        self.label_status.pack(fill="x", padx=10, pady=(5, 2))

        # Progress bar (Phase 4): determinate mode, value updated from worker via _post.
        prog_row = ttk.Frame(status, style="custom.TFrame")
        prog_row.pack(fill="x", padx=10, pady=(0, 5))
        self.progress = ttk.Progressbar(prog_row, mode="determinate", maximum=100)
        self.progress.pack(side="left", expand=True, fill="x", padx=(0, 8))
        self.label_progress = ttk.Label(prog_row, text="0 / 0",
                                         style="custom.TLabel", width=12,
                                         anchor="e")
        self.label_progress.pack(side="left")

    # ---------------- Sidepanel (right) -------------------------------
    def _build_sidepanel(self, parent):
        notebook = ttk.Notebook(parent)
        notebook.pack(expand=True, fill="both", padx=5, pady=5)

        tab_control = ttk.Frame(notebook, style="custom.TFrame")
        notebook.add(tab_control, text="Control")
        self._build_control_tab(tab_control)

        tab_configure = ttk.Frame(notebook, style="custom.TFrame")
        notebook.add(tab_configure, text="Configure")
        self._build_configure_tab(tab_configure)

    def _build_control_tab(self, parent):
        # Logo
        logo_frame = ttk.Frame(parent, style="custom.TFrame")
        logo_frame.pack(expand=False, fill="x", side="top", padx=5, pady=(8, 4))
        try:
            from PIL import Image, ImageTk
            img = Image.open(LOGO_PATH)
            img.thumbnail((220, 80))
            self._photo_logo = ImageTk.PhotoImage(img)
            ttk.Label(logo_frame, image=self._photo_logo,
                      style="custom.TLabel", anchor="center").pack(
                expand=True, fill="x", side="top")
        except Exception:
            ttk.Label(logo_frame, text="TOMOMI RESEARCH, INC.",
                      style="custom.TLabel", anchor="center").pack(
                expand=True, fill="x", side="top")

        # Inputs
        frame_inputs = ttk.LabelFrame(parent, text="Inputs",
                                       style="custom.TLabelframe")
        frame_inputs.pack(expand=False, fill="x", padx=5, pady=5, side="top")
        self._build_path_row(frame_inputs, "OK images dir:", self.ok_dir_var_ui,
                             "browse_ok")
        self._build_path_row(frame_inputs, "Texture dir:", self.tex_dir_var_ui,
                             "browse_tex")
        self._build_path_row(frame_inputs, "Output dir:", self.out_dir_var_ui,
                             "browse_out")

        # Run params
        frame_run = ttk.LabelFrame(parent, text="Run", style="custom.TLabelframe")
        frame_run.pack(expand=False, fill="x", padx=5, pady=5, side="top")

        row1 = ttk.Frame(frame_run, style="custom.TFrame")
        row1.pack(fill="x", padx=10, pady=2)
        ttk.Label(row1, text="Class name:", style="custom.TLabel",
                  width=14).pack(side="left")
        ttk.Entry(row1, textvariable=self.classname_var_ui,
                  style="custom.TEntry").pack(side="left", expand=True, fill="x")

        row2 = ttk.Frame(frame_run, style="custom.TFrame")
        row2.pack(fill="x", padx=10, pady=2)
        ttk.Label(row2, text="N per OK image:", style="custom.TLabel",
                  width=14).pack(side="left")
        ttk.Spinbox(row2, from_=1, to=200, textvariable=self.n_per_ok_var_ui,
                    width=6).pack(side="left")

        # Buttons
        frame_ctl = ttk.LabelFrame(parent, text="Control",
                                    style="custom.TLabelframe")
        frame_ctl.pack(expand=False, fill="x", padx=5, pady=5, side="top")
        self.button_preview = ttk.Button(frame_ctl, text="Preview 1 sample",
                                          style="primary.TButton",
                                          command=lambda: self._dispatch("on_preview"))
        self.button_preview.pack(expand=True, fill="x", padx=30, pady=5,
                                  side="top")
        self.button_generate = ttk.Button(frame_ctl, text="Generate batch",
                                           style="primary.TButton",
                                           command=lambda: self._dispatch("on_generate"))
        self.button_generate.pack(expand=True, fill="x", padx=30, pady=5,
                                   side="top")
        self.button_cancel = ttk.Button(frame_ctl, text="Cancel",
                                         style="secondary.TButton",
                                         command=lambda: self._dispatch("on_cancel"))
        self.button_cancel.pack(expand=True, fill="x", padx=30, pady=5,
                                 side="top")
        self.button_cancel.state(["disabled"])
        self.button_open_out = ttk.Button(frame_ctl, text="Open output folder",
                                           style="secondary.TButton",
                                           command=lambda: self._dispatch("on_open_output"))
        self.button_open_out.pack(expand=True, fill="x", padx=30, pady=5,
                                   side="top")
        self.button_quit = ttk.Button(frame_ctl, text="Quit",
                                       style="secondary.TButton",
                                       command=self.root.destroy)
        self.button_quit.pack(expand=True, fill="x", padx=30, pady=5,
                               side="top")

        # Copyright
        year = datetime.date.today().year
        copyright_frame = ttk.LabelFrame(parent, text="",
                                          style="custom.TLabelframe")
        copyright_frame.pack(expand=False, fill="x", padx=5, pady=(20, 5),
                              side="bottom")
        ttk.Label(copyright_frame,
                  text=f"(C) {year} TOMOMI RESEARCH, INC.",
                  style="custom.TLabelframe.Label",
                  anchor="center").pack(anchor="center", expand=True,
                                         fill="both", padx=5, pady=5)

    # ---------------- Thumbnail strip (Phase 5) -----------------------
    THUMB_H = 88
    THUMB_W = 110

    def _build_thumbnail_strip(self, parent):
        wrap = ttk.LabelFrame(parent, text="Generated samples (click to open)",
                               style="custom.TLabelframe")
        wrap.pack(fill="x", padx=10, pady=(0, 10), side="top")

        outer = ttk.Frame(wrap, style="custom.TFrame")
        outer.pack(fill="x", padx=4, pady=4)

        self.strip_canvas = tk.Canvas(
            outer, height=self.THUMB_H + 16,
            bg="#1a1a1a", highlightthickness=0,
        )
        self.strip_canvas.pack(side="top", fill="x", expand=True)

        scrollbar = ttk.Scrollbar(outer, orient="horizontal",
                                    command=self.strip_canvas.xview)
        scrollbar.pack(side="bottom", fill="x")
        self.strip_canvas.configure(xscrollcommand=scrollbar.set,
                                     scrollregion=(0, 0, 0, self.THUMB_H + 16))

        # Mouse wheel = horizontal scroll
        self.strip_canvas.bind(
            "<MouseWheel>",
            lambda e: self.strip_canvas.xview_scroll(
                int(-1 * (e.delta / 120)), "units"),
        )

        # Empty-state hint as a canvas text item; cleared on first add.
        self._strip_empty_id = self.strip_canvas.create_text(
            8, (self.THUMB_H + 16) // 2,
            text="(no samples yet — click Generate batch to populate)",
            fill="#888", anchor="w",
        )

    # ---------------- Configure tab (Phase 3) -------------------------
    def _build_configure_tab(self, parent):
        # Synthesis params
        frame = ttk.LabelFrame(parent, text="Synthesis params",
                                style="custom.TLabelframe")
        frame.pack(expand=False, fill="x", padx=5, pady=5, side="top")

        # working_size: discrete combobox
        row = ttk.Frame(frame, style="custom.TFrame")
        row.pack(fill="x", padx=10, pady=2)
        ttk.Label(row, text="Working size:", style="custom.TLabel",
                  width=16).pack(side="left")
        cmb = ttk.Combobox(row, textvariable=self.working_size_var_ui,
                            values=(256, 288, 320, 384, 512), width=8,
                            state="readonly")
        cmb.pack(side="left")
        cmb.bind("<<ComboboxSelected>>",
                  lambda e: self._dispatch("on_param_change"))

        # Perlin scale min / max
        for label, var, lo, hi in [
            ("Perlin scale min:", self.perlin_min_var_ui, 0, 4),
            ("Perlin scale max:", self.perlin_max_var_ui, 1, 7),
        ]:
            row = ttk.Frame(frame, style="custom.TFrame")
            row.pack(fill="x", padx=10, pady=2)
            ttk.Label(row, text=label, style="custom.TLabel",
                      width=16).pack(side="left")
            sp = ttk.Spinbox(row, from_=lo, to=hi, textvariable=var, width=6,
                              command=lambda: self._dispatch("on_param_change"))
            sp.pack(side="left")

        # Beta mean (Scale + readout)
        row = ttk.Frame(frame, style="custom.TFrame")
        row.pack(fill="x", padx=10, pady=2)
        ttk.Label(row, text="Beta mean:", style="custom.TLabel",
                  width=16).pack(side="left")
        ttk.Scale(row, from_=0.2, to=0.8, orient="horizontal",
                  variable=self.beta_mean_var_ui,
                  command=lambda v: self._on_scale_beta_mean(v)).pack(
            side="left", expand=True, fill="x", padx=(0, 4))
        ttk.Label(row, textvariable=self.beta_mean_label_var_ui,
                  style="custom.TLabel", width=5).pack(side="left")

        # Beta std (Scale + readout)
        row = ttk.Frame(frame, style="custom.TFrame")
        row.pack(fill="x", padx=10, pady=2)
        ttk.Label(row, text="Beta std:", style="custom.TLabel",
                  width=16).pack(side="left")
        ttk.Scale(row, from_=0.0, to=0.3, orient="horizontal",
                  variable=self.beta_std_var_ui,
                  command=lambda v: self._on_scale_beta_std(v)).pack(
            side="left", expand=True, fill="x", padx=(0, 4))
        ttk.Label(row, textvariable=self.beta_std_label_var_ui,
                  style="custom.TLabel", width=5).pack(side="left")

        # Random texture aug
        row = ttk.Frame(frame, style="custom.TFrame")
        row.pack(fill="x", padx=10, pady=2)
        ttk.Checkbutton(row, text="Random texture augmentation",
                         variable=self.rand_aug_var_ui,
                         style="custom.TCheckbutton",
                         command=lambda: self._dispatch("on_param_change")
                         ).pack(side="left", padx=(0, 0))

        # Seed
        row = ttk.Frame(frame, style="custom.TFrame")
        row.pack(fill="x", padx=10, pady=2)
        ttk.Label(row, text="Seed (blank=rand):", style="custom.TLabel",
                  width=16).pack(side="left")
        ent = ttk.Entry(row, textvariable=self.seed_var_ui,
                         style="custom.TEntry", width=10)
        ent.pack(side="left")
        ent.bind("<FocusOut>",
                  lambda e: self._dispatch("on_param_change"))
        ent.bind("<Return>",
                  lambda e: self._dispatch("on_param_change"))

        # Foreground mask
        fg_frame = ttk.LabelFrame(parent, text="Foreground mask",
                                    style="custom.TLabelframe")
        fg_frame.pack(expand=False, fill="x", padx=5, pady=5, side="top")
        row = ttk.Frame(fg_frame, style="custom.TFrame")
        row.pack(fill="x", padx=10, pady=2)
        ttk.Checkbutton(row, text="Use foreground mask",
                         variable=self.use_fg_var_ui,
                         style="custom.TCheckbutton",
                         command=lambda: self._dispatch("on_param_change")
                         ).pack(side="left")
        self._build_path_row(fg_frame, "FG mask dir:", self.fg_dir_var_ui,
                              "browse_fg")
        ttk.Label(fg_frame,
                  text="Note: per-image lookup by basename. Missing -> warn but proceed.",
                  style="custom.TLabel", wraplength=320,
                  justify="left").pack(fill="x", padx=10, pady=(0, 4))

    def _on_scale_beta_mean(self, value):
        try:
            v = float(value)
            self.beta_mean_label_var_ui.set(f"{v:.2f}")
        except Exception:
            pass
        self._dispatch("on_param_change")

    def _on_scale_beta_std(self, value):
        try:
            v = float(value)
            self.beta_std_label_var_ui.set(f"{v:.2f}")
        except Exception:
            pass
        self._dispatch("on_param_change")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_path_row(self, parent, label, var, dispatch_name):
        row = ttk.Frame(parent, style="custom.TFrame")
        row.pack(fill="x", padx=10, pady=4)
        ttk.Label(row, text=label, style="custom.TLabel", width=14).pack(side="left")
        ttk.Entry(row, textvariable=var, style="custom.TEntry").pack(
            side="left", expand=True, fill="x", padx=(0, 4))
        ttk.Button(row, text="...", style="secondary.TButton", width=4,
                    command=lambda: self._dispatch(dispatch_name)).pack(side="left")

    def _dispatch(self, name):
        """Invoke a method named `name` on self if it exists.

        This lets the UI layer reference future logic-layer callbacks without
        breaking when the UI is instantiated alone for layout testing.
        """
        fn = getattr(self, name, None)
        if callable(fn):
            fn()
        else:
            self.status_var_ui.set(f"(stub) {name} not bound")
