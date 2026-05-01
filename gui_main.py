"""
Logic layer for the GLASS Synthesizer GUI app.

GuiSynth subclasses GuiSynthUI and binds business-logic callbacks (browse,
preview, generate). Phase 2 keeps everything single-threaded; Phase 4 will
move batch generation to a background worker.

Run from the GLASS/ directory with the glass_env interpreter:

    "C:/Users/seong/anaconda3/envs/glass_env/python.exe" -m synthesizer_app.gui_main
"""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import imgaug
import numpy as np
import PIL.Image
import PIL.ImageTk
import torch

# Allow `python -m synthesizer_app.gui_main` from GLASS/ as well as
# direct `python synthesizer_app/gui_main.py`.
_HERE = os.path.dirname(os.path.abspath(__file__))
_GLASS_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
if _GLASS_ROOT not in sys.path:
    sys.path.insert(0, _GLASS_ROOT)

from synthesizer_app.ui.gui_main_ui import GuiSynthUI
from synthesizer_app.core.synthesis import SynthParams, synthesize_one
from synthesizer_app.core.exporter import MvtecExporter
from synthesizer_app.core.io_utils import (
    list_images_recursive,
    list_images_flat,
)


def _seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    imgaug.seed(seed)


def _bgr_to_photo(bgr: np.ndarray, target_w: int, target_h: int) -> PIL.ImageTk.PhotoImage:
    """OpenCV BGR -> ImageTk.PhotoImage, fitted to (target_w, target_h)."""
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    h, w = bgr.shape[:2]
    if target_w <= 0 or target_h <= 0:
        target_w, target_h = w, h
    scale = min(target_w / w, target_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return PIL.ImageTk.PhotoImage(PIL.Image.fromarray(rgb))


def _mask_to_overlay_bgr(mask_uint8: np.ndarray) -> np.ndarray:
    """0/255 mask -> JET-colored BGR for display."""
    return cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)


class GuiSynth(GuiSynthUI):
    """Business logic layer."""

    LIVE_PREVIEW_DEBOUNCE_MS = 200

    def __init__(self, master: tk.Tk):
        super().__init__(master)
        # Cache of last loaded paths (so Preview can re-pick deterministically)
        self._cached_ok_paths = []
        self._cached_tex_paths = []
        # Last preview's source paths (live preview reuses these on slider drag)
        self._last_preview_src = None
        self._last_preview_tex = None
        # Pending after-id for debounced live preview
        self._pending_live_after = None
        # Phase 4: background-worker state
        self._worker_thread = None
        self._cancel_event = None
        # Cross-thread post via a queue + main-thread polling.
        # Tcl is not thread-safe, so calling root.after() from a worker thread
        # is unreliable. We marshal via queue.Queue and drain on a Tk timer.
        self._task_queue = queue.Queue()
        self._POLL_INTERVAL_MS = 40
        self.root.after(self._POLL_INTERVAL_MS, self._drain_task_queue)
        # Phase 5: thumbnail strip state
        self._strip_records = []   # list of (src_path, ng_path, mask_path)
        self._strip_photos = []    # parallel list of PhotoImage refs (GC anchor)
        self.STRIP_CAP = 300       # hard cap to avoid Tk slowdown on huge runs
        # Trap window close so an in-flight worker can be signaled to stop.
        self.root.protocol("WM_DELETE_WINDOW", self._on_window_close)

    # ------------------------------------------------------------------
    # Tk-safe post helper (queue-based, robust to Tcl threading limits)
    # ------------------------------------------------------------------
    def _post(self, fn):
        """Schedule fn() on the Tk main thread. Safe to call from any thread."""
        self._task_queue.put(fn)

    def _drain_task_queue(self):
        try:
            while True:
                fn = self._task_queue.get_nowait()
                try:
                    fn()
                except Exception:
                    traceback.print_exc()
        except queue.Empty:
            pass
        # Reschedule on the main thread.
        try:
            self.root.after(self._POLL_INTERVAL_MS, self._drain_task_queue)
        except tk.TclError:
            pass  # window destroyed

    def _is_busy(self) -> bool:
        return self._worker_thread is not None and self._worker_thread.is_alive()

    # ------------------------------------------------------------------
    # Browse callbacks (UI layer dispatches via name)
    # ------------------------------------------------------------------
    def browse_ok(self):
        d = filedialog.askdirectory(title="Select folder containing OK (normal) images",
                                     initialdir=self._initial_dir(self.ok_dir_var_ui))
        if d:
            self.ok_dir_var_ui.set(d)
            self._cached_ok_paths = []  # invalidate

    def browse_tex(self):
        d = filedialog.askdirectory(title="Select folder containing texture images (DTD or in-house)",
                                     initialdir=self._initial_dir(self.tex_dir_var_ui))
        if d:
            self.tex_dir_var_ui.set(d)
            self._cached_tex_paths = []  # invalidate

    def browse_out(self):
        d = filedialog.askdirectory(title="Select output folder (MVTec-compatible layout will be created)",
                                     initialdir=self._initial_dir(self.out_dir_var_ui))
        if d:
            self.out_dir_var_ui.set(d)

    def browse_fg(self):
        d = filedialog.askdirectory(title="Select foreground mask folder (per-image basename match)",
                                     initialdir=self._initial_dir(self.fg_dir_var_ui))
        if d:
            self.fg_dir_var_ui.set(d)
            # Setting a folder doesn't auto-enable; user toggles checkbox.
            self.on_param_change()

    def _initial_dir(self, var: tk.StringVar) -> str:
        v = var.get().strip()
        return v if v and os.path.isdir(v) else os.path.expanduser("~")

    def on_open_output(self):
        out = self.out_dir_var_ui.get().strip()
        if not out or not os.path.isdir(out):
            messagebox.showwarning("Output folder",
                                    "Output folder is not set or does not exist.")
            return
        try:
            os.startfile(out)  # Windows-only; fine here per CLAUDE.md
        except AttributeError:
            subprocess.Popen(["xdg-open", out])

    # ------------------------------------------------------------------
    # Synthesis callbacks
    # ------------------------------------------------------------------
    def _validate_inputs(self, require_output: bool = True) -> bool:
        if not self.ok_dir_var_ui.get().strip() or not os.path.isdir(self.ok_dir_var_ui.get()):
            messagebox.showerror("Input error", "OK images folder is not set or does not exist.")
            return False
        if not self.tex_dir_var_ui.get().strip() or not os.path.isdir(self.tex_dir_var_ui.get()):
            messagebox.showerror("Input error", "Texture folder is not set or does not exist.")
            return False
        if require_output:
            if not self.out_dir_var_ui.get().strip():
                messagebox.showerror("Input error", "Output folder is not set.")
                return False
            os.makedirs(self.out_dir_var_ui.get(), exist_ok=True)
        return True

    def _ensure_caches(self) -> bool:
        if not self._cached_ok_paths:
            try:
                paths = list_images_flat(self.ok_dir_var_ui.get())
                if not paths:
                    paths = list_images_recursive(self.ok_dir_var_ui.get())
            except FileNotFoundError as e:
                messagebox.showerror("Input error", str(e))
                return False
            if not paths:
                messagebox.showerror("Input error",
                                      "No images found in OK folder.")
                return False
            self._cached_ok_paths = paths
        if not self._cached_tex_paths:
            try:
                paths = list_images_recursive(self.tex_dir_var_ui.get())
            except FileNotFoundError as e:
                messagebox.showerror("Input error", str(e))
                return False
            if not paths:
                messagebox.showerror("Input error",
                                      "No images found in Texture folder.")
                return False
            self._cached_tex_paths = paths
        return True

    def _build_params(self) -> SynthParams:
        # Phase 3: read from Configure tab.
        try:
            ws = int(self.working_size_var_ui.get())
        except (tk.TclError, ValueError):
            ws = 288
        try:
            pmin = int(self.perlin_min_var_ui.get())
            pmax = int(self.perlin_max_var_ui.get())
        except (tk.TclError, ValueError):
            pmin, pmax = 0, 6
        if pmax <= pmin:
            pmax = pmin + 1
        try:
            bm = float(self.beta_mean_var_ui.get())
            bs = float(self.beta_std_var_ui.get())
        except (tk.TclError, ValueError):
            bm, bs = 0.5, 0.1

        use_fg = bool(self.use_fg_var_ui.get())
        if use_fg and not (self.fg_dir_var_ui.get().strip() and
                           os.path.isdir(self.fg_dir_var_ui.get())):
            # Toggle on but no folder -> downgrade silently to off; status shows warning.
            use_fg = False

        return SynthParams(
            working_size=ws,
            output_size=None,        # preserve original resolution (Q3)
            perlin_scale_min=pmin,
            perlin_scale_max=pmax,
            beta_mean=bm,
            beta_std=bs,
            rand_aug=bool(self.rand_aug_var_ui.get()),
            downsampling=8,
            use_foreground=use_fg,
        )

    def _resolve_fg_for(self, src_path: str):
        """Look up the per-image fg mask under fg_dir by basename.

        Returns a PIL.Image.Image or None. Does not raise.
        """
        fg_dir = self.fg_dir_var_ui.get().strip()
        if not fg_dir or not os.path.isdir(fg_dir):
            return None
        stem = os.path.splitext(os.path.basename(src_path))[0]
        for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
            cand = os.path.join(fg_dir, stem + ext)
            if os.path.isfile(cand):
                try:
                    return PIL.Image.open(cand)
                except Exception:
                    return None
        return None

    def _resolve_seed(self):
        """None means 'random per call'; an int seeds all three RNGs."""
        s = self.seed_var_ui.get().strip()
        if not s:
            return None
        try:
            return int(s)
        except ValueError:
            return None

    def on_preview(self, reuse_last: bool = False):
        """Run a single preview synthesis and paint the canvases.

        reuse_last=True is used by the live-preview debounce path: it keeps the
        same OK/texture pair so slider changes are interpreted visually, not
        confused by re-rolling the source pair.
        """
        if self._is_busy():
            return  # don't compete with the batch worker for the synthesis pipeline
        if not self._validate_inputs(require_output=False):
            return
        if not self._ensure_caches():
            return

        try:
            if reuse_last and self._last_preview_src and self._last_preview_tex:
                src_path = self._last_preview_src
                tex_path = self._last_preview_tex
            else:
                src_idx = np.random.randint(0, len(self._cached_ok_paths))
                tex_idx = np.random.randint(0, len(self._cached_tex_paths))
                src_path = self._cached_ok_paths[src_idx]
                tex_path = self._cached_tex_paths[tex_idx]

            src_img = PIL.Image.open(src_path).convert("RGB")
            tex_img = PIL.Image.open(tex_path).convert("RGB")

            params = self._build_params()
            seed = self._resolve_seed()
            if seed is not None:
                _seed_all(seed)

            fg_img = self._resolve_fg_for(src_path) if params.use_foreground else None
            if params.use_foreground and fg_img is None:
                # Per-image fg missing -> Q6 says warn but proceed; fall back to off.
                params.use_foreground = False
                self.status_var_ui.set(
                    f"WARN: no fg mask for {os.path.basename(src_path)}; "
                    "synthesizing without foreground."
                )

            result = synthesize_one(src_img, tex_img, params, fg_mask=fg_img)

            orig_bgr = cv2.cvtColor(np.array(src_img), cv2.COLOR_RGB2BGR)
            mask_jet = _mask_to_overlay_bgr(result.mask_uint8)
            self._show_three(orig_bgr, result.ng_image_bgr, mask_jet)

            self._last_preview_src = src_path
            self._last_preview_tex = tex_path
            self.status_var_ui.set(
                f"Preview: src={os.path.basename(src_path)}, "
                f"tex={os.path.basename(tex_path)}, beta={result.beta_used:.3f}, "
                f"ws={params.working_size}, perlin=[{params.perlin_scale_min},{params.perlin_scale_max}], "
                f"fg={'on' if params.use_foreground else 'off'}"
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Preview error", f"{type(e).__name__}: {e}")
            self.status_var_ui.set(f"Preview failed: {e}")

    # ------------------------------------------------------------------
    # Live preview (Phase 3)
    # ------------------------------------------------------------------
    def on_param_change(self):
        """Called from any Configure-tab widget. Schedules a debounced live preview.

        No-op if no baseline preview has been rendered yet (avoids surprise
        synthesis when the user is just clicking around the Configure tab),
        and no-op while a batch worker is running.
        """
        if self._is_busy():
            return
        if self._last_preview_src is None or self._last_preview_tex is None:
            return
        if self._pending_live_after is not None:
            try:
                self.root.after_cancel(self._pending_live_after)
            except Exception:
                pass
        self._pending_live_after = self.root.after(
            self.LIVE_PREVIEW_DEBOUNCE_MS, self._run_live_preview)

    def _run_live_preview(self):
        self._pending_live_after = None
        self.on_preview(reuse_last=True)

    def on_generate(self):
        if self._is_busy():
            return
        if not self._validate_inputs(require_output=True):
            return
        if not self._ensure_caches():
            return
        try:
            n_per_ok = int(self.n_per_ok_var_ui.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Input error", "N per OK image must be an integer.")
            return
        if n_per_ok < 1:
            messagebox.showerror("Input error", "N per OK image must be >= 1.")
            return

        classname = self.classname_var_ui.get().strip()
        if not classname:
            messagebox.showerror("Input error", "Class name is empty.")
            return

        params = self._build_params()
        params_dict = {k: v for k, v in vars(params).items()}
        out_root = self.out_dir_var_ui.get()

        try:
            exporter = MvtecExporter(out_root, classname, params_dict)
        except Exception as e:
            messagebox.showerror("Output error", f"Cannot create output: {e}")
            return

        # Snapshot inputs so the worker is not coupled to live Tk vars.
        ok_paths = list(self._cached_ok_paths)
        tex_paths = list(self._cached_tex_paths)
        seed_base = self._resolve_seed()
        total = len(ok_paths) * n_per_ok

        self._cancel_event = threading.Event()
        self._strip_clear()
        self._lock_for_run(total)

        self._worker_thread = threading.Thread(
            target=self._worker_generate,
            args=(exporter, ok_paths, tex_paths, params, seed_base,
                  n_per_ok, total, classname, out_root,
                  self._cancel_event),
            daemon=True,
        )
        self._worker_thread.start()

    def on_cancel(self):
        if self._cancel_event is not None and not self._cancel_event.is_set():
            self._cancel_event.set()
            self.status_var_ui.set("Canceling... waiting for current sample to finish.")
            self.button_cancel.state(["disabled"])

    def _on_window_close(self):
        if self._is_busy():
            if not messagebox.askyesno(
                    "Quit",
                    "A batch is running. Quit anyway? "
                    "(Already-saved samples will be kept.)"):
                return
            if self._cancel_event is not None:
                self._cancel_event.set()
        self.root.destroy()

    # ---------------- Thumbnail strip (Phase 5) ----------------------
    def _strip_clear(self):
        # Wipe all canvas items, drop refs, re-add empty hint.
        self.strip_canvas.delete("all")
        self._strip_records = []
        self._strip_photos = []
        self._strip_empty_id = self.strip_canvas.create_text(
            8, (self.THUMB_H + 16) // 2,
            text="(no samples yet — click Generate batch to populate)",
            fill="#888", anchor="w",
        )
        self.strip_canvas.configure(scrollregion=(0, 0, 0, self.THUMB_H + 16))

    def _strip_add(self, src_path: str, ng_path: str, mask_path: str):
        """Append one thumbnail. Called via _post from the worker thread.

        Capped at self.STRIP_CAP entries; oldest are evicted (FIFO) to keep
        Tk performant on large runs.
        """
        if not os.path.isfile(ng_path):
            return  # race: worker faster than disk flush; skip silently.

        # Drop empty-state hint on first add.
        if self._strip_empty_id is not None:
            self.strip_canvas.delete(self._strip_empty_id)
            self._strip_empty_id = None

        bgr = cv2.imread(ng_path, cv2.IMREAD_COLOR)
        if bgr is None:
            return
        photo = self._make_thumbnail_photo(bgr)

        if len(self._strip_records) >= self.STRIP_CAP:
            # Evict the oldest: shift all canvas items left by one slot, drop
            # the leftmost one. We just delete the first thumbnail tag and
            # rebuild positions of everything else. Cheap because STRIP_CAP
            # is bounded.
            self._strip_evict_oldest()

        idx = len(self._strip_records)
        x = idx * (self.THUMB_W + 6) + 4
        y = 4
        tag = f"thumb_{idx}"
        item_id = self.strip_canvas.create_image(
            x, y, image=photo, anchor="nw", tags=(tag,))
        # Capture idx-by-default so the lambda binds to the slot, not the
        # ever-increasing closure variable.
        self.strip_canvas.tag_bind(
            tag, "<Button-1>",
            lambda e, slot=idx: self._strip_on_click(slot),
        )

        self._strip_records.append((src_path, ng_path, mask_path))
        self._strip_photos.append(photo)
        self.strip_canvas.configure(
            scrollregion=(0, 0, (idx + 1) * (self.THUMB_W + 6) + 8,
                           self.THUMB_H + 16),
        )
        # Auto-scroll to the right edge so the latest sample is in view.
        self.strip_canvas.xview_moveto(1.0)

    def _strip_evict_oldest(self):
        # Recreate canvas from the second-onwards records.
        # Keeps memory bounded; tag names get reset.
        kept_records = self._strip_records[1:]
        kept_photos = self._strip_photos[1:]
        self.strip_canvas.delete("all")
        self._strip_records = []
        self._strip_photos = []
        for rec, photo in zip(kept_records, kept_photos):
            idx = len(self._strip_records)
            x = idx * (self.THUMB_W + 6) + 4
            y = 4
            tag = f"thumb_{idx}"
            self.strip_canvas.create_image(
                x, y, image=photo, anchor="nw", tags=(tag,))
            self.strip_canvas.tag_bind(
                tag, "<Button-1>",
                lambda e, slot=idx: self._strip_on_click(slot),
            )
            self._strip_records.append(rec)
            self._strip_photos.append(photo)

    def _make_thumbnail_photo(self, bgr: np.ndarray) -> PIL.ImageTk.PhotoImage:
        h, w = bgr.shape[:2]
        scale = min(self.THUMB_W / w, self.THUMB_H / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        small = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        return PIL.ImageTk.PhotoImage(PIL.Image.fromarray(rgb))

    def _strip_on_click(self, slot: int):
        if not (0 <= slot < len(self._strip_records)):
            return
        src_path, ng_path, mask_path = self._strip_records[slot]
        try:
            src_bgr = cv2.imread(src_path, cv2.IMREAD_COLOR)
            ng_bgr = cv2.imread(ng_path, cv2.IMREAD_COLOR)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        except Exception as e:
            self.status_var_ui.set(f"Open thumbnail failed: {e}")
            return
        if src_bgr is None or ng_bgr is None or mask is None:
            self.status_var_ui.set(
                f"Thumbnail file missing (slot={slot}). "
                f"src={os.path.basename(src_path)}"
            )
            return
        # Resize source to match NG dimensions for visual parity.
        if src_bgr.shape[:2] != ng_bgr.shape[:2]:
            src_bgr = cv2.resize(src_bgr, (ng_bgr.shape[1], ng_bgr.shape[0]),
                                  interpolation=cv2.INTER_AREA)
        mask_jet = _mask_to_overlay_bgr(mask)
        if mask_jet.shape[:2] != ng_bgr.shape[:2]:
            mask_jet = cv2.resize(mask_jet, (ng_bgr.shape[1], ng_bgr.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
        self._show_three(src_bgr, ng_bgr, mask_jet)
        self.status_var_ui.set(
            f"Sample {slot}: {os.path.basename(ng_path)} "
            f"(src={os.path.basename(src_path)})"
        )

    # ---------------- Run-state UI helpers ---------------------------
    def _lock_for_run(self, total: int):
        self.button_preview.state(["disabled"])
        self.button_generate.state(["disabled"])
        self.button_cancel.state(["!disabled"])
        self.progress.configure(maximum=max(1, total), value=0)
        self.label_progress.configure(text=f"0 / {total}")
        self.status_var_ui.set(f"Starting batch ({total} samples)...")

    def _unlock_after_run(self):
        self.button_preview.state(["!disabled"])
        self.button_generate.state(["!disabled"])
        self.button_cancel.state(["disabled"])
        self._worker_thread = None
        self._cancel_event = None

    # ---------------- Worker (background thread) ---------------------
    def _worker_generate(self, exporter, ok_paths, tex_paths, params,
                         seed_base, n_per_ok, total, classname, out_root,
                         cancel_event):
        """Background thread. Must NOT touch widgets; use self._post."""
        try:
            done = 0
            fg_missing_count = 0
            last_result = None
            for src_path in ok_paths:
                if cancel_event.is_set():
                    break
                try:
                    exporter.copy_ok_image(src_path)
                except Exception:
                    pass  # non-fatal: target dir may already have it
                src_img = PIL.Image.open(src_path).convert("RGB")
                fg_img = self._resolve_fg_for(src_path) if params.use_foreground else None
                if params.use_foreground and fg_img is None:
                    fg_missing_count += 1
                eff_use_fg = params.use_foreground and fg_img is not None

                for _ in range(n_per_ok):
                    if cancel_event.is_set():
                        break
                    if seed_base is not None:
                        _seed_all(seed_base + done)
                    tex_path = tex_paths[np.random.randint(0, len(tex_paths))]
                    tex_img = PIL.Image.open(tex_path).convert("RGB")
                    eff_params = SynthParams(
                        working_size=params.working_size,
                        output_size=params.output_size,
                        perlin_scale_min=params.perlin_scale_min,
                        perlin_scale_max=params.perlin_scale_max,
                        beta_mean=params.beta_mean,
                        beta_std=params.beta_std,
                        rand_aug=params.rand_aug,
                        downsampling=params.downsampling,
                        use_foreground=eff_use_fg,
                    )
                    result = synthesize_one(
                        src_img, tex_img, eff_params,
                        fg_mask=fg_img if eff_use_fg else None,
                    )
                    rec = exporter.add_sample(
                        ng_image_bgr=result.ng_image_bgr,
                        mask_uint8=result.mask_uint8,
                        source_image_path=src_path,
                        texture_path=tex_path,
                        beta=result.beta_used,
                        seed=(seed_base + done) if seed_base is not None else None,
                    )
                    last_result = (src_img, result, tex_path, src_path)
                    # Compute absolute paths the strip can reload from disk.
                    ng_abs = os.path.join(
                        exporter.dir_test_synth, f"{rec.sample_id:04d}.png")
                    mask_abs = os.path.join(
                        exporter.dir_gt_synth, f"{rec.sample_id:04d}_mask.png")
                    self._post(lambda s=src_path, n=ng_abs, m=mask_abs:
                                self._strip_add(s, n, m))
                    done += 1
                    if done % 5 == 0 or done == total:
                        self._post(lambda d=done, t=total: self._on_progress(d, t))

            run_json = exporter.finalize()
            aborted = cancel_event.is_set()
            self._post(lambda: self._on_generate_done(
                last_result, done, total, fg_missing_count, run_json,
                classname, out_root, aborted))
        except Exception:
            err = traceback.format_exc()
            self._post(lambda: self._on_generate_error(err))

    # ---------------- Main-thread callbacks --------------------------
    def _on_progress(self, done: int, total: int):
        self.progress.configure(value=done)
        self.label_progress.configure(text=f"{done} / {total}")
        self.status_var_ui.set(f"Generating... {done}/{total}")

    def _on_generate_done(self, last_result, done, total, fg_missing_count,
                          run_json, classname, out_root, aborted):
        self._on_progress(done, total)
        if last_result is not None:
            src_img, result, tex_path, src_path = last_result
            orig_bgr = cv2.cvtColor(np.array(src_img), cv2.COLOR_RGB2BGR)
            mask_jet = _mask_to_overlay_bgr(result.mask_uint8)
            self._show_three(orig_bgr, result.ng_image_bgr, mask_jet)
        msg_tail = ""
        if fg_missing_count > 0:
            msg_tail = (f" WARN: {fg_missing_count} OK image(s) had no fg mask "
                        "(synthesized without foreground).")
        if aborted:
            self.status_var_ui.set(
                f"Canceled. {done}/{total} samples written under "
                f"{os.path.join(out_root, classname)}/.{msg_tail}"
            )
            messagebox.showinfo(
                "Canceled",
                f"Stopped after {done}/{total} samples.\n"
                f"Output: {os.path.join(out_root, classname)}\n"
                f"run.json: {run_json}"
            )
        else:
            self.status_var_ui.set(
                f"Done. {done} samples written under "
                f"{os.path.join(out_root, classname)}/.{msg_tail}"
            )
            messagebox.showinfo(
                "Generate batch",
                f"Wrote {done} synthetic samples to:\n"
                f"{os.path.join(out_root, classname)}\n"
                f"run.json: {run_json}"
            )
        self._unlock_after_run()

    def _on_generate_error(self, err_text: str):
        # Print full traceback to stderr for debugging.
        sys.stderr.write(err_text + "\n")
        self.status_var_ui.set("Generate failed (see error dialog).")
        messagebox.showerror("Generate error", err_text)
        self._unlock_after_run()

    # ------------------------------------------------------------------
    # Canvas helpers
    # ------------------------------------------------------------------
    def _show_three(self, left_bgr, mid_bgr, right_bgr):
        # Canvas dims may not be finalized at first paint; force update.
        self.root.update_idletasks()
        for canvas, attr_name, bgr in [
            (self.canvas_left, "_photo_canvas_left", left_bgr),
            (self.canvas_mid, "_photo_canvas_mid", mid_bgr),
            (self.canvas_right, "_photo_canvas_right", right_bgr),
        ]:
            cw = canvas.winfo_width()
            ch = canvas.winfo_height()
            photo = _bgr_to_photo(bgr, cw, ch)
            setattr(self, attr_name, photo)
            canvas.delete("all")
            canvas.create_image(cw // 2, ch // 2, image=photo, anchor="center")


def main():
    root = tk.Tk()
    GuiSynth(root)
    root.mainloop()


if __name__ == "__main__":
    main()
