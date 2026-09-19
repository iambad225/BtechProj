"""EEG modality tab: Config (10-20 selector), Monitor (band powers + alpha test), Record.

EEG *classification* is deferred (see the Train tab's provisional note); this tab captures
config + data now so models can be trained once the EEG paradigms are designed.
"""
from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

from core.config import Config, default_config, Electrode
from core.dsp import RealTimeFilter
from core import features_eeg as EF
from core.montage import OCCIPITAL
from sources import create_source
from sources.synthetic_eeg import EEGSyntheticSource
import storage.recordings as recordings
from gui.widgets.waveform import Waveform
from gui.widgets.bars import StackedBars
from gui.widgets.montage_10_20 import Montage1020

CONFIG_PATH = "config/eeg.yaml"
BAND_COLORS = {"delta": "#6c5ce7", "theta": "#0984e3", "alpha": "#e17055", "beta": "#00b894"}


class EEGEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.source = None
        self.mode = "idle"                 # idle | monitor | record
        self.events: "queue.Queue" = queue.Queue()
        self._lock = threading.Lock()
        self._thread = None
        self._running = False
        self._filt = None
        self._buf = np.zeros((0, cfg.n_channels))
        self._span = int(cfg.sample_rate * 4)
        self._n = 0
        self._t = 0.0
        self.raw = np.zeros((0, cfg.n_channels))
        self.bands: dict = {}
        self.broadband = np.zeros(cfg.n_channels)
        self.alpha = 0.0
        self.sps = 0.0
        self._schedule = []
        self._seg = None
        self._seg_end = 0.0
        self._rec_sig = []
        self._rec_lab = []

    def start(self):
        if self._running:
            return
        self.source = create_source(self.cfg)
        self.source.start()
        self._filt = RealTimeFilter(self.cfg.sample_rate, self.cfg.n_channels,
                                    notch_hz=self.cfg.filter.notch_hz,
                                    bandpass=(self.cfg.filter.bandpass_low, self.cfg.filter.bandpass_high),
                                    enabled=self.cfg.filter.enabled)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self.source:
            self.source.stop(); self.source = None

    @property
    def is_synthetic(self):
        return isinstance(self.source, EEGSyntheticSource)

    def go_idle(self):
        self.mode = "idle"

    def set_alpha_sim(self, on):
        if self.is_synthetic:
            self.source.set_alpha(on)

    def start_monitor(self):
        self._filt.reset(); self._n = 0; self._t = time.perf_counter()
        with self._lock:
            self._buf = np.zeros((0, self.cfg.n_channels))
        self.mode = "monitor"

    def start_record(self, schedule):
        self._schedule = list(schedule); self._seg = None
        self._rec_sig = []; self._rec_lab = []; self._filt.reset()
        self.mode = "record"

    def _loop(self):
        while self._running:
            new = self.source.get_data() if self.source else np.zeros((0, self.cfg.n_channels))
            if self.mode == "monitor":
                self._do_monitor(new)
            elif self.mode == "record":
                self._do_record(new)
            time.sleep(0.005)

    def _filtered(self, new):
        y = self._filt.process(new)
        if self.cfg.car and y.shape[1] > 1:
            y = y - y.mean(axis=1, keepdims=True)
        return y

    def _do_monitor(self, new):
        if new.shape[0]:
            y = self._filtered(new)
            self._n += new.shape[0]
            with self._lock:
                self._buf = np.vstack([self._buf, y])[-self._span:]
        dt = time.perf_counter() - self._t
        if dt >= self.cfg.step_ms / 1000.0:
            with self._lock:
                buf = self._buf.copy()
            sps = self._n / dt; self._n = 0; self._t = time.perf_counter()
            if buf.shape[0] >= self.cfg.window_samples:
                win = buf[-self.cfg.window_samples:]
                bp, bb = EF.band_powers(win, self.cfg.sample_rate, self.cfg.bands)
                ar = EF.alpha_ratio(bp, bb, self.cfg.alpha_band, self.cfg.alpha_columns())
                with self._lock:
                    self.raw, self.bands, self.broadband, self.alpha, self.sps = buf, bp, bb, ar, sps
            else:
                with self._lock:
                    self.raw, self.sps = buf, sps

    def _do_record(self, new):
        now = time.perf_counter()
        if self._seg is None:
            if not self._schedule:
                sig = np.vstack(self._rec_sig) if self._rec_sig else np.zeros((0, self.cfg.n_channels))
                self.mode = "idle"
                self.events.put(("recorded", sig, list(self._rec_lab)))
                return
            self._seg = self._schedule.pop(0)
            self._seg_end = now + self._seg[2]
            self.events.put(("status", f"{'Recording' if self._seg[0]=='record' else 'Get ready'}: {self._seg[1]}"))
        if self._seg[0] == "record" and new.shape[0]:
            self._rec_sig.append(new.copy())          # store raw; training re-filters
            self._rec_lab += [self._seg[1]] * new.shape[0]
        if now >= self._seg_end:
            self._seg = None


class EEGTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=8)
        self.cfg = self._load_cfg()
        self.engine = EEGEngine(self.cfg)
        self.grid(sticky="nsew")
        master.columnconfigure(0, weight=1); master.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1); self.rowconfigure(0, weight=1)

        nb = ttk.Notebook(self); nb.grid(sticky="nsew")
        self.t_cfg = ttk.Frame(nb, padding=8)
        self.t_mon = ttk.Frame(nb, padding=8)
        self.t_rec = ttk.Frame(nb, padding=8)
        nb.add(self.t_cfg, text="  Config  ")
        nb.add(self.t_mon, text="  Monitor  ")
        nb.add(self.t_rec, text="  Record  ")
        nb.bind("<<NotebookTabChanged>>", lambda e: self._on_tab())
        self._build_config()
        self._build_monitor()
        self._build_record()
        self.after(60, self._tick)

    def _load_cfg(self) -> Config:
        try:
            c = Config.load(CONFIG_PATH)
            if c.modality != "eeg":
                raise ValueError
            return c
        except Exception:
            c = default_config("eeg"); c.save(CONFIG_PATH); return c

    # ---------- config ----------
    def _build_config(self):
        t = self.t_cfg
        t.columnconfigure(1, weight=1); t.rowconfigure(0, weight=1)
        self.montage = Montage1020(t)
        self.montage.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        self.montage.set_electrodes(self.cfg.electrodes)

        form = ttk.Frame(t); form.grid(row=0, column=0, sticky="nw")
        r = 0
        def row(label, var, width=14):
            nonlocal r
            ttk.Label(form, text=label).grid(row=r, column=0, sticky="e", padx=4, pady=2)
            ttk.Entry(form, textvariable=var, width=width).grid(row=r, column=1, sticky="w"); r += 1
        self.v_subject = tk.StringVar(value=self.cfg.subject); row("Subject", self.v_subject)
        self.v_src = tk.StringVar(value=self.cfg.source.type)
        ttk.Label(form, text="Source").grid(row=r, column=0, sticky="e", padx=4)
        ttk.Combobox(form, textvariable=self.v_src, values=["cyton", "lsl", "synthetic"],
                     state="readonly", width=12).grid(row=r, column=1, sticky="w"); r += 1
        self.v_port = tk.StringVar(value=self.cfg.source.serial_port); row("Cyton port", self.v_port)
        self.v_sr = tk.StringVar(value=str(self.cfg.sample_rate)); row("Sample rate", self.v_sr, 8)
        self.v_notch = tk.StringVar(value=str(self.cfg.filter.notch_hz)); row("Notch Hz", self.v_notch, 8)
        self.v_lo = tk.StringVar(value=str(self.cfg.filter.bandpass_low)); row("Band low", self.v_lo, 8)
        self.v_hi = tk.StringVar(value=str(self.cfg.filter.bandpass_high)); row("Band high", self.v_hi, 8)
        self.v_car = tk.BooleanVar(value=self.cfg.car)
        ttk.Checkbutton(form, text="Common-avg ref", variable=self.v_car).grid(row=r, column=0, columnspan=2, sticky="w"); r += 1
        self.v_win = tk.StringVar(value=str(self.cfg.window_ms)); row("Window ms", self.v_win, 8)
        self.v_step = tk.StringVar(value=str(self.cfg.step_ms)); row("Step ms", self.v_step, 8)
        self.v_alpha_thr = tk.StringVar(value=str(self.cfg.alpha_threshold)); row("Alpha thresh", self.v_alpha_thr, 8)
        ttk.Label(form, text="Actions (optional; EEG\nclassification is WIP)",
                  foreground="#888").grid(row=r, column=0, columnspan=2, sticky="w", pady=(6, 0)); r += 1
        self.actions_list = tk.Listbox(form, height=4, width=20, exportselection=False)
        for a in self.cfg.labels:
            self.actions_list.insert(tk.END, a)
        self.actions_list.grid(row=r, column=0, sticky="w")
        ab = ttk.Frame(form); ab.grid(row=r, column=1, sticky="nw")
        self.v_newaction = ttk.Entry(ab, width=14); self.v_newaction.grid(row=0, column=0)
        ttk.Button(ab, text="Add", command=self._add_action).grid(row=1, column=0, sticky="w")
        ttk.Button(ab, text="Remove", command=self._remove_action).grid(row=2, column=0, sticky="w")
        r += 1
        ttk.Button(form, text="Apply & Save", command=self._apply).grid(row=r, column=0, sticky="w", pady=(10, 0))
        self.cfg_status = ttk.Label(form, text="", foreground="#0a7"); self.cfg_status.grid(row=r, column=1, sticky="w", pady=(10, 0))

    def _add_action(self):
        a = self.v_newaction.get().strip().replace(" ", "_")
        if a and a not in self.actions_list.get(0, tk.END):
            self.actions_list.insert(tk.END, a); self.v_newaction.delete(0, tk.END)

    def _remove_action(self):
        sel = self.actions_list.curselection()
        if sel:
            self.actions_list.delete(sel[0])

    def _apply(self):
        try:
            elecs = self.montage.get_electrodes()
            data = [e for e in elecs if e.role == "data"]
            if not data:
                raise ValueError("Enable at least one Data electrode and give it a channel.")
            if any(e.channel is None for e in data):
                raise ValueError("Every Data electrode needs a channel number.")
            chans = [e.channel for e in data]
            if len(set(chans)) != len(chans):
                raise ValueError("Two electrodes share the same channel number.")
            self.cfg.electrodes = elecs
            self.cfg.subject = self.v_subject.get().strip() or "subject"
            self.cfg.source.type = self.v_src.get()
            self.cfg.source.serial_port = self.v_port.get().strip()
            self.cfg.sample_rate = int(float(self.v_sr.get()))
            self.cfg.filter.notch_hz = float(self.v_notch.get())
            self.cfg.filter.bandpass_low = float(self.v_lo.get())
            self.cfg.filter.bandpass_high = float(self.v_hi.get())
            self.cfg.car = bool(self.v_car.get())
            self.cfg.window_ms = int(float(self.v_win.get()))
            self.cfg.step_ms = int(float(self.v_step.get()))
            self.cfg.alpha_threshold = float(self.v_alpha_thr.get())
            self.cfg.labels = list(self.actions_list.get(0, tk.END))
            # auto alpha electrodes = enabled occipital data electrodes
            occ = [e.name for e in data if e.name in OCCIPITAL]
            self.cfg.alpha_electrodes = occ or [e.name for e in data][-2:]
        except Exception as e:
            messagebox.showerror("Invalid config", str(e)); return
        self.cfg.save(CONFIG_PATH)
        self.engine.stop(); self.engine = EEGEngine(self.cfg)
        self._build_band_rows()
        self.cfg_status.config(text=f"Saved -> {CONFIG_PATH} ({self.cfg.n_channels} ch; "
                                    f"alpha on {self.cfg.alpha_electrodes})")

    # ---------- monitor ----------
    def _build_monitor(self):
        t = self.t_mon
        t.columnconfigure(0, weight=1); t.rowconfigure(1, weight=1)
        bar = ttk.Frame(t); bar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.mon_btn = ttk.Button(bar, text="▶ Start", command=self._toggle_monitor)
        self.mon_btn.grid(row=0, column=0)
        self.mon_status = ttk.Label(bar, text="stopped", foreground="#a00"); self.mon_status.grid(row=0, column=1, padx=8)
        self.mon_sps = ttk.Label(bar, text="", foreground="#555"); self.mon_sps.grid(row=0, column=2, padx=8)
        self.alpha_test = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="alpha test", variable=self.alpha_test).grid(row=0, column=3, padx=8)
        self.sim_alpha = tk.BooleanVar(value=False)
        self.sim_chk = ttk.Checkbutton(bar, text="simulate eyes-closed", variable=self.sim_alpha,
                                       command=lambda: self.engine.set_alpha_sim(self.sim_alpha.get()))
        self.sim_chk.grid(row=0, column=4, padx=8)

        main = ttk.Frame(t); main.grid(row=1, column=0, sticky="nsew")
        main.columnconfigure(1, weight=1); main.rowconfigure(0, weight=1)
        left = ttk.Frame(main); left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.rowconfigure(1, weight=1)
        leg = ttk.Frame(left); leg.grid(row=0, column=0, sticky="w")
        for i, (b, col) in enumerate(BAND_COLORS.items()):
            c = tk.Canvas(leg, width=12, height=12, highlightthickness=0); c.grid(row=0, column=i * 2, padx=(4, 2))
            c.create_rectangle(0, 0, 12, 12, fill=col, width=0)
            ttk.Label(leg, text=b).grid(row=0, column=i * 2 + 1)
        self.band_frame = ttk.LabelFrame(left, text="Band composition", padding=6)
        self.band_frame.grid(row=1, column=0, sticky="nsew", pady=6)
        self.band_frame.columnconfigure(0, weight=1); self.band_frame.rowconfigure(0, weight=1)
        self.band_bars = None
        self._build_band_rows()
        ap = ttk.Frame(left); ap.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        ap.columnconfigure(2, weight=1)
        self.alpha_light = tk.Canvas(ap, width=22, height=22, highlightthickness=0); self.alpha_light.grid(row=0, column=0)
        self._dot = self.alpha_light.create_oval(3, 3, 19, 19, fill="#444", width=0)
        self.alpha_lbl = ttk.Label(ap, text="alpha -", font=("", 13, "bold")); self.alpha_lbl.grid(row=0, column=1, padx=8)
        self.alpha_bar = tk.Canvas(ap, height=18, bg="#eee", highlightthickness=1, highlightbackground="#ccc")
        self.alpha_bar.grid(row=0, column=2, sticky="ew")
        self._arect = self.alpha_bar.create_rectangle(0, 0, 0, 18, fill="#e17055", width=0)
        self._aline = self.alpha_bar.create_line(0, 0, 0, 18, fill="#c00", width=2)

        right = ttk.LabelFrame(main, text="EEG traces", padding=4)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1); right.rowconfigure(0, weight=1)
        self.wave = Waveform(right); self.wave.grid(row=0, column=0, sticky="nsew")

    def _build_band_rows(self):
        if self.band_bars is not None:
            self.band_bars.destroy()
        self.band_bars = StackedBars(self.band_frame, self.cfg.electrode_names, BAND_COLORS)
        self.band_bars.grid(row=0, column=0, sticky="nsew")

    def _toggle_monitor(self):
        if self.engine.mode == "monitor":
            self.engine.go_idle(); self.mon_btn.config(text="▶ Start")
            self.mon_status.config(text="stopped", foreground="#a00"); return
        try:
            self.engine.start(); self._build_band_rows()
            self.engine.set_alpha_sim(self.sim_alpha.get()); self.engine.start_monitor()
        except Exception as e:
            messagebox.showerror("Source error", str(e)); return
        self.sim_chk.configure(state=("normal" if self.engine.is_synthetic else "disabled"))
        self.mon_btn.config(text="■ Stop"); self.mon_status.config(text="streaming", foreground="#0a0")

    # ---------- record ----------
    def _build_record(self):
        t = self.t_rec
        ttk.Label(t, text="Captures a session for later EEG training. With no actions "
                          "defined it records a single 'rest' baseline.", foreground="#555").grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, 8))
        ttk.Label(t, text="Seconds per class").grid(row=1, column=0, sticky="e", padx=4)
        self.v_secs = ttk.Spinbox(t, from_=2, to=30, width=5); self.v_secs.set(6)
        self.v_secs.grid(row=1, column=1, sticky="w")
        ttk.Label(t, text="Reps").grid(row=2, column=0, sticky="e", padx=4)
        self.v_reps = ttk.Spinbox(t, from_=1, to=10, width=5); self.v_reps.set(2)
        self.v_reps.grid(row=2, column=1, sticky="w")
        ttk.Label(t, text="Description (optional)").grid(row=3, column=0, sticky="e", padx=4)
        self.v_desc = tk.StringVar()
        ttk.Entry(t, textvariable=self.v_desc, width=40).grid(row=3, column=1, columnspan=3, sticky="w")
        self.rec_btn = ttk.Button(t, text="● Record session", command=self._record)
        self.rec_btn.grid(row=4, column=0, sticky="w", pady=8)
        self.rec_status = ttk.Label(t, text="", foreground="#0a7"); self.rec_status.grid(row=5, column=0, columnspan=4, sticky="w")

    def _record(self):
        self.rec_btn.config(state="disabled")
        try:
            self.engine.start()
        except Exception as e:
            self.rec_btn.config(state="normal"); messagebox.showerror("Source error", str(e)); return
        secs = float(self.v_secs.get()); reps = int(self.v_reps.get())
        classes = self.cfg.labels or ["rest"]
        schedule = []
        for _ in range(reps):
            for label in classes:
                schedule.append(("ready", label, 2.0))
                schedule.append(("record", label, secs))
        self.engine.start_record(schedule)
        self.rec_status.config(text="Recording...")

    # ---------- shared ----------
    def _on_tab(self):
        if self.engine.mode == "monitor":
            self.engine.go_idle(); self.mon_btn.config(text="▶ Start")
            self.mon_status.config(text="stopped", foreground="#a00")

    def _tick(self):
        try:
            while True:
                kind, *rest = self.engine.events.get_nowait()
                if kind == "status":
                    self.rec_status.config(text=rest[0])
                elif kind == "recorded":
                    self._on_recorded(rest[0], rest[1])
        except queue.Empty:
            pass
        if self.engine.mode == "monitor":
            with self.engine._lock:
                buf = self.engine.raw.copy(); bands = self.engine.bands
                bb = self.engine.broadband.copy(); alpha = self.engine.alpha; sps = self.engine.sps
            self.wave.draw(buf)
            if bands:
                for i in range(len(self.band_bars.rows)):
                    total = float(bb[i]) if i < len(bb) else 0.0
                    if total > 1e-9:
                        self.band_bars.set(i, {b: float(bands[b][i]) / total for b in BAND_COLORS if b in bands})
            self.mon_sps.config(text=f"{sps:.0f} Sa/s")
            thr = self.cfg.alpha_threshold
            aw = self.alpha_bar.winfo_width(); aw = aw if aw > 1 else 240
            ah = self.alpha_bar.winfo_height(); ah = ah if ah > 1 else 18
            self.alpha_bar.coords(self._arect, 0, 0, int(aw * min(alpha, 1.0)), ah)
            self.alpha_bar.coords(self._aline, int(aw * min(thr, 1.0)), 0, int(aw * min(thr, 1.0)), ah)
            on = self.alpha_test.get() and alpha >= thr
            self.alpha_light.itemconfig(self._dot, fill="#2ecc40" if on else "#444")
            state = "EYES CLOSED? (alpha high)" if on else "alpha low"
            self.alpha_lbl.config(text=f"alpha {alpha*100:0.0f}%   {state if self.alpha_test.get() else ''}")
        self.after(60, self._tick)

    def _on_recorded(self, sig, labels):
        self.rec_btn.config(state="normal")
        if sig.shape[0] < self.cfg.window_samples:
            self.rec_status.config(text="Too little data captured - not saved."); return
        folder = recordings.save_recording(self.cfg, sig, labels,
                                           description=self.v_desc.get().strip(), training=True)
        self.rec_status.config(text=f"Saved: {folder.name}  ({sig.shape[0]} samples)")

    def stop(self):
        try:
            self.engine.stop()
        except Exception:
            pass
