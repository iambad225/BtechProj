"""EEG modality tab: Config (10-20 selector), Monitor (band powers + alpha test), Record.

EEG *classification* is deferred (see the Train tab's provisional note); this tab captures
config + data now so models can be trained once the EEG paradigms are designed.
"""
from __future__ import annotations

import queue
import random
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

from core.config import Config, default_config, Electrode, mi_errp_montage
from core.dsp import RealTimeFilter
from core import features_eeg as EF
from core.montage import OCCIPITAL
from core.eeg_models import bandpass
from core.quality import channel_status, dead_channels
from sources import create_source
from sources.synthetic_eeg import EEGSyntheticSource
import storage.recordings as recordings
import storage.models as models
from gui.widgets.waveform import Waveform
from gui.widgets.bars import StackedBars, LevelBars
from gui.widgets.montage_10_20 import Montage1020
from gui.widgets.profile_bar import ProfileBar
from gui.widgets.waterfall import Waterfall
from scipy.signal import welch

CONFIG_PATH = "config/eeg.yaml"
BAND_COLORS = {"delta": "#6c5ce7", "theta": "#0984e3", "alpha": "#e17055",
               "beta": "#00b894", "gamma": "#d63031"}
# EEG paradigms that run continuously on a sliding window (band, short tag).
# MI is now rest/left/right, so its own 'rest' class is the gate (no separate ERD model).
LIVE_PARADIGMS = {"mi": ((8.0, 30.0), "MI")}


def mi_cue_order(classes, reps, rng=random):
    """Block-randomised MI cue order.

    Each rep is a random permutation of ``rest`` + the MI classes, so every class occurs exactly
    ``reps`` times and is spread evenly over the run (slow drift/fatigue can't line up with one
    class). The same cue never appears twice in a row: the get-ready gap between blocks isn't
    recorded, so two adjacent same-label blocks would be spliced into a single trial.
    """
    items = ["rest"] + list(classes)
    order, prev = [], None
    for _ in range(reps):
        block = rng.sample(items, k=len(items))
        while block[0] == prev:
            block = rng.sample(items, k=len(items))
        order += block
        prev = block[-1]
    return order


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
        self._rawbuf = np.zeros((0, cfg.n_channels))   # last 1 s of RAW samples (quality check)
        self.quality = []                              # per-channel 'ok'|'near'|'railed'|'flat'
        self._schedule = []
        self._seg = None
        self._seg_end = 0.0
        self._rec_sig = []
        self._rec_lab = []
        # errp event-locked capture
        self._errp_sig = []
        self._errp_markers = []
        self._errp_total = 0
        # live inference (one or more trained models)
        self.live_specs = []
        self._live_buf = np.zeros((0, cfg.n_channels))
        self._live_span = int(cfg.sample_rate * 4)
        self._live_step_s = 0.25
        self._live_t = 0.0
        self.live_ms = 0.0

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
            self._rawbuf = np.zeros((0, self.cfg.n_channels)); self.quality = []
        self.mode = "monitor"

    def start_record(self, schedule):
        self._schedule = list(schedule); self._seg = None
        self._rec_sig = []; self._rec_lab = []; self._filt.reset()
        self.mode = "record"

    def start_errp(self):
        with self._lock:
            self._errp_sig = []; self._errp_markers = []; self._errp_total = 0
        self.mode = "errp"

    def start_live(self, specs):
        """specs: list of dicts {model, cols, band, labels, win, tag, ...}; adds live results."""
        self.live_specs = specs
        for s in specs:
            s["proba"] = None
            s["top"] = -1
        wins = [s["win"] for s in specs] or [self.cfg.window_samples]
        self._live_span = max(wins) + self.cfg.sample_rate      # +1 s of margin
        self._live_step_s = max(self.cfg.step_ms, 150) / 1000.0
        with self._lock:
            self._live_buf = np.zeros((0, self.cfg.n_channels))
        self.live_ms = 0.0; self._live_t = 0.0
        self.mode = "live"

    def _do_live(self, new):
        if new.shape[0]:
            with self._lock:
                self._live_buf = np.vstack([self._live_buf, new])[-self._live_span:]
        now = time.perf_counter()
        if now - self._live_t < self._live_step_s:
            return
        self._live_t = now
        with self._lock:
            buf = self._live_buf.copy()
        t0 = time.perf_counter()
        for s in self.live_specs:
            if buf.shape[0] < s["win"]:
                continue
            w = buf[-s["win"]:][:, s["cols"]]
            wf = bandpass(w, self.cfg.sample_rate, s["band"][0], s["band"][1])
            try:
                proba = np.asarray(s["model"].predict_proba([wf])[0], dtype=float)
            except Exception:
                continue
            with self._lock:
                s["proba"] = proba
                s["top"] = int(np.argmax(proba))
        self.live_ms = (time.perf_counter() - t0) * 1000.0

    def mark_event(self, label):
        """Mark a feedback event ('error'/'correct') at the current sample position."""
        with self._lock:
            self._errp_markers.append((self._errp_total, label))

    def finish_errp(self):
        with self._lock:
            sig = np.vstack(self._errp_sig) if self._errp_sig else np.zeros((0, self.cfg.n_channels))
            markers = list(self._errp_markers)
            self.mode = "idle"
        labels = ["none"] * sig.shape[0]
        for idx, lab in markers:
            if 0 <= idx < len(labels):
                labels[idx] = lab
        self.events.put(("recorded", sig, labels))

    def _loop(self):
        while self._running:
            new = self.source.get_data() if self.source else np.zeros((0, self.cfg.n_channels))
            if self.mode == "monitor":
                self._do_monitor(new)
            elif self.mode == "live":
                self._do_live(new)
            elif self.mode == "record":
                self._do_record(new)
            elif self.mode == "errp":
                if new.shape[0]:
                    with self._lock:
                        self._errp_sig.append(new.copy())
                        self._errp_total += new.shape[0]
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
                self._rawbuf = np.vstack([self._rawbuf, new])[-self.cfg.sample_rate:]
        dt = time.perf_counter() - self._t
        if dt >= self.cfg.step_ms / 1000.0:
            with self._lock:
                buf = self._buf.copy(); rawbuf = self._rawbuf.copy()
            q = channel_status(rawbuf)       # on raw: filtering + CAR would hide a railed channel
            with self._lock:
                self.quality = q
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
            self.events.put(("cue", self._seg[0], self._seg[1], self._seg[2]))
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

        self._cue_deadline = None      # GUI-side countdown for the record cue

        nb = ttk.Notebook(self); nb.grid(sticky="nsew")
        self.t_cfg = ttk.Frame(nb, padding=8)
        self.t_mon = ttk.Frame(nb, padding=8)
        self.t_rec = ttk.Frame(nb, padding=8)
        self.t_live = ttk.Frame(nb, padding=8)
        nb.add(self.t_cfg, text="  Config  ")
        nb.add(self.t_mon, text="  Monitor  ")
        nb.add(self.t_rec, text="  Record  ")
        nb.add(self.t_live, text="  Live  ")
        nb.bind("<<NotebookTabChanged>>", lambda e: self._on_tab())
        self._build_config()
        self._build_monitor()
        self._build_record()
        self._build_live()
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
        t.columnconfigure(1, weight=1); t.rowconfigure(1, weight=1)
        self.profile_bar = ProfileBar(t, "eeg", get_cfg=self._collect_cfg, on_load=self._on_profile_load)
        self.profile_bar.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))
        self.montage = Montage1020(t)
        self.montage.grid(row=1, column=1, sticky="nsew", padx=(8, 0))
        self.montage.set_electrodes(self.cfg.electrodes)

        form = ttk.Frame(t); form.grid(row=1, column=0, sticky="nw")
        r = 0
        def row(label, var, width=14):
            nonlocal r
            ttk.Label(form, text=label).grid(row=r, column=0, sticky="e", padx=4, pady=2)
            ttk.Entry(form, textvariable=var, width=width).grid(row=r, column=1, sticky="w"); r += 1
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
        ttk.Button(form, text="Load MI/ErrP preset", command=self._load_mi_preset).grid(
            row=r, column=0, sticky="w", pady=(10, 2))
        ttk.Button(form, text="Apply & Save", command=self._apply).grid(row=r + 1, column=0, sticky="w")
        self.cfg_status = ttk.Label(form, text="", foreground="#0a7")
        self.cfg_status.grid(row=r + 2, column=0, columnspan=2, sticky="w", pady=(4, 0))

    def _load_mi_preset(self):
        """Load the FCz/CP-free 8-ch montage for MI + ERD + ErrP and MI class labels."""
        self.montage.set_electrodes(mi_errp_montage())
        self.actions_list.delete(0, tk.END)
        for a in ["mi_left", "mi_right"]:
            self.actions_list.insert(tk.END, a)
        self.cfg_status.config(text="Loaded montage Fz,FC5,FC6,C3,Cz,C4,P3,P4 + MI labels. Click Apply & Save.")

    def _add_action(self):
        a = self.v_newaction.get().strip().replace(" ", "_")
        if a and a not in self.actions_list.get(0, tk.END):
            self.actions_list.insert(tk.END, a); self.v_newaction.delete(0, tk.END)

    def _remove_action(self):
        sel = self.actions_list.curselection()
        if sel:
            self.actions_list.delete(sel[0])

    def _collect_cfg(self):
        """Read the Config tab widgets into self.cfg (raises ValueError on invalid input)."""
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
        occ = [e.name for e in data if e.name in OCCIPITAL]
        if occ:
            self.cfg.alpha_electrodes = occ
        else:                                   # posterior-most pair (alpha is posterior)
            from core.montage import NODES
            ranked = sorted((e.name for e in data if e.name in NODES),
                            key=lambda n: NODES[n][1], reverse=True)
            self.cfg.alpha_electrodes = ranked[:2] or [e.name for e in data][-2:]
        return self.cfg

    def _apply(self):
        try:
            self._collect_cfg()
        except Exception as e:
            messagebox.showerror("Invalid config", str(e)); return
        self.cfg.save(CONFIG_PATH)
        self.engine.stop(); self.engine = EEGEngine(self.cfg)
        self._build_band_rows()
        self.cfg_status.config(text=f"Saved -> {CONFIG_PATH} ({self.cfg.n_channels} ch; "
                                    f"alpha on {self.cfg.alpha_electrodes})")

    def _populate_config(self, cfg):
        """Set the Config tab widgets from a Config (used when loading a profile)."""
        self.montage.set_electrodes(cfg.electrodes)
        self.v_src.set(cfg.source.type)
        self.v_port.set(cfg.source.serial_port)
        self.v_sr.set(str(cfg.sample_rate))
        self.v_notch.set(str(cfg.filter.notch_hz))
        self.v_lo.set(str(cfg.filter.bandpass_low))
        self.v_hi.set(str(cfg.filter.bandpass_high))
        self.v_car.set(cfg.car)
        self.v_win.set(str(cfg.window_ms))
        self.v_step.set(str(cfg.step_ms))
        self.v_alpha_thr.set(str(cfg.alpha_threshold))
        self.actions_list.delete(0, tk.END)
        for a in cfg.labels:
            self.actions_list.insert(tk.END, a)

    def _on_profile_load(self, cfg):
        cfg.modality = "eeg"
        self.cfg = cfg
        self._populate_config(cfg)
        cfg.save(CONFIG_PATH)
        self.engine.stop(); self.engine = EEGEngine(self.cfg)
        self._build_band_rows()
        self.cfg_status.config(text=f"Profile applied ({self.cfg.n_channels} ch).")

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
        self.show_wf = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="waterfall", variable=self.show_wf,
                        command=self._toggle_waterfall).grid(row=0, column=5, padx=8)
        # raw-signal quality (railed/flat electrodes are invisible after filtering + CAR)
        self.mon_quality = ttk.Label(bar, text="", foreground="#c0392b", font=("", 9, "bold"))
        self.mon_quality.grid(row=0, column=6, padx=12, sticky="w")

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

        rightcol = ttk.Frame(main); rightcol.grid(row=0, column=1, sticky="nsew")
        rightcol.columnconfigure(0, weight=1)
        rightcol.rowconfigure(0, weight=3); rightcol.rowconfigure(1, weight=2)
        right = ttk.LabelFrame(rightcol, text="EEG traces", padding=4)
        right.grid(row=0, column=0, sticky="nsew")
        right.columnconfigure(0, weight=1); right.rowconfigure(0, weight=1)
        self.wave = Waveform(right); self.wave.grid(row=0, column=0, sticky="nsew")
        self.wf_frame = ttk.LabelFrame(rightcol, text="Waterfall — power spectrum over time (Hz vs t)", padding=4)
        self.wf_frame.columnconfigure(0, weight=1); self.wf_frame.rowconfigure(0, weight=1)
        self.waterfall = Waterfall(self.wf_frame, fmax=self.cfg.filter.bandpass_high)
        self.waterfall.grid(row=0, column=0, sticky="nsew")   # wf_frame shown only when toggled

    def _toggle_waterfall(self):
        if self.show_wf.get():
            self.waterfall.set_fmax(self.cfg.filter.bandpass_high)
            self.waterfall.clear()
            self.wf_frame.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        else:
            self.wf_frame.grid_remove()

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
        t.columnconfigure(1, weight=1); t.rowconfigure(0, weight=1)

        # -- left: compact controls --
        left = ttk.Frame(t); left.grid(row=0, column=0, sticky="nw", padx=(0, 12))
        ttk.Label(left, text="Record a training session (raw data + JSON).\nLoad the MI/ErrP "
                            "preset in Config first.", foreground="#555").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        ttk.Label(left, text="Subject").grid(row=1, column=0, sticky="e", padx=4, pady=2)
        self.v_subject = tk.StringVar(value=self.cfg.subject)
        ttk.Entry(left, textvariable=self.v_subject, width=18).grid(row=1, column=1, sticky="w")
        ttk.Label(left, text="Paradigm").grid(row=2, column=0, sticky="e", padx=4, pady=2)
        self.v_paradigm = tk.StringVar(value="MI (left/right)")
        pc = ttk.Combobox(left, textvariable=self.v_paradigm, state="readonly", width=18,
                          values=["MI (left/right)", "ErrP (feedback task)"])
        pc.grid(row=2, column=1, sticky="w")
        pc.bind("<<ComboboxSelected>>", lambda e: self._rec_mode())

        self.mi_frame = ttk.Frame(left)
        self.mi_frame.grid(row=3, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Label(self.mi_frame, text="Seconds/class").grid(row=0, column=0, sticky="e", padx=4)
        self.v_secs = ttk.Spinbox(self.mi_frame, from_=2, to=30, width=5); self.v_secs.set(4)
        self.v_secs.grid(row=0, column=1, sticky="w")
        ttk.Label(self.mi_frame, text="Reps").grid(row=1, column=0, sticky="e", padx=4)
        self.v_reps = ttk.Spinbox(self.mi_frame, from_=1, to=15, width=5); self.v_reps.set(6)
        self.v_reps.grid(row=1, column=1, sticky="w")

        self.errp_frame = ttk.Frame(left)
        self.errp_frame.grid(row=4, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Label(self.errp_frame, text="Trials").grid(row=0, column=0, sticky="e", padx=4)
        self.v_trials = ttk.Spinbox(self.errp_frame, from_=20, to=400, width=5); self.v_trials.set(60)
        self.v_trials.grid(row=0, column=1, sticky="w")
        ttk.Label(self.errp_frame, text="Error rate").grid(row=1, column=0, sticky="e", padx=4)
        self.v_errrate = ttk.Spinbox(self.errp_frame, from_=0.1, to=0.5, increment=0.05, width=5)
        self.v_errrate.set(0.30); self.v_errrate.grid(row=1, column=1, sticky="w")

        ttk.Label(left, text="Description").grid(row=5, column=0, sticky="e", padx=4, pady=2)
        self.v_desc = tk.StringVar()
        ttk.Entry(left, textvariable=self.v_desc, width=18).grid(row=5, column=1, sticky="w")
        self.rec_btn = ttk.Button(left, text="● Record session", command=self._record)
        self.rec_btn.grid(row=6, column=0, columnspan=2, sticky="w", pady=8)
        self.rec_status = ttk.Label(left, text="", foreground="#0a7", wraplength=220)
        self.rec_status.grid(row=7, column=0, columnspan=2, sticky="w")

        # -- right: large subject-facing cue display --
        cue = ttk.LabelFrame(t, text="Cue", padding=2)
        cue.grid(row=0, column=1, sticky="nsew")
        cue.columnconfigure(0, weight=1); cue.rowconfigure(0, weight=1)
        self.cue_canvas = tk.Canvas(cue, bg="#111", highlightthickness=0)
        self.cue_canvas.grid(row=0, column=0, sticky="nsew")
        # small, off-centre countdown so it doesn't drive a visual response into the EEG
        self.cue_timer = tk.Label(self.cue_canvas, text="", fg="#666", bg="#111",
                                  font=("", 11), bd=0)
        self.cue_timer.place(relx=0.985, rely=0.02, anchor="ne")
        self.cue_canvas.bind("<Configure>", lambda e: self._cue_idle())
        self._rec_mode()

    def _rec_mode(self):
        if self.v_paradigm.get().startswith("ErrP"):
            self.mi_frame.grid_remove(); self.errp_frame.grid()
        else:
            self.errp_frame.grid_remove(); self.mi_frame.grid()

    # ----- big cue rendering (subject-facing) -----
    def _cue_dims(self):
        cv = self.cue_canvas
        w = cv.winfo_width();  w = w if w > 1 else 640
        h = cv.winfo_height(); h = h if h > 1 else 420
        return cv, w, h

    def _cue_idle(self):
        cv, w, h = self._cue_dims(); cv.delete("all")
        cv.create_text(w / 2, h / 2, text="+", fill="#555", font=("", max(24, int(h * 0.16)), "bold"))

    def _cue_fixation(self):
        cv, w, h = self._cue_dims(); cv.delete("all")
        cv.create_text(w / 2, h / 2, text="+", fill="#ddd", font=("", max(30, int(h * 0.22)), "bold"))

    def _cue_text(self, text, color="#eee", sub=""):
        cv, w, h = self._cue_dims(); cv.delete("all")
        cv.create_text(w / 2, h / 2, text=text, fill=color, font=("", max(24, int(h * 0.16)), "bold"))
        if sub:
            cv.create_text(w / 2, h * 0.85, text=sub, fill="#888", font=("", max(10, int(h * 0.05))))

    def _cue_arrow(self, direction, color, label="", sub=""):
        cv, w, h = self._cue_dims(); cv.delete("all")
        cy = h * 0.45; ah = h * 0.16
        if direction == "left":
            cv.create_polygon(w * 0.58, cy - ah, w * 0.58, cy + ah, w * 0.34, cy, fill=color, width=0)
        else:
            cv.create_polygon(w * 0.42, cy - ah, w * 0.42, cy + ah, w * 0.66, cy, fill=color, width=0)
        if label:
            cv.create_text(w / 2, h * 0.78, text=label, fill=color, font=("", max(16, int(h * 0.09)), "bold"))
        if sub:
            cv.create_text(w / 2, h * 0.9, text=sub, fill="#888", font=("", max(10, int(h * 0.045))))

    def _show_cue(self, phase, label, duration):
        """Drive the big cue from a record schedule segment ('ready'/'record')."""
        self._cue_deadline = time.perf_counter() + float(duration)
        ready = (phase == "ready")
        low = label.lower()
        go, prep = "#f1c40f", "get ready"
        if low == "rest":
            self._cue_text("REST", color="#4a90d9", sub=prep if ready else "")
        elif "left" in low:
            self._cue_arrow("left", go if ready else "#2ecc40", "LEFT", sub=prep if ready else "")
        elif "right" in low:
            self._cue_arrow("right", go if ready else "#2ecc40", "RIGHT", sub=prep if ready else "")
        else:
            self._cue_text(label.upper(), color=go if ready else "#2ecc40", sub=prep if ready else "")

    def _record(self):
        self.cfg.subject = self.v_subject.get().strip() or "subject"
        self.rec_status.config(foreground="#0a7")
        self.rec_btn.config(state="disabled")
        try:
            self.engine.start()
        except Exception as e:
            self.rec_btn.config(state="normal"); messagebox.showerror("Source error", str(e)); return
        if self.v_paradigm.get().startswith("ErrP"):
            self._record_errp()
        else:
            self._record_mi()

    def _record_mi(self):
        secs = float(self.v_secs.get()); reps = int(self.v_reps.get())
        classes = [l for l in self.cfg.labels if l.lower() != "rest"] or ["mi_left", "mi_right"]
        self.cfg.paradigm = "mi"; self.cfg.labels = classes
        schedule = []
        for label in mi_cue_order(classes, reps):
            schedule.append(("ready", label, 2.0))
            schedule.append(("record", label, secs))
        self.engine.start_record(schedule)
        self.rec_status.config(text=f"Recording MI — {reps} each of rest, {', '.join(classes)} "
                                    "in random order. Follow the cue.")

    # ----- ErrP cued feedback task (uses the same big cue canvas) -----
    def _record_errp(self):
        self.cfg.paradigm = "errp"; self.cfg.labels = ["correct", "error"]
        self._errp_n = int(self.v_trials.get())
        self._errp_left = self._errp_n
        self._errp_rate = float(self.v_errrate.get())
        self._cue_deadline = None            # ErrP is fast/event-locked: no countdown
        self.engine.start_errp()
        self.rec_status.config(text=f"ErrP task: 0/{self._errp_n}")
        self.after(500, self._errp_next)

    def _errp_next(self):
        if self._errp_left <= 0:
            self.engine.finish_errp(); self._cue_text("done", color="#888")
            return
        self._cue_fixation()
        self.after(700 + random.randint(0, 400), self._errp_cue)

    def _errp_cue(self):
        self._errp_target = random.choice(["left", "right"])
        self._cue_arrow(self._errp_target, "#4a90d9", "TARGET")
        self.after(900, self._errp_outcome)

    def _errp_outcome(self):
        is_err = random.random() < self._errp_rate
        outcome = ("right" if self._errp_target == "left" else "left") if is_err else self._errp_target
        self.engine.mark_event("error" if is_err else "correct")
        self._cue_arrow(outcome, "#e74c3c" if is_err else "#2ecc40",
                        "WRONG" if is_err else "OK")
        self._errp_left -= 1
        self.rec_status.config(text=f"ErrP task: {self._errp_n - self._errp_left}/{self._errp_n}")
        self.after(1000, self._errp_next)

    # ---------- live (run trained models on the stream) ----------
    def _build_live(self):
        t = self.t_live
        t.columnconfigure(0, weight=1); t.rowconfigure(3, weight=1)
        ttk.Label(t, text="Run one or more trained EEG models on the live stream. "
                          "Ctrl/Shift-click to select several (e.g. ERD gate + MI). "
                          "The current config's electrodes must include the model's.",
                  foreground="#555").grid(row=0, column=0, sticky="w", pady=(0, 6))
        bar = ttk.Frame(t); bar.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        bar.columnconfigure(1, weight=1)
        ttk.Label(bar, text="Models").grid(row=0, column=0, sticky="nw")
        lb = ttk.Frame(bar); lb.grid(row=0, column=1, sticky="ew", padx=6)
        lb.columnconfigure(0, weight=1)
        self.live_list = tk.Listbox(lb, selectmode="extended", exportselection=False, height=4)
        self.live_list.grid(row=0, column=0, sticky="ew")
        xs = ttk.Scrollbar(lb, orient="horizontal", command=self.live_list.xview)
        xs.grid(row=1, column=0, sticky="ew")
        self.live_list.config(xscrollcommand=xs.set)
        col = ttk.Frame(bar); col.grid(row=0, column=2, sticky="nw")
        ttk.Button(col, text="refresh", command=self._refresh_models).grid(row=0, column=0, sticky="ew")
        self.live_btn = ttk.Button(col, text="▶ Start", command=self._toggle_live)
        self.live_btn.grid(row=1, column=0, sticky="ew", pady=3)
        self.live_status = ttk.Label(col, text="stopped", foreground="#a00"); self.live_status.grid(row=2, column=0, sticky="w")
        self.live_lat = ttk.Label(col, text="", foreground="#555"); self.live_lat.grid(row=3, column=0, sticky="w")
        self.live_decision = ttk.Label(t, text="DECISION:  -", font=("", 15, "bold"))
        self.live_decision.grid(row=2, column=0, sticky="nw", pady=(4, 6))
        self.live_panels_frame = ttk.Frame(t); self.live_panels_frame.grid(row=3, column=0, sticky="nsew")
        self.live_panels_frame.columnconfigure(0, weight=1)
        self.live_panels = []
        self._eeg_models = []
        self._refresh_models()

    def _refresh_models(self):
        self._eeg_models = models.list_models("eeg")
        self.live_list.delete(0, tk.END)
        for e in self._eeg_models:
            m = e["meta"]
            self.live_list.insert(tk.END,
                f"[{m.paradigm or m.model_type}] {e['folder'].name}   ({', '.join(m.labels)})")

    def _toggle_live(self):
        if self.engine.mode == "live":
            self.engine.go_idle()
            self.live_btn.config(text="▶ Start"); self.live_status.config(text="stopped", foreground="#a00")
            return
        sel = self.live_list.curselection()
        if not sel:
            messagebox.showinfo("No model", "Select one or more trained EEG models."); return
        cur = self.cfg.electrode_names
        specs, skipped = [], []
        for i in sel:
            e = self._eeg_models[i]
            model, meta = models.load_model(e["folder"])
            paradigm = (meta.paradigm or getattr(model, "kind", "")).lower()
            if paradigm not in LIVE_PARADIGMS:
                skipped.append(f"{e['folder'].name} ({paradigm or '?'})"); continue
            data_ch = sorted((c for c in meta.channels if c.role == "data"), key=lambda c: c.channel)
            model_elecs = [c.electrode for c in data_ch]
            missing = [n for n in model_elecs if n not in cur]
            if missing:
                messagebox.showerror("Electrode mismatch",
                    f"{e['folder'].name} needs {missing}, not in the current montage {cur}.\n"
                    "Load a matching config/profile first."); return
            band, tag = LIVE_PARADIGMS[paradigm]
            win = max(1, round(self.cfg.sample_rate * (meta.window_ms or self.cfg.window_ms) / 1000))
            specs.append({"model": model, "labels": list(meta.labels),
                          "cols": [cur.index(n) for n in model_elecs], "band": band,
                          "win": win, "tag": tag, "name": e["folder"].name,
                          "proba": None, "top": -1})
        if not specs:
            messagebox.showinfo("Nothing to run",
                "Selected models can't run continuously — ErrP is event-locked "
                "(use the Record feedback task).\nSkipped: " + ", ".join(skipped)); return
        try:
            self.engine.start(); self.engine.start_live(specs)
        except Exception as ex:
            messagebox.showerror("Source error", str(ex)); return
        self._build_live_panels(specs)
        note = f"  (skipped event-locked: {len(skipped)})" if skipped else ""
        self.live_btn.config(text="■ Stop"); self.live_status.config(text="running" + note, foreground="#0a0")

    def _build_live_panels(self, specs):
        for p in self.live_panels:
            p["frame"].destroy()
        self.live_panels = []
        for r, s in enumerate(specs):
            self.live_panels_frame.rowconfigure(r, weight=1)
            fr = ttk.LabelFrame(self.live_panels_frame, text=f"[{s['tag']}] {s['name']}", padding=6)
            fr.grid(row=r, column=0, sticky="nsew", pady=3)
            fr.columnconfigure(0, weight=1); fr.rowconfigure(1, weight=1)
            active = ttk.Label(fr, text="→ -", font=("", 14, "bold")); active.grid(row=0, column=0, sticky="w")
            bars = LevelBars(fr, s["labels"]); bars.grid(row=1, column=0, sticky="nsew")
            self.live_panels.append({"spec": s, "frame": fr, "bars": bars, "active": active})

    def _update_live(self):
        parts = []
        with self.engine._lock:
            snaps = [(p, None if p["spec"]["proba"] is None else list(p["spec"]["proba"]),
                      p["spec"]["top"]) for p in self.live_panels]
            lat = self.engine.live_ms
        for p, proba, top in snaps:
            labels = p["spec"]["labels"]
            if proba is None or len(proba) != len(p["bars"].rows):
                continue
            for i in range(len(labels)):
                p["bars"].set(i, proba[i], f"{proba[i]*100:0.0f}%", on=(i == top))
            if 0 <= top < len(labels):
                p["active"].config(text=f"→ {labels[top]}   {proba[top]*100:0.0f}%")
                parts.append(f"{p['spec']['tag']}={labels[top]}")
        self.live_decision.config(text="DECISION:  " + ("      ".join(parts) if parts else "-"))
        self.live_lat.config(text=f"{lat:0.1f} ms")

    # ---------- shared ----------
    def _on_tab(self):
        if self.engine.mode in ("monitor", "live"):
            self.engine.go_idle()
            self.mon_btn.config(text="▶ Start"); self.mon_status.config(text="stopped", foreground="#a00")
            if hasattr(self, "live_btn"):
                self.live_btn.config(text="▶ Start"); self.live_status.config(text="stopped", foreground="#a00")

    def _tick(self):
        try:
            while True:
                kind, *rest = self.engine.events.get_nowait()
                if kind == "status":
                    self.rec_status.config(text=rest[0])
                elif kind == "cue":
                    self._show_cue(rest[0], rest[1], rest[2])
                elif kind == "recorded":
                    self._cue_deadline = None; self._cue_idle()
                    self._on_recorded(rest[0], rest[1])
        except queue.Empty:
            pass
        # inconspicuous MI countdown in the corner
        if self._cue_deadline is not None and self.engine.mode == "record":
            rem = self._cue_deadline - time.perf_counter()
            self.cue_timer.config(text=f"{rem:0.0f}" if rem > 0.05 else "")
        elif self.engine.mode != "record":
            self.cue_timer.config(text="")
        if self.engine.mode == "live":
            self._update_live()
        if self.engine.mode == "monitor":
            with self.engine._lock:
                buf = self.engine.raw.copy(); bands = self.engine.bands
                bb = self.engine.broadband.copy(); alpha = self.engine.alpha; sps = self.engine.sps
                quality = list(self.engine.quality)
            names = self.cfg.electrode_names
            self.wave.draw(buf, fs=self.cfg.sample_rate, labels=names, flags=quality)
            dead = [n for n, s in zip(names, quality) if s in ("railed", "flat")]
            near = [n for n, s in zip(names, quality) if s == "near"]
            if dead:
                self.mon_quality.config(text=f"No signal (railed/flat): {', '.join(dead)} — "
                                             "check contact/wiring", foreground="#c0392b")
            elif near:
                self.mon_quality.config(text=f"Near rail: {', '.join(near)} — reseat",
                                        foreground="#b9770e")
            else:
                self.mon_quality.config(text="")
            if bands:
                for i in range(len(self.band_bars.rows)):
                    total = float(bb[i]) if i < len(bb) else 0.0
                    if total > 1e-9:
                        self.band_bars.set(i, {b: float(bands[b][i]) / total for b in BAND_COLORS if b in bands})
            if self.show_wf.get() and buf.shape[0] >= self.cfg.window_samples:
                win = buf[-self.cfg.window_samples:]
                f, psd = welch(win, fs=self.cfg.sample_rate, axis=0,
                               nperseg=int(min(win.shape[0], 256)))
                self.waterfall.push(psd.mean(axis=1), f)
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
        dead = dead_channels(sig, self.cfg.electrode_names)
        if dead:
            self.rec_status.config(
                text=f"Saved {folder.name}, but {', '.join(dead)} had no signal (railed/flat), so "
                     "this run can't train. Fix contact/wiring and re-record.", foreground="#c0392b")
        else:
            self.rec_status.config(text=f"Saved: {folder.name}  ({sig.shape[0]} samples)",
                                   foreground="#0a7")

    def stop(self):
        try:
            self.engine.stop()
        except Exception:
            pass
