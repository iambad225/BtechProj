"""EEG modality tab: Config (10-20 selector), Monitor (band powers + alpha test), Record.

EEG *classification* is deferred (see the Train tab's provisional note); this tab captures
config + data now so models can be trained once the EEG paradigms are designed.
"""
from __future__ import annotations

import queue
from collections import deque
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
from core.eeg_models import StreamBandpass, epoch_signal
from core.eeg_training import ERRP_BAND, ERRP_TMIN, ERRP_TMAX
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
        self._live_filt = {}                # band -> StreamBandpass (causal, as in training)
        self._live_fbuf = {}                # band -> filtered stream buffer
        self._live_span = int(cfg.sample_rate * 4)
        self._live_step_s = 0.25
        self._live_t = 0.0
        self.live_ms = 0.0
        # live ErrP test (event-locked: the GUI marks each outcome, the engine classifies it)
        self._el_buf = np.zeros((0, cfg.n_channels))
        self._el_gend = None               # global index just past _el_buf's newest sample
        self._el_pending = []              # (global_sample_index, label) awaiting post-event data
        self._el_spec = None               # {"model": ErrPModel, "cols": [...]}
        self._el_qt = 0.0
        self._errp_g0 = None               # global index of the first recorded ErrP sample
        self._clock_reset()

    # ----- stream clock: PC time <-> sample index, immune to buffering bursts -----
    # Sample k is acquired at off + k/fs. A chunk whose newest sample k arrives at time t gives
    # t - k/fs >= off, with equality only when nothing delayed it; the running minimum over a
    # window therefore tracks the stream's fastest path (and slow PC/board clock drift), while
    # late or bursty deliveries can't move it.
    CLOCK_WIN_S, STALE_WIN_S = 20.0, 5.0

    def _clock_reset(self):
        self._rx_total = 0                  # samples received since the source started
        self._clk = deque()                 # monotonic min-deque of (t_arrival, t - k/fs)
        self._stale = deque()               # (t, ms the newest sample lags the fastest path)

    def _clock_update(self, t, n):
        """Called once per loop iteration with the chunk size n (may be 0). Returns the global
        index just past the newest received sample."""
        fs = self.cfg.sample_rate
        with self._lock:
            if n:
                self._rx_total += n
                v = t - (self._rx_total - 1) / fs
                dq = self._clk
                while dq and dq[-1][1] >= v:
                    dq.pop()
                dq.append((t, v))
                while dq[0][0] < t - self.CLOCK_WIN_S:
                    dq.popleft()
            if self._clk:
                lag = (t - self._clk[0][1]) - (self._rx_total - 1) / fs
                self._stale.append((t, max(0.0, lag) * 1000.0))
                while self._stale and self._stale[0][0] < t - self.STALE_WIN_S:
                    self._stale.popleft()
            return self._rx_total

    def index_at(self, t):
        """Global index of the sample acquired at PC time ``t`` (time.perf_counter), offset only by
        the stream's fixed minimum latency - so it is the same in Record and ErrP Live."""
        with self._lock:
            if not self._clk:
                return self._rx_total
            off = self._clk[0][1]
        return int(round((t - off) * self.cfg.sample_rate))

    def latency_stats(self):
        """How stale the newest sample is (ms beyond the fastest path) over the last few seconds:
        {"typ": median, "p95": 95th percentile, "max": worst}, or None before data arrives."""
        with self._lock:
            lags = [ms for _, ms in self._stale]
        if len(lags) < 20:
            return None
        a = np.asarray(lags)
        return {"typ": float(np.median(a)), "p95": float(np.percentile(a, 95)), "max": float(a.max())}

    def start(self):
        if self._running:
            return
        self.source = create_source(self.cfg)
        self.source.start()
        self._filt = RealTimeFilter(self.cfg.sample_rate, self.cfg.n_channels,
                                    notch_hz=self.cfg.filter.notch_hz,
                                    bandpass=(self.cfg.filter.bandpass_low, self.cfg.filter.bandpass_high),
                                    enabled=self.cfg.filter.enabled)
        self._clock_reset()
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
            self._errp_g0 = None
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
        # one causal filter per band, run continuously over the stream (as in training), so the
        # newest samples of a window are valid rather than distorted by a filter edge
        bands = {tuple(s["band"]) for s in specs}
        self._live_filt = {b: StreamBandpass(self.cfg.sample_rate, b[0], b[1], self.cfg.n_channels)
                           for b in bands}
        with self._lock:
            self._live_fbuf = {b: np.zeros((0, self.cfg.n_channels)) for b in bands}
        self.live_ms = 0.0; self._live_t = 0.0
        self.mode = "live"

    def _do_live(self, new):
        if new.shape[0]:
            filtered = {b: f.process(new) for b, f in self._live_filt.items()}
            with self._lock:
                for b, y in filtered.items():
                    self._live_fbuf[b] = np.vstack([self._live_fbuf[b], y])[-self._live_span:]
        now = time.perf_counter()
        if now - self._live_t < self._live_step_s:
            return
        self._live_t = now
        with self._lock:
            fbuf = {b: v.copy() for b, v in self._live_fbuf.items()}
        t0 = time.perf_counter()
        for s in self.live_specs:
            buf = fbuf[tuple(s["band"])]
            if buf.shape[0] < s["win"]:
                continue
            wf = buf[-s["win"]:][:, s["cols"]]
            try:
                proba = np.asarray(s["model"].predict_proba([wf])[0], dtype=float)
            except Exception:
                continue
            with self._lock:
                s["proba"] = proba
                s["top"] = int(np.argmax(proba))
        self.live_ms = (time.perf_counter() - t0) * 1000.0

    def mark_event(self, label, t=None):
        """Mark a feedback event ('error'/'correct') shown at PC time ``t`` (default: now). The
        sample is located through the stream clock, not 'samples received so far', so buffering
        delays at that instant don't shift the marker."""
        g = self.index_at(time.perf_counter() if t is None else t)
        with self._lock:
            self._errp_markers.append((g, label))           # global index; localised at finish

    # ----- live ErrP test -----
    EL_KEEP_S = 8.0                        # seconds of stream kept for epoching

    def start_errp_live(self, spec):
        """spec: {"model": ErrPModel, "cols": model electrode columns}. Outcomes are marked with
        mark_live_event(); each is classified once enough post-event data has arrived and the
        result is posted as ("errp_result", label, p_error, dead_channels)."""
        with self._lock:
            self._el_buf = np.zeros((0, self.cfg.n_channels)); self._el_gend = None
            self._el_fbuf = np.zeros((0, self.cfg.n_channels))
            self._el_pending = []; self.quality = []
        self._el_filt = StreamBandpass(self.cfg.sample_rate, *ERRP_BAND, self.cfg.n_channels)
        self._el_spec = spec; self._el_qt = 0.0
        self.mode = "errp_live"

    def mark_live_event(self, label, t=None):
        g = self.index_at(time.perf_counter() if t is None else t)
        with self._lock:
            self._el_pending.append((g, label))

    def _do_errp_live(self, new, g_end):
        fs = self.cfg.sample_rate
        pre, post = int(round(-ERRP_TMIN * fs)), int(round(ERRP_TMAX * fs))
        keep = int(self.EL_KEEP_S * fs)
        y = self._el_filt.process(new) if new.shape[0] else None   # causal 1-10 Hz, as training
        with self._lock:
            if new.shape[0]:
                self._el_buf = np.vstack([self._el_buf, new])[-keep:]
                self._el_fbuf = np.vstack([self._el_fbuf, y])[-keep:]
                self._el_gend = g_end
            if self._el_gend is None:
                return
            raw, fbuf, total = self._el_buf, self._el_fbuf, self._el_gend
            # causal filtering needs nothing after the epoch: classify as soon as it ends
            ready = [ev for ev in self._el_pending if total >= ev[0] + post]
            self._el_pending = [ev for ev in self._el_pending if total < ev[0] + post]
        now = time.perf_counter()
        if now - self._el_qt > 0.5 and raw.shape[0]:
            self._el_qt = now
            q = channel_status(raw[-fs:])
            with self._lock:
                self.quality = q
        start = total - fbuf.shape[0]                                   # global index of fbuf[0]
        for g, label in ready:
            if g - pre < start:                                         # epoch already scrolled out
                self.events.put(("errp_result", label, None, [])); continue
            # baseline-corrected -0.2..+0.8 s epoch from the filtered stream: same as training
            eps = epoch_signal(fbuf, fs, [g - start], ERRP_TMIN, ERRP_TMAX, baseline=(ERRP_TMIN, 0.0))
            if not eps:
                self.events.put(("errp_result", label, None, [])); continue
            try:
                p = float(self._el_spec["model"].proba_error([eps[0][:, self._el_spec["cols"]]])[0])
            except Exception:
                p = None
            seg = raw[g - pre - start:g + post - start]
            self.events.put(("errp_result", label, p, dead_channels(seg, self.cfg.electrode_names)))

    def finish_errp(self):
        with self._lock:
            sig = np.vstack(self._errp_sig) if self._errp_sig else np.zeros((0, self.cfg.n_channels))
            markers = list(self._errp_markers)
            g0 = self._errp_g0 or 0
            self.mode = "idle"
        labels = ["none"] * sig.shape[0]
        for g, lab in markers:
            idx = g - g0                                     # global -> recording position
            if 0 <= idx < len(labels):
                labels[idx] = lab
        self.events.put(("recorded", sig, labels))

    def _loop(self):
        while self._running:
            new = self.source.get_data() if self.source else np.zeros((0, self.cfg.n_channels))
            g_end = self._clock_update(time.perf_counter(), new.shape[0])
            mode = self.mode
            if mode == "monitor":
                self._do_monitor(new)
            elif mode == "live":
                self._do_live(new)
            elif mode == "errp_live":
                self._do_errp_live(new, g_end)
            elif mode == "record":
                self._do_record(new)
            elif mode == "errp":
                if new.shape[0]:
                    with self._lock:
                        if self._errp_g0 is None:
                            self._errp_g0 = g_end - new.shape[0]
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
        self._nb = nb
        self.t_cfg = ttk.Frame(nb, padding=8)
        self.t_mon = ttk.Frame(nb, padding=8)
        self.t_rec = ttk.Frame(nb, padding=8)
        self.t_live = ttk.Frame(nb, padding=8)
        self.t_errp = ttk.Frame(nb, padding=8)
        nb.add(self.t_cfg, text="  Config  ")
        nb.add(self.t_mon, text="  Monitor  ")
        nb.add(self.t_rec, text="  Record  ")
        nb.add(self.t_live, text="  Live  ")
        nb.add(self.t_errp, text="  ErrP Live  ")
        nb.bind("<<NotebookTabChanged>>", lambda e: self._on_tab())
        self._build_config()
        self._build_monitor()
        self._build_record()
        self._build_live()
        self._build_errp_live()
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
    def _cue_dims(self, cv=None):
        cv = cv if cv is not None else self.cue_canvas
        w = cv.winfo_width();  w = w if w > 1 else 640
        h = cv.winfo_height(); h = h if h > 1 else 420
        return cv, w, h

    def _cue_idle(self, cv=None):
        cv, w, h = self._cue_dims(cv); cv.delete("all")
        cv.create_text(w / 2, h / 2, text="+", fill="#555", font=("", max(24, int(h * 0.16)), "bold"))

    def _cue_fixation(self, cv=None):
        cv, w, h = self._cue_dims(cv); cv.delete("all")
        cv.create_text(w / 2, h / 2, text="+", fill="#ddd", font=("", max(30, int(h * 0.22)), "bold"))

    def _cue_text(self, text, color="#eee", sub="", cv=None):
        cv, w, h = self._cue_dims(cv); cv.delete("all")
        cv.create_text(w / 2, h / 2, text=text, fill=color, font=("", max(24, int(h * 0.16)), "bold"))
        if sub:
            cv.create_text(w / 2, h * 0.85, text=sub, fill="#888", font=("", max(10, int(h * 0.05))))

    def _cue_arrow(self, direction, color, label="", sub="", cv=None):
        cv, w, h = self._cue_dims(cv); cv.delete("all")
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

    # ----- ErrP cued feedback task (shared by Record and the ErrP Live test) -----
    def _record_errp(self):
        self.cfg.paradigm = "errp"; self.cfg.labels = ["correct", "error"]
        self._cue_deadline = None            # ErrP is fast/event-locked: no countdown
        self.engine.start_errp()
        self._errp_task_start(
            int(self.v_trials.get()), float(self.v_errrate.get()), self.cue_canvas,
            on_event=lambda lab, ts: self.engine.mark_event(lab, ts),
            on_done=lambda: self.engine.finish_errp(),
            on_progress=lambda k, n: self.rec_status.config(text=f"ErrP task: {k}/{n}"),
            lead_ms=3000)       # same lead-in as ErrP Live: the causal filter settles first

    def _errp_task_start(self, n, rate, cv, on_event, on_done, on_progress, lead_ms=500):
        """Run n trials: fixation -> target arrow -> outcome arrow (wrong at ``rate``).
        ``on_event(label)`` fires at each outcome onset - the instant the ErrP is time-locked to."""
        self._errp_token = getattr(self, "_errp_token", 0) + 1
        self._errp_t = {"n": n, "left": n, "rate": rate, "cv": cv, "target": None,
                        "on_event": on_event, "on_done": on_done, "on_progress": on_progress}
        on_progress(0, n)
        self.after(lead_ms, self._errp_next, self._errp_token)

    def _errp_abort(self):
        self._errp_token = getattr(self, "_errp_token", 0) + 1     # orphans pending callbacks

    def _errp_next(self, token):
        if token != self._errp_token:
            return
        t = self._errp_t
        if t["left"] <= 0:
            self._cue_text("done", color="#888", cv=t["cv"]); t["on_done"]()
            return
        self._cue_fixation(cv=t["cv"])
        self.after(700 + random.randint(0, 400), self._errp_cue, token)

    def _errp_cue(self, token):
        if token != self._errp_token:
            return
        t = self._errp_t
        t["target"] = random.choice(["left", "right"])
        self._cue_arrow(t["target"], "#4a90d9", "TARGET", cv=t["cv"])
        self.after(900, self._errp_outcome, token)

    def _errp_outcome(self, token):
        if token != self._errp_token:
            return
        t = self._errp_t
        is_err = random.random() < t["rate"]
        outcome = ("right" if t["target"] == "left" else "left") if is_err else t["target"]
        self._cue_arrow(outcome, "#e74c3c" if is_err else "#2ecc40",
                        "WRONG" if is_err else "OK", cv=t["cv"])
        t["cv"].update_idletasks()               # paint the outcome now, then timestamp its onset
        t["on_event"]("error" if is_err else "correct", time.perf_counter())
        t["left"] -= 1
        t["on_progress"](t["n"] - t["left"], t["n"])
        self.after(1000, self._errp_next, token)

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

    @staticmethod
    def _delay_info(lat):
        """(short readout, warning or None) from EEGEngine.latency_stats()."""
        if not lat:
            return "", None
        txt = f"data delay typ {lat['typ']:.0f} / max {lat['max']:.0f} ms"
        warn = None
        if lat["p95"] > 50:
            warn = (f"Data arriving in bursts (up to {lat['max']:.0f} ms late). On Windows set the "
                    "dongle's FTDI latency timer to 1 ms (see README).")
        return txt, warn

    def _warn_old_filter(self, meta, name):
        if (meta.metrics or {}).get("filter") != "causal":
            messagebox.showwarning(
                "Retrain recommended",
                f"{name} was trained before the switch to causal filtering, so live results won't "
                "match its training scores. Retrain it on the Train tab (same recordings).")

    def _model_cols(self, meta, name):
        """Columns of the model's electrodes in the current stream (model order), or None after
        telling the user which electrodes the current montage is missing."""
        cur = self.cfg.electrode_names
        data_ch = sorted((c for c in meta.channels if c.role == "data"), key=lambda c: c.channel)
        model_elecs = [c.electrode for c in data_ch]
        missing = [n for n in model_elecs if n not in cur]
        if missing:
            messagebox.showerror("Electrode mismatch",
                f"{name} needs {missing}, not in the current montage {cur}.\n"
                "Load a matching config/profile first.")
            return None
        return [cur.index(n) for n in model_elecs]

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
            self._warn_old_filter(meta, e["folder"].name)
            paradigm = (meta.paradigm or getattr(model, "kind", "")).lower()
            if paradigm not in LIVE_PARADIGMS:
                skipped.append(f"{e['folder'].name} ({paradigm or '?'})"); continue
            cols = self._model_cols(meta, e["folder"].name)
            if cols is None:
                return
            band, tag = LIVE_PARADIGMS[paradigm]
            win = max(1, round(self.cfg.sample_rate * (meta.window_ms or self.cfg.window_ms) / 1000))
            specs.append({"model": model, "labels": list(meta.labels),
                          "cols": cols, "band": band,
                          "win": win, "tag": tag, "name": e["folder"].name,
                          "proba": None, "top": -1})
        if not specs:
            messagebox.showinfo("Nothing to run",
                "Selected models can't run continuously — ErrP is event-locked "
                "(test it on the ErrP Live tab).\nSkipped: " + ", ".join(skipped)); return
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
        dtxt, dwarn = self._delay_info(self.engine.latency_stats())
        self.live_lat.config(text=f"compute {lat:0.1f} ms" + (f"  \u00b7  {dtxt}" if dtxt else ""),
                             foreground="#b9770e" if dwarn else "#555")

    # ---------- ErrP Live (test a trained ErrP model on live data) ----------
    def _build_errp_live(self):
        t = self.t_errp
        t.columnconfigure(1, weight=1); t.rowconfigure(0, weight=1)
        left = ttk.Frame(t); left.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
        left.rowconfigure(12, weight=1)
        ttk.Label(left, text="Test a trained ErrP model on live EEG. The feedback task runs on the "
                            "right; each outcome is classified from the live stream and scored "
                            "against what was actually shown.",
                  foreground="#555", wraplength=330, justify="left").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        ttk.Label(left, text="Model").grid(row=1, column=0, sticky="e", padx=4)
        self.el_model = ttk.Combobox(left, state="readonly", width=38)
        self.el_model.grid(row=1, column=1, sticky="w")
        ttk.Button(left, text="refresh", command=self._el_refresh).grid(
            row=2, column=1, sticky="w", pady=(2, 6))
        ttk.Label(left, text="Trials").grid(row=3, column=0, sticky="e", padx=4)
        self.el_trials = ttk.Spinbox(left, from_=10, to=400, width=6); self.el_trials.set(40)
        self.el_trials.grid(row=3, column=1, sticky="w")
        ttk.Label(left, text="Error rate").grid(row=4, column=0, sticky="e", padx=4)
        self.el_rate = ttk.Spinbox(left, from_=0.1, to=0.5, increment=0.05, width=6)
        self.el_rate.set(0.30); self.el_rate.grid(row=4, column=1, sticky="w")
        ttk.Label(left, text="Threshold").grid(row=5, column=0, sticky="e", padx=4)
        self.el_thr = ttk.Spinbox(left, from_=0.05, to=0.95, increment=0.05, width=6,
                                  command=self._el_render)
        self.el_thr.set(0.50); self.el_thr.grid(row=5, column=1, sticky="w")
        self.el_thr.bind("<KeyRelease>", lambda e: self._el_render())   # re-scores every trial
        self.el_btn = ttk.Button(left, text="▶ Start test", command=self._el_toggle)
        self.el_btn.grid(row=6, column=0, sticky="w", pady=8)
        self.el_delay = ttk.Label(left, text="", foreground="#555")
        self.el_delay.grid(row=6, column=1, sticky="w", padx=8)
        self.el_status = ttk.Label(left, text="", foreground="#555")
        self.el_status.grid(row=7, column=0, columnspan=2, sticky="w")
        self.el_quality = ttk.Label(left, text="", foreground="#c0392b", wraplength=330,
                                    font=("", 9, "bold"))
        self.el_quality.grid(row=8, column=0, columnspan=2, sticky="w")
        self.el_last = ttk.Label(left, text="last trial: -", font=("", 11, "bold"))
        self.el_last.grid(row=9, column=0, columnspan=2, sticky="w", pady=(8, 2))
        self.el_bar = tk.Canvas(left, height=18, width=320, bg="#eee", highlightthickness=1,
                                highlightbackground="#ccc")
        self.el_bar.grid(row=10, column=0, columnspan=2, sticky="w")
        self.el_stats = ttk.Label(left, text="", font=("Consolas", 9), justify="left")
        self.el_stats.grid(row=11, column=0, columnspan=2, sticky="w", pady=(8, 4))
        self.el_log = tk.Listbox(left, height=10, width=50, font=("Consolas", 9),
                                 exportselection=False)
        self.el_log.grid(row=12, column=0, columnspan=2, sticky="nsew")
        cue = ttk.LabelFrame(t, text="Cue — watch here; react naturally to wrong outcomes, "
                                    "but don’t move", padding=2)
        cue.grid(row=0, column=1, sticky="nsew")
        cue.columnconfigure(0, weight=1); cue.rowconfigure(0, weight=1)
        self.el_canvas = tk.Canvas(cue, bg="#111", highlightthickness=0)
        self.el_canvas.grid(row=0, column=0, sticky="nsew")
        self.el_canvas.bind("<Configure>", lambda e: self._cue_idle(self.el_canvas))
        self._el_res = []          # (shown_label, p_error or None, dead_channels)
        self._el_models = []
        self._el_run = 0
        self._el_refresh()
        self._el_render()

    def _el_refresh(self):
        self._el_models = [e for e in models.list_models("eeg")
                           if (e["meta"].paradigm or e["meta"].model_type) == "errp"]
        names = [e["folder"].name for e in self._el_models]
        self.el_model["values"] = names
        if names:
            self.el_model.current(0)
        else:
            self.el_model.set("")
        self.el_status.config(text="" if names else
                              "No ErrP models yet — train one (Train → EEG → ErrP).")

    def _el_toggle(self):
        if self.engine.mode == "errp_live":
            self._el_stop("stopped"); return
        if self.engine.mode in ("record", "errp"):
            messagebox.showinfo("Busy", "A recording is in progress — wait for it to finish.")
            return
        sel = self.el_model.current()
        if sel < 0 or sel >= len(self._el_models):
            messagebox.showinfo("No model", "Train an ErrP model first (Train → EEG → ErrP).")
            return
        e = self._el_models[sel]
        try:
            model, meta = models.load_model(e["folder"])
        except Exception as ex:
            messagebox.showerror("Model error", str(ex)); return
        self._warn_old_filter(meta, e["folder"].name)
        cols = self._model_cols(meta, e["folder"].name)
        if cols is None:
            return
        try:
            self.engine.start(); self.engine.start_errp_live({"model": model, "cols": cols})
        except Exception as ex:
            messagebox.showerror("Source error", str(ex)); return
        self._el_run += 1
        rid = self._el_run
        self._el_res = []; self.el_quality.config(text=""); self._el_render()
        self.el_btn.config(text="■ Stop test")
        self._cue_text("get ready", color="#888", cv=self.el_canvas)
        self._errp_task_start(
            int(self.el_trials.get()), float(self.el_rate.get()), self.el_canvas,
            on_event=lambda lab, ts: self.engine.mark_live_event(lab, ts),
            # the last outcome needs ~1.8 s more EEG before it can be classified
            on_done=lambda: self.after(1500, lambda: self._el_finish(rid)),
            on_progress=lambda k, n: self.el_status.config(text=f"trial {k}/{n}",
                                                           foreground="#555"),
            lead_ms=3000)       # buffer EEG first so even trial 1 has filter context

    def _el_finish(self, rid):
        if rid == self._el_run and self.engine.mode == "errp_live":
            n = sum(1 for r in self._el_res if r[1] is not None)
            self._el_stop(f"done — {n} trials scored")

    def _el_stop(self, why):
        self._errp_abort()
        if self.engine.mode == "errp_live":
            self.engine.go_idle()
        self.el_btn.config(text="▶ Start test")
        self.el_status.config(text=why, foreground="#555")
        if not why.startswith("done"):
            self._cue_idle(self.el_canvas)

    def _el_on_result(self, label, p, dead):
        self._el_res.append((label, p, list(dead)))
        self._el_render()

    def _el_thr_value(self):
        try:
            return min(0.99, max(0.01, float(self.el_thr.get())))
        except ValueError:
            return 0.5

    def _el_render(self):
        """Score every trial at the current threshold: stats, confusion matrix, log, last bar."""
        thr = self._el_thr_value()
        scored = [(lab, p) for lab, p, _ in self._el_res if p is not None]
        tp = sum(lab == "error" and p >= thr for lab, p in scored)
        fn = sum(lab == "error" and p < thr for lab, p in scored)
        tn = sum(lab == "correct" and p < thr for lab, p in scored)
        fp = sum(lab == "correct" and p >= thr for lab, p in scored)
        n = len(scored)
        rec_e = tp / (tp + fn) if tp + fn else None
        rec_c = tn / (tn + fp) if tn + fp else None
        recalls = [r for r in (rec_e, rec_c) if r is not None]
        bal = float(np.mean(recalls)) if recalls else None
        acc = (tp + tn) / n if n else None
        skipped = len(self._el_res) - n

        def frac(a, b):
            return f"{a}/{b} ({a / b:.2f})" if b else f"{a}/{b}"
        lines = [f"scored trials  {n}" + (f"   (skipped {skipped})" if skipped else ""),
                 "accuracy       " + (f"{acc:.2f}" if acc is not None else "-"),
                 "balanced acc   " + (f"{bal:.2f}   (chance 0.50)" if bal is not None else "-"),
                 f"errors caught  {frac(tp, tp + fn)}",
                 f"false alarms   {frac(fp, fp + tn)}",
                 "",
                 "                pred CORRECT  pred ERROR",
                 f"shown CORRECT   {tn:12d}  {fp:10d}",
                 f"shown ERROR     {fn:12d}  {tp:10d}"]
        self.el_stats.config(text="\n".join(lines))

        self.el_log.delete(0, tk.END)                       # newest first
        for k in range(len(self._el_res) - 1, -1, -1):
            lab, p, dead = self._el_res[k]
            if p is None:
                self.el_log.insert(tk.END, f"#{k + 1:<3} shown {lab.upper():7s}  skipped (no data)")
                self.el_log.itemconfig(tk.END, foreground="#888")
                continue
            pred = "error" if p >= thr else "correct"
            ok = pred == lab
            note = "  dead ch!" if dead else ""
            self.el_log.insert(tk.END, f"#{k + 1:<3} shown {lab.upper():7s}  P(err) {p:.2f}  "
                                       f"-> {pred.upper():7s} {'ok' if ok else 'MISS'}{note}")
            self.el_log.itemconfig(tk.END, foreground="#1e8449" if ok else "#c0392b")

        cv = self.el_bar; cv.delete("all")
        W = cv.winfo_width() if cv.winfo_width() > 1 else 320
        last = next(((lab, p) for lab, p, _ in reversed(self._el_res) if p is not None), None)
        if last:
            lab, p = last
            pred = "error" if p >= thr else "correct"
            cv.create_rectangle(0, 0, int(W * p), 18, width=0,
                                fill="#e74c3c" if p >= thr else "#2ecc40")
            self.el_last.config(text=f"last: shown {lab.upper()}  →  P(error) {p:.2f}  "
                                     f"({'ok' if pred == lab else 'MISS'})",
                                foreground="#1e8449" if pred == lab else "#c0392b")
        else:
            self.el_last.config(text="last trial: -", foreground="#333")
        x = int(W * thr)
        cv.create_line(x, 0, x, 18, fill="#333", width=2)

    # ---------- shared ----------
    def _on_tab(self):
        if self.engine.mode in ("monitor", "live"):
            self.engine.go_idle()
            self.mon_btn.config(text="▶ Start"); self.mon_status.config(text="stopped", foreground="#a00")
            if hasattr(self, "live_btn"):
                self.live_btn.config(text="▶ Start"); self.live_status.config(text="stopped", foreground="#a00")
        # Tk also fires this event when the window first maps, so only stop the ErrP test when
        # its tab really is no longer the selected one
        if self.engine.mode == "errp_live" and self._nb.select() != str(self.t_errp):
            self._el_stop("stopped (left the tab)")

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
                elif kind == "errp_result":
                    self._el_on_result(rest[0], rest[1], rest[2])
        except queue.Empty:
            pass
        if self.engine.mode == "errp_live":             # dead electrodes make the test meaningless
            with self.engine._lock:
                q = list(self.engine.quality)
            dead = [n for n, s in zip(self.cfg.electrode_names, q) if s in ("railed", "flat")]
            dtxt, dwarn = self._delay_info(self.engine.latency_stats())
            self.el_delay.config(text=dtxt)
            if not dead and dwarn:
                self.el_quality.config(text=dwarn)
            else:
                self.el_quality.config(text=(f"No signal (railed/flat): {', '.join(dead)} — results "
                                         "are meaningless; fix contact." if dead else ""))
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
                dwarn = self._delay_info(self.engine.latency_stats())[1]
                self.mon_quality.config(text=dwarn or "", foreground="#b9770e")
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
            dtxt = self._delay_info(self.engine.latency_stats())[0]
            self.mon_sps.config(text=f"{sps:.0f} Sa/s" + (f"  \u00b7  {dtxt}" if dtxt else ""))
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
