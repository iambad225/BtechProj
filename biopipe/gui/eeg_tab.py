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
from core.eeg_models import StreamBandpass, epoch_signal, ErrPModel
from core.eeg_training import ERRP_BAND, ERRP_TMIN, ERRP_TMAX
from core.quality import channel_status, dead_channels
from core import errp_facts
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
# ErrP Live tests: the event-locked arrow task, or statements read for as long as they're shown
EL_TESTS = ("Cursor (press ←/→)", "Arrow (watch)", "True / false facts")
ERRP_TASKS = {EL_TESTS[0]: "cursor", EL_TESTS[1]: "arrow"}
REC_ERRP = {"ErrP (cursor task)": "cursor", "ErrP (arrow task)": "arrow"}
OUTCOME_COLOUR = "#e8e8e8"      # one colour for right and wrong outcomes: only the direction differs


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
    SCAN_STEP_S = 0.1                      # facts test: spacing of the sliding epochs

    def start_errp_live(self, spec):
        """spec: {"model": ErrPModel, "cols": model electrode columns}. Outcomes are marked with
        mark_live_event(); each is classified once enough post-event data has arrived and the
        result is posted as ("errp_result", label, p_error, dead_channels). For the facts test,
        begin_scan()/end_scan() bracket each statement instead (see there)."""
        with self._lock:
            self._el_buf = np.zeros((0, self.cfg.n_channels)); self._el_gend = None
            self._el_fbuf = np.zeros((0, self.cfg.n_channels))
            self._el_pending = []; self._el_scans = []; self.quality = []
        self._el_filt = StreamBandpass(self.cfg.sample_rate, *ERRP_BAND, self.cfg.n_channels)
        self._el_spec = spec; self._el_qt = 0.0
        self.mode = "errp_live"

    def mark_live_event(self, label, t=None):
        g = self.index_at(time.perf_counter() if t is None else t)
        with self._lock:
            self._el_pending.append((g, label))

    def begin_scan(self, label, t=None):
        """A statement appeared at PC time ``t``. Its onset is not one known instant (the error is
        noticed while reading), so every point from here to end_scan() is treated as a possible
        event: the -0.2..+0.8 s epoch around it is classified, every SCAN_STEP_S. Each score is
        posted as ("errp_scan_p", p); when the statement is done, ("errp_scan", label, peak, mean,
        n_epochs, dead_channels)."""
        g = self.index_at(time.perf_counter() if t is None else t)
        with self._lock:
            self._el_scans.append({"label": label, "next": g, "g_off": None, "ps": [], "dead": set()})

    def end_scan(self, t=None):
        g = self.index_at(time.perf_counter() if t is None else t)
        with self._lock:
            for s in self._el_scans:
                if s["g_off"] is None:
                    s["g_off"] = g

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
        self._do_scans(raw, fbuf, start, total, pre, post)

    def _do_scans(self, raw, fbuf, start, total, pre, post):
        """Facts test: classify each due sliding epoch of every open statement."""
        fs = self.cfg.sample_rate
        step = max(1, int(round(self.SCAN_STEP_S * fs)))
        with self._lock:
            scans = list(self._el_scans)
        done = []
        for s in scans:
            gs = []
            while total >= s["next"] + post and (s["g_off"] is None or s["next"] <= s["g_off"]):
                if s["next"] - pre >= start:                    # still inside the kept stream
                    gs.append(s["next"])
                s["next"] += step
            if gs:
                eps = epoch_signal(fbuf, fs, [g - start for g in gs], ERRP_TMIN, ERRP_TMAX,
                                   baseline=(ERRP_TMIN, 0.0))
                try:
                    ps = [float(p) for p in self._el_spec["model"].proba_error(
                        [ep[:, self._el_spec["cols"]] for ep in eps])] if eps else []
                except Exception:
                    ps = []
                s["ps"] += ps
                for p in ps:
                    self.events.put(("errp_scan_p", p))
                # same raw-sample check as the arrow test, over the span these epochs cover
                seg = raw[max(0, gs[0] - pre - start):gs[-1] + post - start]
                s["dead"].update(dead_channels(seg, self.cfg.electrode_names))
            if s["g_off"] is not None and s["next"] > s["g_off"]:
                done.append(s)
                ps = s["ps"]
                self.events.put(("errp_scan", s["label"], max(ps) if ps else None,
                                 float(np.mean(ps)) if ps else None, len(ps), sorted(s["dead"])))
        if done:
            with self._lock:
                self._el_scans = [s for s in self._el_scans if s not in done]

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
        win, step = int(float(self.v_win.get())), int(float(self.v_step.get()))
        self._check_timing(win, step)
        self.cfg.window_ms, self.cfg.step_ms = win, step
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

    @staticmethod
    def _check_timing(win_ms, step_ms):
        """The Monitor's band composition, waterfall and alpha test take a spectrum of the last
        Window ms; a few ms gives a one-sample 'spectrum' and they silently show nothing."""
        if win_ms < 250:
            raise ValueError(f"Window {win_ms} ms is too short: band composition, the waterfall "
                             "and the alpha test need at least 250 ms (1000-2000 ms recommended).")
        if step_ms < 50:
            raise ValueError(f"Step {step_ms} ms is too short: use 50 ms or more (250 ms is typical).")

    def _on_profile_load(self, cfg):
        try:
            self._check_timing(cfg.window_ms, cfg.step_ms)
        except ValueError as e:
            messagebox.showerror("Invalid profile", f"{e}\nFix it in the profile, or set it here "
                                                    "and Apply & Save."); return
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
                          values=["MI (left/right)", *REC_ERRP])
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
        wait_lbl = ttk.Label(self.errp_frame, text="Max wait s")
        wait_lbl.grid(row=2, column=0, sticky="e", padx=4)
        self.v_wait_s = ttk.Spinbox(self.errp_frame, from_=1.0, to=10.0, increment=0.5, width=5)
        self.v_wait_s.set(3.0); self.v_wait_s.grid(row=2, column=1, sticky="w")
        self._rec_wait_widgets = [wait_lbl, self.v_wait_s]           # cursor task only
        ttk.Label(self.errp_frame, text="Outcome s").grid(row=3, column=0, sticky="e", padx=4)
        self.v_outcome_s = ttk.Spinbox(self.errp_frame, from_=0.8, to=5.0, increment=0.1, width=5)
        self.v_outcome_s.set(1.0); self.v_outcome_s.grid(row=3, column=1, sticky="w")

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
        task = REC_ERRP.get(self.v_paradigm.get())
        if task:
            self.mi_frame.grid_remove(); self.errp_frame.grid()
            for w in self._rec_wait_widgets:
                w.grid() if task == "cursor" else w.grid_remove()
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
        task = REC_ERRP.get(self.v_paradigm.get())
        if task:
            self._record_errp(task)
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

    # ----- ErrP tasks (shared by Record and the ErrP Live test) -----
    def _record_errp(self, task):
        self.cfg.paradigm = "errp"; self.cfg.labels = ["correct", "error"]
        self._cue_deadline = None            # ErrP is fast/event-locked: no countdown
        self.engine.start_errp()
        self._errp_task_start(
            int(self.v_trials.get()), float(self.v_errrate.get()), self.cue_canvas,
            on_event=lambda lab, ts: self.engine.mark_event(lab, ts),
            on_done=lambda: self.engine.finish_errp(),
            on_progress=lambda k, n, note="": self.rec_status.config(
                text=f"ErrP task: {k}/{n}" + (f"  ({note})" if note else "")),
            lead_ms=3000,       # same lead-in as ErrP Live: the causal filter settles first
            task=task, **self._errp_times(self.v_wait_s, self.v_outcome_s))

    @staticmethod
    def _errp_times(wait_w, outcome_w):
        """Response window / outcome display time from two spinboxes. The outcome stays up at least
        until the classic epoch end (ERRP_TMAX), so its disappearance never falls inside it."""
        def val(w, default, lo, hi):
            try:
                return min(hi, max(lo, float(w.get())))
            except ValueError:
                return default
        return {"wait_s": val(wait_w, 3.0, 1.0, 10.0),
                "outcome_s": val(outcome_w, 1.0, ERRP_TMAX, 5.0)}

    def _errp_task_start(self, n, rate, cv, on_event, on_done, on_progress, lead_ms=500,
                         task="cursor", wait_s=3.0, outcome_s=1.0):
        """n labelled trials: fixation -> target (left or right) -> a command toward it is
        executed, the opposite way at ``rate``. Right and wrong outcomes look identical except
        for the direction, so only the error itself can drive the response.
          task="cursor": the subject presses Left/Right (interaction ErrP: their own command goes
            wrong, the stronger response). A key away from the target (a slip) or no key within
            ``wait_s`` leaves the trial unlabelled and it is repeated.
          task="arrow": the target arrow shows for 0.9 s, then the outcome arrow (observation).
        ``on_event(label, t)`` fires at the outcome: 'error' if it went against the command."""
        self._errp_token = getattr(self, "_errp_token", 0) + 1
        self._errp_t = {"n": n, "left": n, "rate": rate, "cv": cv, "target": None, "task": task,
                        "wait_s": wait_s, "outcome_s": outcome_s, "armed": False, "timer": None,
                        "slips": 0, "misses": 0, "on_event": on_event, "on_done": on_done,
                        "on_progress": on_progress}
        self._errp_bind_keys()
        on_progress(0, n)
        self.after(lead_ms, self._errp_next, self._errp_token)

    def _errp_bind_keys(self):
        if not getattr(self, "_errp_keys_bound", False):
            top = self.winfo_toplevel()
            top.bind_all("<KeyPress-Left>", lambda e: self._errp_key("left"), add="+")
            top.bind_all("<KeyPress-Right>", lambda e: self._errp_key("right"), add="+")
            self._errp_keys_bound = True

    def _errp_abort(self):
        self._errp_token = getattr(self, "_errp_token", 0) + 1     # orphans pending callbacks
        t = getattr(self, "_errp_t", None)
        if t:
            t["armed"] = False

    def _errp_progress(self, note=""):
        t = self._errp_t
        extra = [s for s in (f"{t['slips']} slips" if t["slips"] else "",
                             f"{t['misses']} too slow" if t["misses"] else "", note) if s]
        t["on_progress"](t["n"] - t["left"], t["n"], ", ".join(extra))

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
        if t["task"] == "arrow":                 # observation: the 'command' is the target itself
            self._cue_arrow(t["target"], "#4a90d9", "TARGET", cv=t["cv"])
            self.after(900, self._errp_move, token, t["target"])
            return
        self._cue_cursor(t["target"], "center", hint=True, cv=t["cv"])
        # keyboard focus on the task display every trial: otherwise the arrows go to whatever was
        # clicked last (the tab bar even switches tabs on Left/Right, which stops the test).
        # focus_set, not focus_force: never steal the keyboard from another program
        t["cv"].focus_set()
        t["armed"] = True
        t["timer"] = self.after(int(t["wait_s"] * 1000), self._errp_timeout, token)

    def _errp_timeout(self, token):
        t = self._errp_t
        if token != self._errp_token or not t["armed"]:
            return
        t["armed"] = False; t["misses"] += 1
        self._cue_text("too slow", color="#888", cv=t["cv"])
        self._errp_progress()
        self.after(800, self._errp_next, token)

    def _errp_key(self, pressed):
        t = getattr(self, "_errp_t", None)
        if not t or not t["armed"]:
            return
        t["armed"] = False
        if t["timer"] is not None:
            self.after_cancel(t["timer"]); t["timer"] = None
        self._errp_move(self._errp_token, pressed)

    def _errp_move(self, token, pressed):
        """Execute the command ``pressed`` (wrong at the error rate) and mark the outcome."""
        if token != self._errp_token:
            return
        t = self._errp_t
        is_err = random.random() < t["rate"]
        moved = ("right" if pressed == "left" else "left") if is_err else pressed
        if t["task"] == "arrow":
            self._cue_arrow(moved, OUTCOME_COLOUR, cv=t["cv"])
        else:
            self._cue_cursor(t["target"], moved, cv=t["cv"])
        t["cv"].update_idletasks()               # paint the outcome now, then timestamp its onset
        if pressed == t["target"]:
            t["on_event"]("error" if is_err else "correct", time.perf_counter())
            t["left"] -= 1
        else:                                    # own slip: the subject's error, not the machine's
            t["slips"] += 1
        self._errp_progress()
        self.after(int(t["outcome_s"] * 1000), self._errp_next, token)

    def _cue_cursor(self, target, cursor, hint=False, cv=None):
        """Left/right boxes with the target one highlighted, and the cursor at 'center', 'left' or
        'right'. Drawn the same way after a right or a wrong move, so only the direction differs."""
        cv, w, h = self._cue_dims(cv); cv.delete("all")
        cy, s = h * 0.45, h * 0.11
        xs = {"left": w * 0.22, "center": w * 0.5, "right": w * 0.78}
        for side in ("left", "right"):
            on = side == target
            cv.create_rectangle(xs[side] - s, cy - s, xs[side] + s, cy + s, width=4,
                                outline="#4a90d9" if on else "#555",
                                fill="#1d3550" if on else "")
        cv.create_line(xs["left"] + s, cy, xs["right"] - s, cy, fill="#333", width=2)
        r = h * 0.045
        cv.create_oval(xs[cursor] - r, cy - r, xs[cursor] + r, cy + r, fill="#f1c40f", width=0)
        if hint:
            cv.create_text(w / 2, h * 0.82, text="press  ←  or  →  toward the blue box",
                           fill="#777", font=("", max(10, int(h * 0.04))))

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
        left.rowconfigure(13, weight=1)
        self.el_help = ttk.Label(left, text="", foreground="#555", wraplength=330, justify="left")
        self.el_help.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        ttk.Label(left, text="Model").grid(row=1, column=0, sticky="e", padx=4)
        self.el_model = ttk.Combobox(left, state="readonly", width=38)
        self.el_model.grid(row=1, column=1, sticky="w")
        ttk.Button(left, text="refresh", command=self._el_refresh).grid(
            row=2, column=1, sticky="w", pady=(2, 6))
        ttk.Label(left, text="Test").grid(row=3, column=0, sticky="e", padx=4)
        self.v_el_test = tk.StringVar(value=EL_TESTS[0])
        self.el_test = ttk.Combobox(left, textvariable=self.v_el_test, values=EL_TESTS,
                                    state="readonly", width=22)
        self.el_test.grid(row=3, column=1, sticky="w", pady=(0, 2))
        self.el_test.bind("<<ComboboxSelected>>", lambda e: self._el_mode())
        self.el_trials_lbl = ttk.Label(left, text="Trials")
        self.el_trials_lbl.grid(row=4, column=0, sticky="e", padx=4)
        self.el_trials = ttk.Spinbox(left, from_=10, to=400, width=6); self.el_trials.set(40)
        self.el_trials.grid(row=4, column=1, sticky="w")
        self.el_rate_lbl = ttk.Label(left, text="Error rate")
        self.el_rate_lbl.grid(row=5, column=0, sticky="e", padx=4)
        self.el_rate = ttk.Spinbox(left, from_=0.1, to=0.5, increment=0.05, width=6)
        self.el_rate.set(0.30); self.el_rate.grid(row=5, column=1, sticky="w")
        # facts test only: how long each statement stays up, and how its scores are summarised
        secs_lbl = ttk.Label(left, text="Fact time (s)")
        secs_lbl.grid(row=6, column=0, sticky="e", padx=4)
        self.el_secs = ttk.Spinbox(left, from_=2, to=60, increment=1, width=6); self.el_secs.set(15)
        self.el_secs.grid(row=6, column=1, sticky="w")
        score_lbl = ttk.Label(left, text="Score")
        score_lbl.grid(row=7, column=0, sticky="e", padx=4)
        self.v_el_score = tk.StringVar(value="peak")
        score = ttk.Combobox(left, textvariable=self.v_el_score, values=("peak", "mean"),
                             state="readonly", width=6)
        score.grid(row=7, column=1, sticky="w")
        score.bind("<<ComboboxSelected>>", lambda e: self._el_render())      # re-scores every fact
        self._el_fact_widgets = [secs_lbl, self.el_secs, score_lbl, score]
        # cursor test only: how long to wait for the key, and how long the moved cursor stays up.
        # An outcome at least as long as the model's epoch keeps it on screen until its verdict
        tgt_lbl = ttk.Label(left, text="Max wait (s)")
        tgt_lbl.grid(row=6, column=0, sticky="e", padx=4)
        self.el_wait_s = ttk.Spinbox(left, from_=1.0, to=10.0, increment=0.5, width=6)
        self.el_wait_s.set(3.0); self.el_wait_s.grid(row=6, column=1, sticky="w")
        out_lbl = ttk.Label(left, text="Outcome (s)")
        out_lbl.grid(row=7, column=0, sticky="e", padx=4)
        self.el_outcome_s = ttk.Spinbox(left, from_=0.8, to=5.0, increment=0.1, width=6)
        self.el_outcome_s.set(1.0); self.el_outcome_s.grid(row=7, column=1, sticky="w")
        self._el_arrow_widgets = [tgt_lbl, self.el_wait_s, out_lbl, self.el_outcome_s]
        self._el_wait_widgets = [tgt_lbl, self.el_wait_s]                # cursor test only
        ttk.Label(left, text="Threshold").grid(row=8, column=0, sticky="e", padx=4)
        self.el_thr = ttk.Spinbox(left, from_=0.05, to=0.95, increment=0.05, width=6,
                                  command=self._el_render)
        self.el_thr.set(0.50); self.el_thr.grid(row=8, column=1, sticky="w")
        self.el_thr.bind("<KeyRelease>", lambda e: self._el_render())   # re-scores every trial
        self.el_btn = ttk.Button(left, text="▶ Start test", command=self._el_toggle)
        self.el_btn.grid(row=9, column=0, sticky="w", pady=8)
        self.el_delay = ttk.Label(left, text="", foreground="#555")
        self.el_delay.grid(row=9, column=1, sticky="w", padx=8)
        self.el_status = ttk.Label(left, text="", foreground="#555")
        self.el_status.grid(row=10, column=0, columnspan=2, sticky="w")
        self.el_quality = ttk.Label(left, text="", foreground="#c0392b", wraplength=330,
                                    font=("", 9, "bold"))
        self.el_quality.grid(row=11, column=0, columnspan=2, sticky="w")
        # the subject shouldn't see verdicts mid-test: a shown "MISS" is itself error feedback
        self.v_el_show = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, text="Show results (untick while you are the subject)",
                        variable=self.v_el_show, command=self._el_show_toggle).grid(
            row=12, column=0, columnspan=2, sticky="w", pady=(8, 0))
        res = self.el_results = ttk.Frame(left)
        res.grid(row=13, column=0, columnspan=2, sticky="nsew")
        res.columnconfigure(0, weight=1); res.rowconfigure(4, weight=1)
        self.el_last = ttk.Label(res, text="last trial: -", font=("", 11, "bold"))
        self.el_last.grid(row=0, column=0, sticky="w", pady=(6, 2))
        self.el_bar = tk.Canvas(res, height=18, width=320, bg="#eee", highlightthickness=1,
                                highlightbackground="#ccc")
        self.el_bar.grid(row=1, column=0, sticky="w")
        # facts test: P(error) of the statement on screen, one point per sliding epoch
        self.el_trace = tk.Canvas(res, height=64, width=320, bg="#fafafa", highlightthickness=1,
                                  highlightbackground="#ccc")
        self.el_trace.grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.el_stats = ttk.Label(res, text="", font=("Consolas", 9), justify="left")
        self.el_stats.grid(row=3, column=0, sticky="w", pady=(8, 4))
        self.el_log = tk.Listbox(res, height=10, width=50, font=("Consolas", 9),
                                 exportselection=False)
        self.el_log.grid(row=4, column=0, sticky="nsew")
        xs = ttk.Scrollbar(res, orient="horizontal", command=self.el_log.xview)
        xs.grid(row=5, column=0, sticky="ew")                       # statement text runs long
        self.el_log.config(xscrollcommand=xs.set)
        self.el_cue =ttk.LabelFrame(t, text="", padding=2)
        self.el_cue.grid(row=0, column=1, sticky="nsew")
        self.el_cue.columnconfigure(0, weight=1); self.el_cue.rowconfigure(0, weight=1)
        self.el_canvas = tk.Canvas(self.el_cue, bg="#111", highlightthickness=0)
        self.el_canvas.grid(row=0, column=0, sticky="nsew")
        self.el_canvas.bind("<Configure>", lambda e: self._cue_idle(self.el_canvas))
        self._el_res = []          # one dict per trial / statement (see _el_on_result, _el_on_scan)
        self._el_kind = "arrow"    # test the results in _el_res came from
        self._el_fact_q = deque()  # texts of statements whose scan result is still to come
        self._el_trace_ps = []
        self._el_models = []
        self._el_run = 0
        self._el_refresh()
        self._el_mode()

    def _el_facts(self):
        return self.v_el_test.get() == EL_TESTS[2]

    def _el_mode(self):
        facts = self._el_facts()
        for w in self._el_fact_widgets + [self.el_trace]:
            w.grid() if facts else w.grid_remove()
        task = ERRP_TASKS.get(self.v_el_test.get())
        for w in self._el_arrow_widgets:
            w.grid_remove() if facts else w.grid()
        for w in self._el_wait_widgets:
            w.grid() if task == "cursor" else w.grid_remove()
        self.el_trials_lbl.config(text="Facts" if facts else "Trials")
        self.el_rate_lbl.config(text="False share" if facts else "Error rate")
        if facts:
            self.el_help.config(text="Easy statements appear on the right, one at a time; about "
                                     "'False share' of them are false. Read each silently. While "
                                     "a statement is shown, the model scores the live EEG every "
                                     "0.1 s; a statement counts as ERROR when its peak (or mean) "
                                     "P(error) reaches the threshold.")
            self.el_cue.config(text="Statements — read silently; don’t answer aloud or move")
        elif task == "cursor":
            self.el_help.config(text="Test a trained ErrP model on live EEG. Press ← or → "
                                     "toward the blue box; the cursor sometimes moves the wrong way. "
                                     "Each move is classified from the live stream and scored "
                                     "against what actually happened.")
            self.el_cue.config(text="Task — press ← / → toward the blue box; keep the "
                                    "rest of your body still")
        else:
            self.el_help.config(text="Test a trained ErrP model on live EEG. A target arrow shows, "
                                     "then the outcome arrow, which sometimes points the other way. "
                                     "Each outcome is classified from the live stream and scored "
                                     "against what was actually shown.")
            self.el_cue.config(text="Cue — watch here; react naturally to wrong outcomes, "
                                    "but don’t move")
        self._el_render()

    def _el_show_toggle(self):
        if self.v_el_show.get():
            self.el_results.grid(); self._el_render(); self._el_trace_draw()
        else:
            self.el_results.grid_remove()

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
        if getattr(model, "method", "") != ErrPModel.method:
            messagebox.showinfo("Retrain", f"{e['folder'].name} uses the old ErrP pipeline, which "
                                "can't see the error response on this headset. Retrain it "
                                "(Train → EEG → ErrP)."); return
        cols = self._model_cols(meta, e["folder"].name)
        if cols is None:
            return
        try:
            self.engine.start(); self.engine.start_errp_live({"model": model, "cols": cols})
        except Exception as ex:
            messagebox.showerror("Source error", str(ex)); return
        self._el_run += 1
        rid = self._el_run
        facts = self._el_facts()
        self._el_kind = "facts" if facts else "arrow"
        self._el_res = []; self._el_fact_q.clear(); self._el_trace_reset()
        self.el_quality.config(text=""); self._el_render()
        self.el_btn.config(text="■ Stop test"); self.el_test.config(state="disabled")
        self._cue_text("get ready", color="#888", cv=self.el_canvas)
        if facts:
            try:
                secs = min(60.0, max(2.0, float(self.el_secs.get())))
            except ValueError:
                secs = 15.0
            self._facts_task_start(
                int(self.el_trials.get()), float(self.el_rate.get()), secs, self.el_canvas,
                on_start=self._el_fact_start,
                on_end=lambda ts: self.engine.end_scan(ts),
                # the last statement's final epochs need ~0.8 s more EEG before they're scored
                on_done=lambda: self.after(1500, lambda: self._el_finish(rid)),
                on_progress=lambda k, n: self.el_status.config(text=f"fact {k}/{n}",
                                                               foreground="#555"),
                lead_ms=3000)
            return
        self._errp_task_start(
            int(self.el_trials.get()), float(self.el_rate.get()), self.el_canvas,
            on_event=lambda lab, ts: self.engine.mark_live_event(lab, ts),
            # the last outcome needs ~0.8 s more EEG before it can be classified
            on_done=lambda: self.after(1500, lambda: self._el_finish(rid)),
            on_progress=lambda k, n, note="": self.el_status.config(
                text=f"trial {k}/{n}" + (f"  ({note})" if note else ""), foreground="#555"),
            lead_ms=3000,       # buffer EEG first so even trial 1 has filter context
            task=ERRP_TASKS[self.v_el_test.get()],
            **self._errp_times(self.el_wait_s, self.el_outcome_s))

    def _el_fact_start(self, label, text, ts):
        self._el_fact_q.append(text)
        self._el_trace_reset()
        self.engine.begin_scan(label, ts)

    def _el_finish(self, rid):
        if rid == self._el_run and self.engine.mode == "errp_live":
            n = sum(1 for r in self._el_res if self._el_score(r) is not None)
            self._el_stop(f"done — {n} {'statements' if self._el_kind == 'facts' else 'trials'} scored")

    def _el_stop(self, why):
        self._errp_abort()
        if self.engine.mode == "errp_live":
            self.engine.go_idle()
        self.el_btn.config(text="▶ Start test"); self.el_test.config(state="readonly")
        self.el_status.config(text=why, foreground="#555")
        if not why.startswith("done"):
            self._cue_idle(self.el_canvas)

    def _el_on_result(self, label, p, dead):
        self._el_res.append({"label": label, "p": p, "dead": list(dead)})
        self._el_render()

    def _el_on_scan(self, label, peak, mean, n, dead):
        text = self._el_fact_q.popleft() if self._el_fact_q else ""
        self._el_res.append({"label": label, "peak": peak, "mean": mean, "n": n,
                             "dead": list(dead), "text": text})
        self._el_render()

    def _el_score(self, r):
        """The number a trial is thresholded on: P(error) of the outcome epoch (arrow test), or
        the peak / mean of a statement's sliding-epoch scores (facts test)."""
        return r[self.v_el_score.get()] if "peak" in r else r["p"]

    def _el_trace_reset(self):
        self._el_trace_ps = []
        self._el_trace_draw()

    def _el_trace_add(self, p):
        self._el_trace_ps.append(p)
        if self.v_el_show.get():
            self._el_trace_draw()

    def _el_trace_draw(self):
        cv = self.el_trace; cv.delete("all")
        W = cv.winfo_width() if cv.winfo_width() > 1 else 320
        H = cv.winfo_height() if cv.winfo_height() > 1 else 64
        try:
            secs = min(60.0, max(2.0, float(self.el_secs.get())))
        except ValueError:
            secs = 15.0
        n_max = max(2, int(secs / EEGEngine.SCAN_STEP_S) + 1)
        y_thr = H - 2 - (H - 4) * self._el_thr_value()
        cv.create_line(0, y_thr, W, y_thr, fill="#999", dash=(3, 2))
        cv.create_text(4, 2, text="P(error) now", anchor="nw", fill="#888", font=("", 8))
        ps = self._el_trace_ps[-n_max:]
        if len(ps) > 1:
            pts = []
            for i, p in enumerate(ps):
                pts += [W * i / (n_max - 1), H - 2 - (H - 4) * p]
            cv.create_line(*pts, fill="#c0392b", width=2)

    def _el_thr_value(self):
        try:
            return min(0.99, max(0.01, float(self.el_thr.get())))
        except ValueError:
            return 0.5

    def _el_render(self):
        """Score every trial at the current threshold: stats, confusion matrix, log, last bar."""
        thr = self._el_thr_value()
        facts = self._el_kind == "facts"
        shown = {"correct": "TRUE", "error": "FALSE"} if facts else {"correct": "CORRECT", "error": "ERROR"}
        scored = [(r["label"], self._el_score(r)) for r in self._el_res
                  if self._el_score(r) is not None]
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
        rows = [("scored " + ("statements" if facts else "trials"),
                 f"{n}" + (f"   (skipped {skipped})" if skipped else "")),
                ("accuracy", f"{acc:.2f}" if acc is not None else "-"),
                ("balanced acc", f"{bal:.2f}   (chance 0.50)" if bal is not None else "-"),
                ("false caught" if facts else "errors caught", frac(tp, tp + fn)),
                ("true flagged" if facts else "false alarms", frac(fp, fp + tn))]
        lines = [f"{k:<18s}{v}" for k, v in rows] + [
                 "",
                 "                 pred CORRECT  pred ERROR",
                 f"shown {shown['correct']:10s} {tn:12d}  {fp:10d}",
                 f"shown {shown['error']:10s} {fn:12d}  {tp:10d}"]
        self.el_stats.config(text="\n".join(lines))

        self.el_log.delete(0, tk.END)                       # newest first
        for k in range(len(self._el_res) - 1, -1, -1):
            r = self._el_res[k]
            lab, p = r["label"], self._el_score(r)
            if p is None:
                self.el_log.insert(tk.END, f"#{k + 1:<3} shown {shown[lab]:7s}  skipped (no data)")
                self.el_log.itemconfig(tk.END, foreground="#888")
                continue
            pred = "error" if p >= thr else "correct"
            ok = pred == lab
            note = "  dead ch!" if r["dead"] else ""
            if facts:
                line = (f"#{k + 1:<3} {shown[lab]:5s} peak {r['peak']:.2f} mean {r['mean']:.2f} "
                        f"-> {pred.upper():7s} {'ok' if ok else 'MISS'}{note}  {r['text']}")
            else:
                line = (f"#{k + 1:<3} shown {lab.upper():7s}  P(err) {p:.2f}  "
                        f"-> {pred.upper():7s} {'ok' if ok else 'MISS'}{note}")
            self.el_log.insert(tk.END, line)
            self.el_log.itemconfig(tk.END, foreground="#1e8449" if ok else "#c0392b")

        cv = self.el_bar; cv.delete("all")
        W = cv.winfo_width() if cv.winfo_width() > 1 else 320
        last = next(((k + 1, r["label"], self._el_score(r))
                     for k, r in reversed(list(enumerate(self._el_res)))
                     if self._el_score(r) is not None), None)
        if last:
            k, lab, p = last
            pred = "error" if p >= thr else "correct"
            cv.create_rectangle(0, 0, int(W * p), 18, width=0,
                                fill="#e74c3c" if p >= thr else "#2ecc40")
            score = f"{self.v_el_score.get()} P(error)" if facts else "P(error)"
            self.el_last.config(text=f"last: #{k} {shown[lab]}  →  {score} {p:.2f}  "
                                     f"({'ok' if pred == lab else 'MISS'})",
                                foreground="#1e8449" if pred == lab else "#c0392b")
        else:
            self.el_last.config(text="last trial: -", foreground="#333")
        x = int(W * thr)
        cv.create_line(x, 0, x, 18, fill="#333", width=2)
        if facts:
            self._el_trace_draw()                            # threshold line moved

    # ----- facts test: easy true / false statements, each shown for a set time -----
    def _facts_task_start(self, n, rate, secs, cv, on_start, on_end, on_done, on_progress,
                          lead_ms=3000):
        """Show n statements (``rate`` of them false) for ``secs`` each, with a fixation gap.
        ``on_start(label, text, t)`` fires once a statement is painted, ``on_end(t)`` once it's
        gone; the model scores the EEG in between. Shares the arrow task's abort token."""
        self._errp_token = getattr(self, "_errp_token", 0) + 1
        items = errp_facts.fact_order(n, rate)
        self._facts_t = {"items": items, "k": 0, "secs": secs, "cv": cv, "on_start": on_start,
                         "on_end": on_end, "on_done": on_done, "on_progress": on_progress}
        on_progress(0, len(items))
        self.after(lead_ms, self._facts_show, self._errp_token)

    def _facts_show(self, token):
        if token != self._errp_token:
            return
        t = self._facts_t
        text, is_false = t["items"][t["k"]]
        self._cue_statement(text, cv=t["cv"])
        t["cv"].update_idletasks()               # paint the statement now, then timestamp it
        t["on_start"]("error" if is_false else "correct", text, time.perf_counter())
        t["k"] += 1
        t["on_progress"](t["k"], len(t["items"]))
        self.after(int(t["secs"] * 1000), self._facts_hide, token)

    def _facts_hide(self, token):
        if token != self._errp_token:
            return
        t = self._facts_t
        self._cue_fixation(cv=t["cv"])
        t["cv"].update_idletasks()
        t["on_end"](time.perf_counter())
        if t["k"] >= len(t["items"]):
            self.after(1000, self._facts_done, token)
        else:                                    # rest gap; long enough to finish scoring (0.8 s)
            self.after(2000 + random.randint(0, 500), self._facts_show, token)

    def _facts_done(self, token):
        if token != self._errp_token:
            return
        self._cue_text("done", color="#888", cv=self._facts_t["cv"])
        self._facts_t["on_done"]()

    def _cue_statement(self, text, cv=None):
        # same neutral colour for true and false statements: only the meaning differs
        cv, w, h = self._cue_dims(cv); cv.delete("all")
        cv.create_text(w / 2, h / 2, text=text, fill="#eee", width=int(w * 0.85),
                       justify="center", font=("", max(18, int(h * 0.075)), "bold"))

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
                elif kind == "errp_scan_p":
                    self._el_trace_add(rest[0])
                elif kind == "errp_scan":
                    self._el_on_scan(*rest)
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
