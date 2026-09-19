"""EMG modality tab: Config, Monitor, Record, Live (embeds in the main window)."""
from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

from core.config import Config, default_config, Electrode, SourceConfig, FilterConfig
from core.dsp import RealTimeFilter
from core.pipeline import InferencePipeline
from sources import create_source
from sources.synthetic import SyntheticSource
import storage.recordings as recordings
import storage.models as models
from gui.widgets.waveform import Waveform
from gui.widgets.bars import LevelBars

CONFIG_PATH = "config/emg.yaml"


class EMGEngine:
    """One source + worker loop, switched between monitor / record / live."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.source = None
        self.mode = "idle"                 # idle | monitor | record | live
        self.events: "queue.Queue" = queue.Queue()
        self._lock = threading.Lock()
        self._thread = None
        self._running = False
        self._filt = None
        # monitor
        self._buf = np.zeros((0, cfg.n_channels))
        self._span = int(cfg.sample_rate * 4)
        self.rms = np.zeros(cfg.n_channels)
        self.sps = 0.0
        self._n = 0
        self._t = 0.0
        # record
        self._schedule = []
        self._seg = None
        self._seg_end = 0.0
        self._rec_sig = []
        self._rec_lab = []
        # live
        self.pipe = None
        self.latest = None

    def start(self):
        if self._running:
            return
        self.source = create_source(self.cfg)
        self.source.start()
        self._filt = self._new_filter()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _new_filter(self):
        return RealTimeFilter(self.cfg.sample_rate, self.cfg.n_channels,
                              notch_hz=self.cfg.filter.notch_hz,
                              bandpass=(self.cfg.filter.bandpass_low, self.cfg.filter.bandpass_high),
                              enabled=self.cfg.filter.enabled)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self.source:
            self.source.stop()
            self.source = None

    @property
    def is_synthetic(self):
        return isinstance(self.source, SyntheticSource)

    def go_idle(self):
        self.mode = "idle"
        if self.is_synthetic:
            self.source.set_active([])

    def start_monitor(self):
        self._filt.reset(); self._n = 0; self._t = time.perf_counter()
        with self._lock:
            self._buf = np.zeros((0, self.cfg.n_channels))
        self.mode = "monitor"

    def start_live(self, model):
        self.pipe = InferencePipeline(self.cfg, model)
        self.pipe.reset()
        self.mode = "live"

    def start_record(self, schedule):
        self._schedule = list(schedule)
        self._seg = None
        self._rec_sig = []
        self._rec_lab = []
        self._filt.reset()
        self.mode = "record"

    def _loop(self):
        while self._running:
            new = self.source.get_data() if self.source else np.zeros((0, self.cfg.n_channels))
            mode = self.mode
            if mode == "monitor":
                self._do_monitor(new)
            elif mode == "live" and self.pipe is not None:
                dec = self.pipe.process(new)
                with self._lock:
                    self.latest = dec
            elif mode == "record":
                self._do_record(new)
            time.sleep(0.005)

    def _do_monitor(self, new):
        if new.shape[0]:
            y = self._filt.process(new)
            self._n += new.shape[0]
            with self._lock:
                self._buf = np.vstack([self._buf, y])[-self._span:]
        dt = time.perf_counter() - self._t
        if dt >= 0.1:
            with self._lock:
                buf = self._buf
                if buf.shape[0] >= 2:
                    self.rms = np.sqrt(np.mean((buf - buf.mean(0)) ** 2, axis=0))
                self.sps = self._n / dt
            self._n = 0
            self._t = time.perf_counter()

    def _do_record(self, new):
        now = time.perf_counter()
        if self._seg is None:
            if not self._schedule:
                self._finish_record()
                return
            self._seg = self._schedule.pop(0)          # (phase, label, seconds)
            self._seg_end = now + self._seg[2]
            phase, label, _ = self._seg
            if self.is_synthetic:
                idx = [self.cfg.labels.index(label)] if (phase == "record" and label in self.cfg.labels) else []
                self.source.set_active(idx)
            self.events.put(("status", f"{'Recording' if phase == 'record' else 'Get ready'}: {label}"))
        if self._seg[0] == "record" and new.shape[0]:
            self._rec_sig.append(new.copy())
            self._rec_lab += [self._seg[1]] * new.shape[0]
        if now >= self._seg_end:
            self._seg = None

    def _finish_record(self):
        sig = np.vstack(self._rec_sig) if self._rec_sig else np.zeros((0, self.cfg.n_channels))
        if self.is_synthetic:
            self.source.set_active([])
        self.mode = "idle"
        self.events.put(("recorded", sig, list(self._rec_lab)))


class EMGTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=8)
        self.cfg = self._load_cfg()
        self.engine = EMGEngine(self.cfg)
        self.grid(sticky="nsew")
        master.columnconfigure(0, weight=1); master.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1); self.rowconfigure(0, weight=1)

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

    # ---------- config ----------
    def _load_cfg(self) -> Config:
        try:
            c = Config.load(CONFIG_PATH)
            if c.modality != "emg":
                raise ValueError
            return c
        except Exception:
            c = default_config("emg")
            c.save(CONFIG_PATH)
            return c

    def _build_config(self):
        t = self.t_cfg
        r = 0
        ttk.Label(t, text="Subject").grid(row=r, column=0, sticky="e", padx=4, pady=2)
        self.v_subject = tk.StringVar(value=self.cfg.subject)
        ttk.Entry(t, textvariable=self.v_subject, width=18).grid(row=r, column=1, sticky="w"); r += 1

        ttk.Label(t, text="Source").grid(row=r, column=0, sticky="e", padx=4, pady=2)
        self.v_src = tk.StringVar(value=self.cfg.source.type)
        ttk.Combobox(t, textvariable=self.v_src, values=["cyton", "lsl", "synthetic"],
                     state="readonly", width=15).grid(row=r, column=1, sticky="w"); r += 1
        ttk.Label(t, text="Cyton port").grid(row=r, column=0, sticky="e", padx=4, pady=2)
        self.v_port = tk.StringVar(value=self.cfg.source.serial_port)
        ttk.Entry(t, textvariable=self.v_port, width=15).grid(row=r, column=1, sticky="w"); r += 1
        ttk.Label(t, text="Sample rate").grid(row=r, column=0, sticky="e", padx=4, pady=2)
        self.v_sr = tk.StringVar(value=str(self.cfg.sample_rate))
        ttk.Entry(t, textvariable=self.v_sr, width=8).grid(row=r, column=1, sticky="w"); r += 1

        ttk.Label(t, text="Filter notch/low/high").grid(row=r, column=0, sticky="e", padx=4, pady=2)
        fr = ttk.Frame(t); fr.grid(row=r, column=1, sticky="w")
        self.v_notch = tk.StringVar(value=str(self.cfg.filter.notch_hz))
        self.v_lo = tk.StringVar(value=str(self.cfg.filter.bandpass_low))
        self.v_hi = tk.StringVar(value=str(self.cfg.filter.bandpass_high))
        for i, v in enumerate((self.v_notch, self.v_lo, self.v_hi)):
            ttk.Entry(fr, textvariable=v, width=6).grid(row=0, column=i, padx=2)
        r += 1
        ttk.Label(t, text="Window/Step (ms)").grid(row=r, column=0, sticky="e", padx=4, pady=2)
        wr = ttk.Frame(t); wr.grid(row=r, column=1, sticky="w")
        self.v_win = tk.StringVar(value=str(self.cfg.window_ms))
        self.v_step = tk.StringVar(value=str(self.cfg.step_ms))
        ttk.Entry(wr, textvariable=self.v_win, width=6).grid(row=0, column=0, padx=2)
        ttk.Entry(wr, textvariable=self.v_step, width=6).grid(row=0, column=1, padx=2)
        r += 1

        ttk.Separator(t, orient="horizontal").grid(row=r, column=0, columnspan=3, sticky="ew", pady=8); r += 1
        ttk.Label(t, text="Electrodes (channel + muscle)", font=("", 9, "bold")).grid(
            row=r, column=0, columnspan=3, sticky="w"); r += 1
        self.elec_frame = ttk.Frame(t); self.elec_frame.grid(row=r, column=0, columnspan=3, sticky="w"); r += 1
        self._elec_rows = []
        self._render_electrodes()
        ttk.Button(t, text="+ channel", command=self._add_electrode).grid(row=r, column=0, sticky="w", pady=2); r += 1

        ttk.Separator(t, orient="horizontal").grid(row=r, column=0, columnspan=3, sticky="ew", pady=8); r += 1
        ttk.Label(t, text="Actions (classes; 'rest' is implicit)", font=("", 9, "bold")).grid(
            row=r, column=0, columnspan=3, sticky="w"); r += 1
        self.actions_list = tk.Listbox(t, height=5, width=24, exportselection=False)
        self.actions_list.grid(row=r, column=0, sticky="w")
        for a in self.cfg.labels:
            self.actions_list.insert(tk.END, a)
        ab = ttk.Frame(t); ab.grid(row=r, column=1, sticky="nw")
        self.v_newaction = ttk.Entry(ab, width=16); self.v_newaction.grid(row=0, column=0)
        ttk.Button(ab, text="Add", command=self._add_action).grid(row=1, column=0, sticky="w", pady=2)
        ttk.Button(ab, text="Remove", command=self._remove_action).grid(row=2, column=0, sticky="w")
        r += 1

        ttk.Button(t, text="Apply & Save", command=self._apply).grid(row=r, column=0, sticky="w", pady=(10, 0))
        self.cfg_status = ttk.Label(t, text="", foreground="#0a7")
        self.cfg_status.grid(row=r, column=1, columnspan=2, sticky="w", pady=(10, 0))

    def _render_electrodes(self):
        for w in self.elec_frame.winfo_children():
            w.destroy()
        self._elec_rows = []
        ttk.Label(self.elec_frame, text="ch", width=4).grid(row=0, column=0)
        ttk.Label(self.elec_frame, text="muscle / electrode label").grid(row=0, column=1, sticky="w")
        for i, e in enumerate(self.cfg.data_electrodes(), start=1):
            ch = tk.StringVar(value=str(e.channel))
            nm = tk.StringVar(value=e.name)
            ttk.Entry(self.elec_frame, textvariable=ch, width=4).grid(row=i, column=0, padx=2, pady=1)
            ttk.Entry(self.elec_frame, textvariable=nm, width=28).grid(row=i, column=1, padx=2, pady=1, sticky="w")
            btn = ttk.Button(self.elec_frame, text="x", width=2,
                             command=lambda idx=i - 1: self._remove_electrode(idx))
            btn.grid(row=i, column=2, padx=2)
            self._elec_rows.append((ch, nm))

    def _read_electrodes(self):
        out = []
        for ch, nm in self._elec_rows:
            if ch.get().strip().isdigit() and nm.get().strip():
                out.append(Electrode(channel=int(ch.get()), name=nm.get().strip(), role="data"))
        return out

    def _add_electrode(self):
        self.cfg.electrodes = self._read_electrodes()
        nxt = max([e.channel for e in self.cfg.electrodes], default=-1) + 1
        self.cfg.electrodes.append(Electrode(channel=nxt, name=f"channel {nxt}", role="data"))
        self._render_electrodes()

    def _remove_electrode(self, idx):
        elecs = self._read_electrodes()
        if 0 <= idx < len(elecs):
            elecs.pop(idx)
        self.cfg.electrodes = elecs
        self._render_electrodes()

    def _add_action(self):
        a = self.v_newaction.get().strip().replace(" ", "_")
        if a and a not in self.actions_list.get(0, tk.END):
            self.actions_list.insert(tk.END, a)
            self.v_newaction.delete(0, tk.END)

    def _remove_action(self):
        sel = self.actions_list.curselection()
        if sel:
            self.actions_list.delete(sel[0])

    def _apply(self):
        try:
            self.cfg.subject = self.v_subject.get().strip() or "subject"
            self.cfg.source.type = self.v_src.get()
            self.cfg.source.serial_port = self.v_port.get().strip()
            self.cfg.sample_rate = int(float(self.v_sr.get()))
            self.cfg.filter.notch_hz = float(self.v_notch.get())
            self.cfg.filter.bandpass_low = float(self.v_lo.get())
            self.cfg.filter.bandpass_high = float(self.v_hi.get())
            self.cfg.window_ms = int(float(self.v_win.get()))
            self.cfg.step_ms = int(float(self.v_step.get()))
            self.cfg.electrodes = self._read_electrodes()
            self.cfg.labels = list(self.actions_list.get(0, tk.END))
            if not self.cfg.electrodes:
                raise ValueError("Add at least one electrode (channel + label).")
            if not self.cfg.labels:
                raise ValueError("Add at least one action.")
        except Exception as e:
            messagebox.showerror("Invalid config", str(e)); return
        self.cfg.save(CONFIG_PATH)
        self.engine.stop()
        self.engine = EMGEngine(self.cfg)
        self._build_monitor_bars()
        self._build_live_bars()
        self._refresh_record_plan()
        self.cfg_status.config(text=f"Saved -> {CONFIG_PATH} ({self.cfg.n_channels} ch, {self.cfg.n_labels} actions)")

    # ---------- monitor ----------
    def _build_monitor(self):
        t = self.t_mon
        t.columnconfigure(0, weight=1); t.rowconfigure(1, weight=1)
        bar = ttk.Frame(t); bar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.mon_btn = ttk.Button(bar, text="▶ Start", command=self._toggle_monitor)
        self.mon_btn.grid(row=0, column=0)
        self.mon_status = ttk.Label(bar, text="stopped", foreground="#a00"); self.mon_status.grid(row=0, column=1, padx=10)
        self.mon_sps = ttk.Label(bar, text="", foreground="#555"); self.mon_sps.grid(row=0, column=2, padx=10)
        main = ttk.Frame(t); main.grid(row=1, column=0, sticky="nsew")
        main.columnconfigure(1, weight=1); main.rowconfigure(0, weight=1)
        self.mon_left = ttk.LabelFrame(main, text="Per-channel RMS", padding=6)
        self.mon_left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.mon_left.columnconfigure(0, weight=1); self.mon_left.rowconfigure(0, weight=1)
        self.mon_bars = None
        self._build_monitor_bars()
        right = ttk.LabelFrame(main, text="EMG traces", padding=4)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1); right.rowconfigure(0, weight=1)
        self.mon_wave = Waveform(right); self.mon_wave.grid(row=0, column=0, sticky="nsew")

    def _build_monitor_bars(self):
        if self.mon_bars is not None:
            self.mon_bars.destroy()
        self._mon_scale = [20.0] * self.cfg.n_channels
        self.mon_bars = LevelBars(self.mon_left, self.cfg.electrode_names)
        self.mon_bars.grid(row=0, column=0, sticky="nsew")

    def _toggle_monitor(self):
        if self.engine.mode == "monitor":
            self.engine.go_idle(); self.mon_btn.config(text="▶ Start")
            self.mon_status.config(text="stopped", foreground="#a00"); return
        try:
            self.engine.start(); self._build_monitor_bars(); self.engine.start_monitor()
        except Exception as e:
            messagebox.showerror("Source error", str(e)); return
        self.mon_btn.config(text="■ Stop"); self.mon_status.config(text="streaming", foreground="#0a0")

    # ---------- record ----------
    def _build_record(self):
        t = self.t_rec
        ttk.Label(t, text="Guided capture: records 'rest' + each action, cycling for the "
                          "chosen reps, into one session.", foreground="#555").grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, 8))
        ttk.Label(t, text="Seconds per class").grid(row=1, column=0, sticky="e", padx=4)
        self.v_secs = ttk.Spinbox(t, from_=2, to=15, width=5); self.v_secs.set(4)
        self.v_secs.grid(row=1, column=1, sticky="w")
        ttk.Label(t, text="Reps").grid(row=2, column=0, sticky="e", padx=4)
        self.v_reps = ttk.Spinbox(t, from_=1, to=10, width=5); self.v_reps.set(3)
        self.v_reps.grid(row=2, column=1, sticky="w")
        ttk.Label(t, text="Description (optional)").grid(row=3, column=0, sticky="e", padx=4)
        self.v_desc = tk.StringVar()
        ttk.Entry(t, textvariable=self.v_desc, width=40).grid(row=3, column=1, columnspan=3, sticky="w")
        self.rec_btn = ttk.Button(t, text="● Record session", command=self._record)
        self.rec_btn.grid(row=4, column=0, sticky="w", pady=8)
        self.rec_status = ttk.Label(t, text="", foreground="#0a7")
        self.rec_status.grid(row=5, column=0, columnspan=4, sticky="w")
        self.rec_plan = ttk.Label(t, text="", foreground="#555")
        self.rec_plan.grid(row=6, column=0, columnspan=4, sticky="w")
        self._refresh_record_plan()

    def _refresh_record_plan(self):
        if hasattr(self, "rec_plan"):
            self.rec_plan.config(text="Plan: " + " -> ".join(["rest"] + self.cfg.labels))

    def _record(self):
        self.rec_btn.config(state="disabled")
        try:
            self.engine.start()
        except Exception as e:
            self.rec_btn.config(state="normal"); messagebox.showerror("Source error", str(e)); return
        secs = float(self.v_secs.get()); reps = int(self.v_reps.get())
        schedule = []
        for _ in range(reps):
            for label in ["rest"] + self.cfg.labels:
                schedule.append(("ready", label, 2.0))
                schedule.append(("record", label, secs))
        self.engine.start_record(schedule)
        self.rec_status.config(text="Recording...")

    # ---------- live ----------
    def _build_live(self):
        t = self.t_live
        t.columnconfigure(0, weight=1); t.rowconfigure(2, weight=1)
        bar = ttk.Frame(t); bar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(bar, text="Model").grid(row=0, column=0)
        self.v_model = tk.StringVar()
        self.model_combo = ttk.Combobox(bar, textvariable=self.v_model, state="readonly", width=42)
        self.model_combo.grid(row=0, column=1, padx=6)
        ttk.Button(bar, text="refresh", command=self._refresh_models).grid(row=0, column=2)
        self.live_btn = ttk.Button(bar, text="▶ Start", command=self._toggle_live)
        self.live_btn.grid(row=0, column=3, padx=6)
        self.live_status = ttk.Label(bar, text="stopped", foreground="#a00"); self.live_status.grid(row=0, column=4)
        self.latency_lbl = ttk.Label(bar, text="", foreground="#555"); self.latency_lbl.grid(row=0, column=5, padx=8)
        self.active_lbl = ttk.Label(t, text="ACTIVE: -", font=("", 15, "bold"))
        self.active_lbl.grid(row=1, column=0, sticky="w", pady=(0, 8))
        self.live_bars_frame = ttk.Frame(t); self.live_bars_frame.grid(row=2, column=0, sticky="nsew")
        self.live_bars_frame.columnconfigure(0, weight=1); self.live_bars_frame.rowconfigure(0, weight=1)
        self.live_bars = None
        self._build_live_bars()
        self._models = []
        self._refresh_models()

    def _build_live_bars(self):
        if self.live_bars is not None:
            self.live_bars.destroy()
        self.live_bars = LevelBars(self.live_bars_frame, self.cfg.labels)
        self.live_bars.grid(row=0, column=0, sticky="nsew")

    def _refresh_models(self):
        self._models = models.list_models("emg")
        self.model_combo["values"] = [e["folder"].name for e in self._models]
        if self._models:
            self.model_combo.current(0)

    def _toggle_live(self):
        if self.engine.mode == "live":
            self.engine.go_idle(); self.live_btn.config(text="▶ Start")
            self.live_status.config(text="stopped", foreground="#a00"); return
        sel = self.model_combo.current()
        if sel < 0 or sel >= len(self._models):
            messagebox.showinfo("No model", "Train a model first (Train tab)."); return
        try:
            model, meta = models.load_model(self._models[sel]["folder"])
            if list(meta.labels) != list(self.cfg.labels):
                self.cfg.labels = list(meta.labels)
                self._build_live_bars()
            self.engine.start()
            self.engine.start_live(model)
        except Exception as e:
            messagebox.showerror("Error", str(e)); return
        self.live_btn.config(text="■ Stop"); self.live_status.config(text="running", foreground="#0a0")

    # ---------- shared ----------
    def _on_tab(self):
        if self.engine.mode in ("monitor", "live"):
            self.engine.go_idle()
            self.mon_btn.config(text="▶ Start"); self.mon_status.config(text="stopped", foreground="#a00")
            self.live_btn.config(text="▶ Start"); self.live_status.config(text="stopped", foreground="#a00")

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
                buf = self.engine._buf.copy(); rms = np.array(self.engine.rms); sps = self.engine.sps
            self.mon_wave.draw(buf)
            for i in range(min(len(self.mon_bars.rows), len(rms))):
                self._mon_scale[i] = max(self._mon_scale[i] * 0.95, rms[i] * 1.2, 1e-3)
                self.mon_bars.set(i, rms[i] / self._mon_scale[i], f"{rms[i]:.1f}")
            self.mon_sps.config(text=f"{sps:.0f} Sa/s")
        elif self.engine.mode == "live" and self.engine.latest is not None:
            with self.engine._lock:
                dec = self.engine.latest
            if dec is not None and len(self.live_bars.rows) == len(dec.actions):
                for i in range(len(dec.actions)):
                    self.live_bars.set(i, dec.proba[i], f"{dec.proba[i]*100:0.0f}%", on=bool(dec.active[i]))
                names = dec.active_names()
                self.active_lbl.config(text="ACTIVE: " + (" + ".join(names) if names else "-"))
                self.latency_lbl.config(text=f"{dec.latency_ms:0.1f} ms")
        self.after(60, self._tick)

    def _on_recorded(self, sig, labels):
        self.rec_btn.config(state="normal")
        if sig.shape[0] < self.cfg.window_samples:
            self.rec_status.config(text="Too little data captured — not saved."); return
        folder = recordings.save_recording(self.cfg, sig, labels,
                                           description=self.v_desc.get().strip(), training=True)
        self.rec_status.config(text=f"Saved: {folder.name}  ({sig.shape[0]} samples)")

    def stop(self):
        try:
            self.engine.stop()
        except Exception:
            pass
