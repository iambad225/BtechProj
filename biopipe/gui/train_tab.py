"""Unified Train tab: build a model from one or more recorded sessions.

Pick a modality, select recordings (only compatible ones can be pooled — same
electrode->channel map, sample rate, and label set), review each session's metadata +
description, train, and save the model with its metrics.
"""
from __future__ import annotations

import threading
import queue
import tkinter as tk
from tkinter import ttk, messagebox

from core.config import Config, SourceConfig, FilterConfig, Electrode
from core import training
import storage.recordings as recordings
import storage.models as models
from storage.schema import ModelMeta, ChannelInfo, now_iso


def _cfg_from_meta(meta, model_type, window_ms, step_ms) -> Config:
    f = meta.filter or {}
    bp = f.get("bandpass", [1, 40])
    cfg = Config(
        modality=meta.modality, subject=meta.subject, sample_rate=meta.sample_rate,
        electrodes=[Electrode(c.channel, c.electrode, c.role) for c in meta.channels],
        filter=FilterConfig(enabled=f.get("enabled", True), notch_hz=f.get("notch_hz", 50),
                            bandpass_low=bp[0], bandpass_high=bp[1]),
        window_ms=window_ms, step_ms=step_ms, labels=list(meta.labels), model_type=model_type,
    )
    return cfg


class TrainTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=10)
        self.grid(sticky="nsew")
        master.columnconfigure(0, weight=1); master.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1); self.columnconfigure(1, weight=1); self.rowconfigure(1, weight=1)

        self._entries = []          # recording entries currently listed
        self._trained = None        # (model, meta) pending save
        self.events: "queue.Queue" = queue.Queue()

        top = ttk.Frame(self); top.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        ttk.Label(top, text="Modality:").grid(row=0, column=0)
        self.v_mod = tk.StringVar(value="emg")
        for i, m in enumerate(["emg", "eeg"]):
            ttk.Radiobutton(top, text=m.upper(), value=m, variable=self.v_mod,
                            command=self._refresh).grid(row=0, column=i + 1, padx=4)
        ttk.Label(top, text="   Model:").grid(row=0, column=3)
        self.v_model = tk.StringVar(value="lda")
        ttk.Combobox(top, textvariable=self.v_model, values=["lda", "logreg", "rf", "svm"],
                     state="readonly", width=8).grid(row=0, column=4, padx=4)
        ttk.Label(top, text="Window/Step ms:").grid(row=0, column=5, padx=(10, 2))
        self.v_win = tk.StringVar(value="150"); self.v_step = tk.StringVar(value="40")
        ttk.Entry(top, textvariable=self.v_win, width=6).grid(row=0, column=6)
        ttk.Entry(top, textvariable=self.v_step, width=6).grid(row=0, column=7, padx=2)

        # left: recordings list (multi-select)
        lf = ttk.LabelFrame(self, text="Recordings (Ctrl/Shift-click to pool)", padding=6)
        lf.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
        lf.columnconfigure(0, weight=1); lf.rowconfigure(0, weight=1)
        self.rec_list = tk.Listbox(lf, selectmode="extended", exportselection=False)
        self.rec_list.grid(row=0, column=0, sticky="nsew")
        self.rec_list.bind("<<ListboxSelect>>", lambda e: self._on_select())

        # right: metadata of highlighted recording
        rf = ttk.LabelFrame(self, text="Selected recording", padding=6)
        rf.grid(row=1, column=1, sticky="nsew")
        rf.columnconfigure(0, weight=1); rf.rowconfigure(0, weight=1)
        self.meta_text = tk.Text(rf, width=42, height=16, wrap="word")
        self.meta_text.grid(row=0, column=0, sticky="nsew")

        # bottom: compatibility, train, metrics, save
        bottom = ttk.Frame(self); bottom.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        bottom.columnconfigure(3, weight=1)
        self.compat = ttk.Label(bottom, text="", foreground="#555")
        self.compat.grid(row=0, column=0, columnspan=4, sticky="w")
        self.train_btn = ttk.Button(bottom, text="Train", command=self._train)
        self.train_btn.grid(row=1, column=0, sticky="w", pady=6)
        ttk.Label(bottom, text="Description:").grid(row=1, column=1, padx=(10, 2))
        self.v_desc = tk.StringVar()
        ttk.Entry(bottom, textvariable=self.v_desc, width=36).grid(row=1, column=2)
        self.save_btn = ttk.Button(bottom, text="Save model", command=self._save, state="disabled")
        self.save_btn.grid(row=1, column=3, sticky="w", padx=10)
        self.metrics_lbl = ttk.Label(bottom, text="", foreground="#0a7")
        self.metrics_lbl.grid(row=2, column=0, columnspan=4, sticky="w")

        self._refresh()
        self.after(120, self._tick)

    def _refresh(self):
        self._entries = recordings.list_recordings(self.v_mod.get(), training=True)
        self.rec_list.delete(0, tk.END)
        for e in self._entries:
            m = e["meta"]
            self.rec_list.insert(tk.END, f"{e['folder'].name}   [{', '.join(m.labels)}]")
        self.meta_text.delete("1.0", tk.END)
        self.compat.config(text=f"{len(self._entries)} recording(s) for {self.v_mod.get().upper()}")
        self._trained = None
        self.save_btn.config(state="disabled")
        self.metrics_lbl.config(text="")

    def _selected_metas(self):
        return [self._entries[i]["meta"] for i in self.rec_list.curselection()]

    def _on_select(self):
        sel = self.rec_list.curselection()
        if not sel:
            return
        m = self._entries[sel[-1]]["meta"]
        chans = "\n".join(f"  ch{c.channel}: {c.electrode}" for c in m.channels)
        self.meta_text.delete("1.0", tk.END)
        self.meta_text.insert(tk.END,
            f"subject: {m.subject}\ncreated: {m.created}\nboard: {m.board}\n"
            f"sample_rate: {m.sample_rate} Hz\nsamples: {m.n_samples} ({m.duration_s}s)\n"
            f"labels: {', '.join(m.labels)}\nreference: {m.reference}  bias: {m.bias}\n"
            f"electrodes:\n{chans}\n\ndescription:\n{m.description or '(none)'}")
        metas = self._selected_metas()
        if len(metas) >= 2:
            ok, msg = recordings.validate_merge(metas)
            self.compat.config(text=msg, foreground=("#0a7" if ok else "#a00"))
        else:
            self.compat.config(text=f"{len(metas)} selected", foreground="#555")

    def _train(self):
        metas = self._selected_metas()
        if not metas:
            messagebox.showinfo("Select", "Select one or more recordings first."); return
        ok, msg = recordings.validate_merge(metas)
        if not ok:
            messagebox.showerror("Incompatible", msg); return
        try:
            win = int(float(self.v_win.get())); step = int(float(self.v_step.get()))
        except ValueError:
            messagebox.showerror("Invalid", "Window/Step must be numbers."); return
        self.train_btn.config(state="disabled")
        self.metrics_lbl.config(text="Training...", foreground="#555")
        folders = [self._entries[i]["folder"] for i in self.rec_list.curselection()]
        threading.Thread(target=self._train_worker,
                         args=(folders, metas, self.v_model.get(), win, step), daemon=True).start()

    def _train_worker(self, folders, metas, model_type, win, step):
        try:
            cfg = _cfg_from_meta(metas[0], model_type, win, step)
            sessions = []
            for f in folders:
                sig, labels, _ = recordings.load_session(f)
                sessions.append((sig, labels))
            X, Y = training.build_dataset(sessions, cfg)
            model, metrics = training.train(X, Y, cfg.labels, model_type)
            meta = ModelMeta(
                modality=cfg.modality, subject=metas[0].subject, created=now_iso(),
                model_type=model_type,
                source_files=[str(f) for f in folders],
                channels=[ChannelInfo(c.channel, c.electrode, c.role) for c in metas[0].channels],
                sample_rate=cfg.sample_rate, labels=cfg.labels,
                features=("bandpower" if cfg.modality == "eeg" else "hudgins+cov"),
                window_ms=win, step_ms=step, metrics=metrics, description="",
            )
            self.events.put(("done", model, meta, metrics))
        except Exception as e:
            self.events.put(("error", str(e)))

    def _tick(self):
        try:
            while True:
                kind, *rest = self.events.get_nowait()
                if kind == "done":
                    model, meta, metrics = rest
                    self._trained = (model, meta)
                    per = ", ".join(f"{k}={v}" for k, v in metrics["per_class_f1"].items())
                    self.metrics_lbl.config(
                        text=f"accuracy {metrics['accuracy']:.3f}  |  F1: {per}  |  "
                             f"{metrics['n_windows']} windows", foreground="#0a7")
                    self.save_btn.config(state="normal")
                    self.train_btn.config(state="normal")
                elif kind == "error":
                    self.metrics_lbl.config(text="Training failed: " + rest[0], foreground="#a00")
                    self.train_btn.config(state="normal")
        except queue.Empty:
            pass
        self.after(120, self._tick)

    def _save(self):
        if not self._trained:
            return
        model, meta = self._trained
        meta.description = self.v_desc.get().strip()
        folder = models.save_model(model, meta)
        self.metrics_lbl.config(text=f"Saved model -> {folder.name}", foreground="#0a7")
        self.save_btn.config(state="disabled")

    def stop(self):
        pass
