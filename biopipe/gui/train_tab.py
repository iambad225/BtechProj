"""Unified Train tab: build a model from one or more recorded sessions.

EMG -> multi-label gesture classifier (band-power/Hudgins features).
EEG -> pick a paradigm: MI (CSP+LDA), ERD gate (band-power+LDA), or ErrP (ERP+LDA).
Only compatible recordings can be pooled (same electrode->channel map, sample rate, labels).
"""
from __future__ import annotations

import threading
import queue
import tkinter as tk
from tkinter import ttk, messagebox

from core.config import Config, FilterConfig, Electrode
from core import training as emg_training
from core import eeg_training
import storage.recordings as recordings
import storage.models as models
from storage.schema import ModelMeta, ChannelInfo, now_iso

# EEG paradigm -> (label shown, recording tag it trains from, features name)
# Two models total: MI (rest/left/right, gate folded in) and ErrP.
EEG_PARADIGMS = {
    "MI (rest/L/R)": ("mi", "mi", "csp"),
    "ErrP": ("errp", "errp", "erp+lda"),
}
MI_LABEL = "MI (rest/L/R)"


def _cfg_from_meta(meta, window_ms, step_ms, paradigm="") -> Config:
    f = meta.filter or {}
    bp = f.get("bandpass", [1, 40])
    return Config(
        modality=meta.modality, subject=meta.subject, sample_rate=meta.sample_rate,
        electrodes=[Electrode(c.channel, c.electrode, c.role) for c in meta.channels],
        filter=FilterConfig(enabled=f.get("enabled", True), notch_hz=f.get("notch_hz", 50),
                            bandpass_low=bp[0], bandpass_high=bp[1]),
        window_ms=window_ms, step_ms=step_ms, labels=list(meta.labels), paradigm=paradigm,
    )


class TrainTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=10)
        self.grid(sticky="nsew")
        master.columnconfigure(0, weight=1); master.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1); self.columnconfigure(1, weight=1); self.rowconfigure(1, weight=1)

        self._entries = []
        self._trained = None
        self.events: "queue.Queue" = queue.Queue()

        top = ttk.Frame(self); top.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        ttk.Label(top, text="Modality:").grid(row=0, column=0)
        self.v_mod = tk.StringVar(value="emg")
        for i, m in enumerate(["emg", "eeg"]):
            ttk.Radiobutton(top, text=m.upper(), value=m, variable=self.v_mod,
                            command=self._on_modality).grid(row=0, column=i + 1, padx=4)
        self.v_par = tk.StringVar(value=MI_LABEL)
        self.par_combo = ttk.Combobox(top, textvariable=self.v_par, state="readonly", width=16,
                                      values=list(EEG_PARADIGMS.keys()))
        self.par_combo.grid(row=0, column=3, padx=(8, 2))
        self.par_combo.bind("<<ComboboxSelected>>", lambda e: self._on_paradigm())
        self.lbl_model = ttk.Label(top, text="Model:")
        self.lbl_model.grid(row=0, column=4, padx=(8, 0))
        self.v_model = tk.StringVar(value="lda")
        self.model_combo = ttk.Combobox(top, textvariable=self.v_model, state="readonly", width=8,
                                        values=["lda", "svm", "logreg", "rf", "tree"])
        self.model_combo.grid(row=0, column=5, padx=2)
        ttk.Label(top, text="Win/Step ms:").grid(row=0, column=6, padx=(8, 2))
        self.v_win = tk.StringVar(value="150"); self.v_step = tk.StringVar(value="40")
        ttk.Entry(top, textvariable=self.v_win, width=6).grid(row=0, column=7)
        ttk.Entry(top, textvariable=self.v_step, width=6).grid(row=0, column=8, padx=2)
        ttk.Button(top, text="↻ Refresh", command=self._refresh).grid(row=0, column=9, padx=(8, 0))

        lf = ttk.LabelFrame(self, text="Recordings (Ctrl/Shift-click to pool)", padding=6)
        lf.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
        lf.columnconfigure(0, weight=1); lf.rowconfigure(0, weight=1)
        self.rec_list = tk.Listbox(lf, selectmode="extended", exportselection=False)
        self.rec_list.grid(row=0, column=0, sticky="nsew")
        self.rec_list.bind("<<ListboxSelect>>", lambda e: self._on_select())

        rf = ttk.LabelFrame(self, text="Selected recording", padding=6)
        rf.grid(row=1, column=1, sticky="nsew")
        rf.columnconfigure(0, weight=1); rf.rowconfigure(0, weight=1)
        self.meta_text = tk.Text(rf, width=42, height=16, wrap="word")
        self.meta_text.grid(row=0, column=0, sticky="nsew")

        bottom = ttk.Frame(self); bottom.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        bottom.columnconfigure(3, weight=1)
        self.compat = ttk.Label(bottom, text="", foreground="#555")
        self.compat.grid(row=0, column=0, columnspan=4, sticky="w")
        self.train_btn = ttk.Button(bottom, text="Train", command=self._train)
        self.train_btn.grid(row=1, column=0, sticky="w", pady=6)
        ttk.Label(bottom, text="Description:").grid(row=1, column=1, padx=(10, 2))
        self.v_desc = tk.StringVar()
        ttk.Entry(bottom, textvariable=self.v_desc, width=34).grid(row=1, column=2)
        self.save_btn = ttk.Button(bottom, text="Save model", command=self._save, state="disabled")
        self.save_btn.grid(row=1, column=3, sticky="w", padx=10)
        self.metrics_lbl = ttk.Label(bottom, text="", foreground="#0a7")
        self.metrics_lbl.grid(row=2, column=0, columnspan=4, sticky="w")

        self._on_modality()
        self.after(120, self._tick)

    # ---- modality / paradigm ----
    def _is_eeg(self):
        return self.v_mod.get() == "eeg"

    def _on_modality(self):
        if self._is_eeg():
            self.par_combo.grid()
            self.v_win.set("1000"); self.v_step.set("250")
        else:
            self.par_combo.grid_remove()
            self.v_win.set("150"); self.v_step.set("40")
        self._update_model_vis()
        self._refresh()

    def _on_paradigm(self):
        self._update_model_vis()
        self._refresh()

    def _update_model_vis(self):
        """Show the classifier picker for EMG, and for EEG only on the MI paradigm."""
        show = (not self._is_eeg()) or (self.v_par.get() == MI_LABEL)
        if show:
            self.lbl_model.grid(); self.model_combo.grid()
        else:
            self.lbl_model.grid_remove(); self.model_combo.grid_remove()

    def _eeg_paradigm(self):
        return EEG_PARADIGMS[self.v_par.get()]           # (name, source_tag, features)

    def _refresh(self):
        entries = recordings.list_recordings(self.v_mod.get(), training=True)
        if self._is_eeg():
            src_tag = self._eeg_paradigm()[1]
            entries = [e for e in entries if (e["meta"].paradigm or "") == src_tag]
        self._entries = entries
        self.rec_list.delete(0, tk.END)
        for e in self._entries:
            m = e["meta"]
            self.rec_list.insert(tk.END, f"{e['folder'].name}   [{', '.join(m.labels)}]")
        self.meta_text.delete("1.0", tk.END)
        tag = f" ({self._eeg_paradigm()[1]} recordings)" if self._is_eeg() else ""
        self.compat.config(text=f"{len(self._entries)} recording(s) for {self.v_mod.get().upper()}{tag}")
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
            f"subject: {m.subject}\ncreated: {m.created}\nparadigm: {m.paradigm or '-'}\n"
            f"sample_rate: {m.sample_rate} Hz\nsamples: {m.n_samples} ({m.duration_s}s)\n"
            f"labels: {', '.join(m.labels)}\nreference: {m.reference}  bias: {m.bias}\n"
            f"electrodes:\n{chans}\n\ndescription:\n{m.description or '(none)'}")
        metas = self._selected_metas()
        if len(metas) >= 2:
            ok, msg = recordings.validate_merge(metas)
            self.compat.config(text=msg, foreground=("#0a7" if ok else "#a00"))
        else:
            self.compat.config(text=f"{len(metas)} selected", foreground="#555")

    # ---- train ----
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
        args = (folders, metas, win, step, self.v_mod.get(),
                (self._eeg_paradigm() if self._is_eeg() else None), self.v_model.get())
        threading.Thread(target=self._train_worker, args=args, daemon=True).start()

    def _train_worker(self, folders, metas, win, step, modality, eeg_par, model_type):
        try:
            sessions = [recordings.load_session(f) for f in folders]  # (sig, labels, trials, meta)
            m0 = metas[0]
            if modality == "eeg":
                pname, src_tag, feats = eeg_par
                cfg = _cfg_from_meta(m0, win, step, paradigm=pname)
                if pname == "mi":
                    cfg.model_type = model_type          # chosen classifier on CSP features
                    feats = f"csp+{model_type}"
                else:
                    model_type = pname                   # errp: fixed pipeline
                model, metrics = eeg_training.train_paradigm(
                    pname, [(s[0], s[1]) for s in sessions], cfg,
                    session_names=[f.name for f in folders])
                labels = metrics.get("classes", list(m0.labels))
                acc = metrics.get("balanced_accuracy") or 0.0
            else:
                cfg = _cfg_from_meta(m0, win, step)
                cfg.labels = list(m0.labels)
                X, Y, groups = emg_training.build_dataset(
                    [(s[0], s[1], s[2]) for s in sessions], cfg)
                model, metrics = emg_training.train(X, Y, groups, cfg.labels, model_type)
                labels = cfg.labels
                acc = metrics.get("accuracy", 0.0)
                feats = "hudgins+cov"
            meta = ModelMeta(
                modality=modality, subject=m0.subject, created=now_iso(), model_type=model_type,
                source_files=[str(f) for f in folders],
                channels=[ChannelInfo(c.channel, c.electrode, c.role) for c in m0.channels],
                sample_rate=m0.sample_rate, labels=list(labels), features=feats,
                window_ms=win, step_ms=step,
                metrics={**metrics, "accuracy": round(float(acc), 3)},
                paradigm=(eeg_par[0] if modality == "eeg" else ""),
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
                    self.metrics_lbl.config(text=self._fmt_metrics(metrics), foreground="#0a7")
                    self._show_report(metrics)
                    self.save_btn.config(state="normal")
                    self.train_btn.config(state="normal")
                elif kind == "error":
                    self.metrics_lbl.config(text="Training failed: " + rest[0], foreground="#a00")
                    self.train_btn.config(state="normal")
        except queue.Empty:
            pass
        self.after(120, self._tick)

    def _fmt_metrics(self, m):
        if "balanced_accuracy" in m:      # EEG (k-fold CV)
            clf = f" [{m['classifier']}]" if m.get("classifier") else ""
            if m.get("balanced_accuracy") is None:
                return (f"{m['paradigm'].upper()}{clf}: "
                        f"{m.get('note', 'not enough trials for CV')}")
            return (f"{m['paradigm'].upper()}{clf}: balanced acc {m['balanced_accuracy']}"
                    f"±{m.get('balanced_accuracy_std')} ({m.get('cv_folds')}-fold grouped CV)  "
                    f"macro-F1 {m.get('macro_f1')}  ·  {m.get('n_trials', '?')} trials"
                    f"  — per-class report at right")
        folds = m.get("cv_folds", 0)
        cvtxt = f"{folds}-fold grouped CV" if folds and folds >= 2 else m.get("split", "")
        return (f"EMG: subset acc {m.get('accuracy')}±{m.get('accuracy_std')} ({cvtxt})  "
                f"macro-F1 {m.get('macro_f1')}  ·  {m.get('n_trials', '?')} trials"
                f"  — per-class report at right")

    def _show_report(self, m):
        """Write the full per-class precision/recall/f1/support table into the right-hand box."""
        self.meta_text.delete("1.0", tk.END)
        self.meta_text.insert(tk.END, self._report_text(m))

    def _report_text(self, m):
        per = m.get("per_class") or {}
        lines = []
        if "balanced_accuracy" in m:      # EEG
            if m.get("balanced_accuracy") is None:
                return m.get("note", "Not enough trials for cross-validation.")
            lines += [f"{m['paradigm'].upper()}  —  {m.get('cv_folds')}-fold grouped CV "
                      f"(stratified by class)",
                      f"balanced accuracy : {m['balanced_accuracy']} "
                      f"± {m.get('balanced_accuracy_std')}",
                      f"CV accuracy       : {m.get('cv_accuracy')}",
                      f"macro F1          : {m.get('macro_f1')}"]
        else:                             # EMG
            folds = m.get("cv_folds", 0)
            head = (f"{folds}-fold grouped CV (stratified by label combo)"
                    if folds and folds >= 2 else m.get("split", ""))
            lines += [f"EMG multi-label  —  {head}",
                      f"subset accuracy : {m.get('accuracy')} ± {m.get('accuracy_std')}",
                      f"macro F1        : {m.get('macro_f1')}",
                      f"windows / trials: {m.get('n_windows')} / {m.get('n_trials')}"]
        lines += ["", f"{'class':<14}{'prec':>7}{'rec':>7}{'f1':>7}{'supp':>8}", "-" * 43]

        def fx(x):
            return "  -  " if x is None else f"{x:.3f}"
        for name, v in per.items():
            lines.append(f"{str(name):<14}{fx(v.get('precision')):>7}{fx(v.get('recall')):>7}"
                         f"{fx(v.get('f1')):>7}{int(v.get('support', 0)):>8}")
        return "\n".join(lines)

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
