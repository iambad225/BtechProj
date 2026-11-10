"""Unified Train tab: build a model from one or more recorded sessions.

EMG -> multi-label gesture classifier (band-power/Hudgins features).
EEG -> pick a paradigm: MI (two stages: rest vs MI, then left vs right) or ErrP (xDAWN + Riemann).
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
from core.eeg_models import MI_FEATURES
import storage.recordings as recordings
import storage.models as models
from storage.schema import ModelMeta, ChannelInfo, now_iso, name_token

# EEG paradigm -> (label shown, recording tag it trains from, features name)
# MI: rest vs MI (gate), then left vs right - one model with two stages. ErrP: event-locked.
EEG_PARADIGMS = {
    "MI (2-stage)": ("mi", "mi", "riemann"),
    "ErrP": ("errp", "errp", "xdawn+riemann"),
}
MI_LABEL = "MI (2-stage)"


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
        # MI features (EEG MI only): riemann tangent space (default) or CSP
        self.v_feat = tk.StringVar(value=MI_FEATURES[0])
        self.feat_combo = ttk.Combobox(top, textvariable=self.v_feat, state="readonly", width=8,
                                       values=list(MI_FEATURES))
        self.feat_combo.grid(row=0, column=5, padx=2)
        self.v_model = tk.StringVar(value="lda")
        self.model_combo = ttk.Combobox(top, textvariable=self.v_model, state="readonly", width=8,
                                        values=["lda", "svm", "logreg", "rf", "tree"])
        self.model_combo.grid(row=0, column=6, padx=2)
        self.v_win = tk.StringVar(value="150"); self.v_step = tk.StringVar(value="40")
        self._win_widgets = [ttk.Label(top, text="Win/Step ms:"),
                             ttk.Entry(top, textvariable=self.v_win, width=6),
                             ttk.Entry(top, textvariable=self.v_step, width=6)]
        for col, w in zip((7, 8, 9), self._win_widgets):
            w.grid(row=0, column=col, padx=(8, 2) if col == 7 else 2)
        ttk.Button(top, text="↻ Refresh", command=self._refresh).grid(row=0, column=10, padx=(8, 0))

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
        bottom.columnconfigure(7, weight=1)
        self.compat = ttk.Label(bottom, text="", foreground="#555")
        self.compat.grid(row=0, column=0, columnspan=8, sticky="w")
        self.train_btn = ttk.Button(bottom, text="Train", command=self._train)
        self.train_btn.grid(row=1, column=0, sticky="w", pady=6)
        # model name = <subject>_<name>_<scores>: the subject is typed in and required
        ttk.Label(bottom, text="Subject*:").grid(row=1, column=1, padx=(10, 2))
        self.v_subject = tk.StringVar()
        ttk.Entry(bottom, textvariable=self.v_subject, width=20).grid(row=1, column=2)
        ttk.Label(bottom, text="Name:").grid(row=1, column=3, padx=(10, 2))
        self.v_name = tk.StringVar()
        ttk.Entry(bottom, textvariable=self.v_name, width=20).grid(row=1, column=4)
        self.v_scores = tk.BooleanVar(value=True)
        ttk.Checkbutton(bottom, text="+ scores", variable=self.v_scores).grid(row=1, column=5,
                                                                            padx=(8, 0))
        self.save_btn = ttk.Button(bottom, text="Save model", command=self._save, state="disabled")
        self.save_btn.grid(row=1, column=6, sticky="w", padx=10)
        ttk.Label(bottom, text="Description:").grid(row=2, column=1, padx=(10, 2), sticky="e")
        self.v_desc = tk.StringVar()
        ttk.Entry(bottom, textvariable=self.v_desc, width=50).grid(row=2, column=2, columnspan=4,
                                                                   sticky="w")
        self.name_lbl = ttk.Label(bottom, text="", foreground="#555")
        self.name_lbl.grid(row=3, column=0, columnspan=8, sticky="w", pady=(4, 0))
        self.metrics_lbl = ttk.Label(bottom, text="", foreground="#0a7")
        self.metrics_lbl.grid(row=4, column=0, columnspan=8, sticky="w")
        for v in (self.v_subject, self.v_name, self.v_scores):
            v.trace_add("write", lambda *a: self._show_name())

        self._on_modality()
        self.after(120, self._tick)

    # ---- modality / paradigm ----
    def _is_eeg(self):
        return self.v_mod.get() == "eeg"

    def _on_modality(self):
        if self._is_eeg():
            self.par_combo.grid()
            # 2 s windows beat 1 s on both MI stages (PhysioNet check); a new decision every 0.5 s
            self.v_win.set("2000"); self.v_step.set("500"); self.v_model.set("logreg")
        else:
            self.par_combo.grid_remove()
            self.v_win.set("150"); self.v_step.set("40"); self.v_model.set("lda")
        self._update_model_vis()
        self._refresh()

    def _on_paradigm(self):
        self._update_model_vis()
        self._refresh()

    def _update_model_vis(self):
        """Classifier picker and Win/Step for EMG and EEG MI; ErrP has a fixed pipeline and
        cuts one fixed epoch per outcome, so neither applies to it."""
        mi = self._is_eeg() and self.v_par.get() == MI_LABEL
        show = (not self._is_eeg()) or mi
        for w in [self.lbl_model, self.model_combo] + self._win_widgets:
            w.grid() if show else w.grid_remove()
        self.feat_combo.grid() if mi else self.feat_combo.grid_remove()

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
        self._show_name()

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

    @staticmethod
    def _scores_text(metrics):
        """Headline score(s) for the name: both stages for two-stage MI, balanced accuracy for
        other EEG models, subset accuracy for EMG."""
        st = metrics.get("stages")
        if st and None not in (st["gate"].get("balanced_accuracy"),
                               st["side"].get("balanced_accuracy")):
            return (f"gate{st['gate']['balanced_accuracy']:.2f}"
                    f"_side{st['side']['balanced_accuracy']:.2f}")
        if "balanced_accuracy" in metrics:
            ba = metrics["balanced_accuracy"]
            return "nocv" if ba is None else f"bal{ba:.2f}"
        return f"acc{float(metrics.get('accuracy', 0.0)):.2f}"

    def _model_name(self):
        """<subject>_<name>_<scores>; the name and scores parts are optional."""
        scores = ""
        if self.v_scores.get():
            scores = self._scores_text(self._trained[1].metrics) if self._trained else "<scores>"
        parts = [name_token(self.v_subject.get()), name_token(self.v_name.get()), scores]
        return "_".join(p for p in parts if p)

    def _has_subject(self):
        return bool(name_token(self.v_subject.get()))

    def _show_name(self):
        self.name_lbl.config(text=f"Saves as:  {self._model_name()}" if self._has_subject()
                             else "Saves as:  (type the subject - required)")

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
                (self._eeg_paradigm() if self._is_eeg() else None), self.v_model.get(),
                self.v_feat.get())
        threading.Thread(target=self._train_worker, args=args, daemon=True).start()

    def _train_worker(self, folders, metas, win, step, modality, eeg_par, model_type, mi_feats):
        try:
            sessions = [recordings.load_session(f) for f in folders]  # (sig, labels, trials, meta)
            m0 = metas[0]
            if modality == "eeg":
                pname, src_tag, feats = eeg_par
                cfg = _cfg_from_meta(m0, win, step, paradigm=pname)
                if pname == "mi":
                    cfg.model_type = model_type          # chosen classifier ...
                    cfg.mi_features = mi_feats           # ... on riemann or CSP features
                    feats = f"{mi_feats}+{model_type}"
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
                    self._show_name()
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
            clf = (f" [{m['features']}+{m['classifier']}]" if m.get("features")
                   else f" [{m['classifier']}]" if m.get("classifier") else "")
            if m.get("balanced_accuracy") is None:
                return (f"{m['paradigm'].upper()}{clf}: "
                        f"{m.get('note', 'not enough trials for CV')}")
            st = m.get("stages")
            if st:                        # two-stage MI: each stage's own score first
                return (f"MI{clf}: rest vs MI {st['gate'].get('balanced_accuracy')}  ·  "
                        f"left vs right {st['side'].get('balanced_accuracy')}  ·  overall "
                        f"{m['balanced_accuracy']} (balanced acc, {m.get('cv_folds')}-fold grouped"
                        f" CV)  ·  {m.get('n_trials', '?')} trials  — report at right")
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
        if m.get("stages"):               # two-stage MI: a block per stage, then the cascade
            for title, key in (("STAGE 1  rest vs MI (gate)", "gate"),
                               ("STAGE 2  left vs right (MI windows)", "side")):
                st = m["stages"][key]
                lines += [title]
                if st.get("balanced_accuracy") is None:
                    lines += ["  " + st.get("note", "too few trials for CV"), ""]
                    continue
                lines += [f"  balanced accuracy : {st['balanced_accuracy']} "
                          f"± {st.get('balanced_accuracy_std')}"]
                lines += self._class_rows(st.get("per_class") or {}, indent="  ") + [""]
            lines += ["OVERALL  rest / left / right (both stages)"]
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
        return "\n".join(lines + [""] + self._class_rows(per))

    @staticmethod
    def _class_rows(per, indent=""):
        """Per-class precision / recall / f1 / support table lines."""
        def fx(x):
            return "  -  " if x is None else f"{x:.3f}"
        rows = [f"{indent}{'class':<14}{'prec':>7}{'rec':>7}{'f1':>7}{'supp':>8}", indent + "-" * 43]
        for name, v in per.items():
            rows.append(f"{indent}{str(name):<14}{fx(v.get('precision')):>7}"
                        f"{fx(v.get('recall')):>7}{fx(v.get('f1')):>7}{int(v.get('support', 0)):>8}")
        return rows

    def _save(self):
        if not self._trained:
            return
        if not self._has_subject():
            messagebox.showinfo("Subject", "Type the subject's name before saving the model.")
            return
        name = self._model_name()
        if models.model_exists(name) and not messagebox.askyesno(
                "Model exists", f"A model named\n{name}\nalready exists. Replace it?"):
            return
        model, meta = self._trained
        meta.description = self.v_desc.get().strip()
        meta.subject = self.v_subject.get().strip()
        folder = models.save_model(model, meta, name)
        self.metrics_lbl.config(text=f"Saved model -> {folder.name}", foreground="#0a7")
        self.save_btn.config(state="disabled")

    def stop(self):
        pass
