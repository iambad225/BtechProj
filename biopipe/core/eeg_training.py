"""Train the EEG paradigms from recorded sessions.

  mi   : rest / left / right        -> CSP (one-vs-rest for 3-class) + a pluggable classifier;
                                       the ``rest`` class folds the ERD/ERS gate into this model
  errp : error vs correct feedback  -> event-locked ERP + shrinkage LDA

Windows/epochs are grouped by trial (recording segment / feedback event) and validated with
**trial-grouped k-fold cross-validation** (whole trials held out per fold, stratified by class
so no trial's windows leak across folds), reporting per-class precision / recall / f1 / support
on the pooled out-of-fold predictions; the final model is then refit on all data.
"""
from __future__ import annotations

from typing import List, Tuple
import numpy as np
from sklearn.metrics import (balanced_accuracy_score, accuracy_score,
                             precision_recall_fscore_support)

from core.eeg_models import MIModel, ErrPModel, bandpass, epoch_signal
from core.quality import check_sessions

MI_BAND = (8.0, 30.0)
ERRP_BAND = (1.0, 10.0)
ERRP_TMIN, ERRP_TMAX = -0.2, 0.8


def _segments(labels: List[str]):
    if not labels:
        return
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[start]:
            yield start, i, labels[start]
            start = i
    yield start, len(labels), labels[start]


def _windows(seg: np.ndarray, win: int, step: int):
    return [seg[i:i + win] for i in range(0, seg.shape[0] - win + 1, step)]


# ---------- collection ----------
def collect_mi(sessions, cfg):
    """Windows for the MI model, with rest folded in as a class (the gate).

    Returns (epochs, y, groups, classes) where ``classes`` is rest-first, e.g.
    ``["rest", "mi_left", "mi_right"]`` (2-class if rest is absent from the data). Rest windows
    make the model's ``rest`` class act as the ERD/ERS soft gate, so no separate gate model.
    """
    mi = [l for l in cfg.labels if l.lower() != "rest"][:2]
    wanted = ["rest"] + mi
    win, step, trans = cfg.window_samples, cfg.step_samples, cfg.transient_samples
    buckets = {c: [] for c in wanted}                 # class -> list of (epoch, trial)
    trial = 0
    for sig, labels in sessions:
        if sig.shape[0] == 0:
            continue
        f = bandpass(sig, cfg.sample_rate, *MI_BAND)
        for s, e, lab in _segments(labels):
            key = "rest" if lab.lower() == "rest" else lab
            if key not in buckets:
                continue
            seg = f[s + trans:e] if (e - s) > trans else f[s:e]
            wins = _windows(seg, win, step)
            for w in wins:
                buckets[key].append((w, trial))
            if wins:
                trial += 1
    classes = [c for c in wanted if buckets[c]]        # keep present classes, rest first
    epochs, y, groups = [], [], []
    for i, c in enumerate(classes):
        for w, tr in buckets[c]:
            epochs.append(w); y.append(i); groups.append(tr)
    return epochs, np.array(y), np.array(groups), classes


def collect_errp(sessions, cfg):
    epochs, y, groups = [], [], []
    trial = 0
    for sig, labels in sessions:
        if sig.shape[0] == 0:
            continue
        f = bandpass(sig, cfg.sample_rate, *ERRP_BAND)
        onsets, tags = [], []
        i = 0
        while i < len(labels):
            lab = labels[i]
            if lab in ("error", "correct"):
                onsets.append(i); tags.append(lab)
                while i < len(labels) and labels[i] == lab:
                    i += 1
            else:
                i += 1
        eps = epoch_signal(f, cfg.sample_rate, onsets, ERRP_TMIN, ERRP_TMAX,
                           baseline=(ERRP_TMIN, 0.0))
        # epoch_signal skips out-of-range onsets; align tags by re-running the range check
        valid_tags = []
        a, b = int(round(ERRP_TMIN * cfg.sample_rate)), int(round(ERRP_TMAX * cfg.sample_rate))
        for o, tg in zip(onsets, tags):
            if o + a >= 0 and o + b <= f.shape[0]:
                valid_tags.append(tg)
        for ep, tg in zip(eps, valid_tags):
            epochs.append(ep); y.append(1 if tg == "error" else 0); groups.append(trial); trial += 1
    return epochs, np.array(y), np.array(groups)


# ---------- cross-validation ----------
def _class_report(y_true, y_pred, classes) -> dict:
    """Per-class precision / recall / f1 / support (as a dict keyed by class name)."""
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(classes))), zero_division=0)
    return {classes[i]: {"precision": round(float(p[i]), 3), "recall": round(float(r[i]), 3),
                         "f1": round(float(f[i]), 3), "support": int(s[i])}
            for i in range(len(classes))}


def _grouped_cv(make_model, epochs, y, groups, n_classes, max_folds=5):
    """Trial-grouped, class-stratified k-fold CV. Returns pooled out-of-fold (y_true, y_pred),
    the per-fold balanced accuracies, and the fold count k; None if there aren't enough trials."""
    y = np.asarray(y); groups = np.asarray(groups)
    per_class_groups = [len(np.unique(groups[y == c])) for c in range(n_classes)]
    k = min(max_folds, min(per_class_groups)) if per_class_groups else 0
    if k < 2:
        return None
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=0)
    except Exception:                              # older sklearn: unstratified grouped folds
        from sklearn.model_selection import GroupKFold
        splitter = GroupKFold(n_splits=k)
    yt, yp, fold_acc = [], [], []
    for tr, te in splitter.split(np.zeros(len(y)), y, groups):
        if len(np.unique(y[tr])) < n_classes:      # need all classes present to train a fold
            continue
        m = make_model().fit([epochs[i] for i in tr], y[tr])
        pred = m.predict_proba([epochs[i] for i in te]).argmax(1)
        yt.append(y[te]); yp.append(pred)
        fold_acc.append(balanced_accuracy_score(y[te], pred))
    if not fold_acc:
        return None
    return np.concatenate(yt), np.concatenate(yp), fold_acc, k


def _cv_metrics(cv, classes) -> dict:
    if cv is None:
        return {"balanced_accuracy": None, "cv_folds": 0, "per_class": {},
                "note": "too few trials for cross-validation - record more trials"}
    yt, yp, fold_acc, k = cv
    rep = _class_report(yt, yp, classes)
    return {
        "balanced_accuracy": round(float(np.mean(fold_acc)), 3),
        "balanced_accuracy_std": round(float(np.std(fold_acc)), 3),
        "cv_accuracy": round(float(accuracy_score(yt, yp)), 3),
        "macro_f1": round(float(np.mean([v["f1"] for v in rep.values()])), 3),
        "cv_folds": k,
        "per_class": rep,
    }


# ---------- trainers ----------
def train_mi(sessions, cfg) -> Tuple[MIModel, dict]:
    epochs, y, groups, classes = collect_mi(sessions, cfg)
    clf = getattr(cfg, "model_type", "lda") or "lda"
    if len(classes) < 2:
        raise ValueError("Need at least two classes (e.g. rest + MI, or mi_left + mi_right) "
                         "in the recordings.")
    if len(epochs) < 12 or len(np.unique(y)) < len(classes):
        raise ValueError(f"Not enough MI data ({len(epochs)} windows over classes {classes}). "
                         "Record rest + both MI classes with more reps.")
    cv = _grouped_cv(lambda: MIModel(classes, clf=clf), epochs, y, groups, len(classes))
    metrics = {"paradigm": "mi", "classes": classes, "classifier": clf,
               "n_windows": len(epochs), "n_trials": int(len(np.unique(groups)))}
    metrics.update(_cv_metrics(cv, classes))
    model = MIModel(classes, clf=clf).fit(epochs, y)      # final model on all data
    return model, metrics


def train_errp(sessions, cfg) -> Tuple[ErrPModel, dict]:
    epochs, y, groups = collect_errp(sessions, cfg)
    if len(epochs) < 20 or len(np.unique(y)) < 2:
        raise ValueError(f"Not enough ErrP epochs ({len(epochs)}) with both error and correct.")
    classes = ["correct", "error"]
    cv = _grouped_cv(lambda: ErrPModel(), epochs, y, groups, 2)
    metrics = {"paradigm": "errp", "classes": classes, "n_epochs": len(epochs),
               "n_errors": int(np.sum(y == 1)), "n_trials": int(len(np.unique(groups)))}
    metrics.update(_cv_metrics(cv, classes))
    model = ErrPModel().fit(epochs, y)                    # final model on all data
    return model, metrics


def train_paradigm(paradigm, sessions, cfg, session_names=None):
    # dead (railed/flat) channels make CSP's covariance singular; fail with the real cause
    check_sessions(sessions, cfg.electrode_names, session_names)
    if paradigm == "mi":
        return train_mi(sessions, cfg)
    if paradigm == "errp":
        return train_errp(sessions, cfg)
    raise ValueError(f"Unknown EEG paradigm: {paradigm}")
