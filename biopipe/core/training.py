"""Turn recorded sessions into a trained multi-label model.

Windows are cut *within* runs of a constant label so a window never straddles two classes;
each window's target is a multi-hot vector over the configured actions ('rest' and unknown
labels -> all zeros, so several actions can be active at once via 'a+b' labels). EMG uses
Hudgins time-domain features; EEG uses per-channel band powers (provisional, pending the
deferred EEG paradigms).
"""
from __future__ import annotations

from typing import List, Tuple
import numpy as np

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import precision_recall_fscore_support

from core.dsp import RealTimeFilter
from core import features_emg, features_eeg
from core.model import MultiLabelClassifier


def _segments(labels: List[str]):
    """Yield (start, end, label) for each run of a constant label."""
    if not labels:
        return
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[start]:
            yield start, i, labels[start]
            start = i
    yield start, len(labels), labels[start]


def _trials(labels: List[str]) -> List[int]:
    """Per-sample trial id (increments on each label change); used when a recording predates
    the stored ``trial`` column."""
    out, tid = [], 0
    for i, lab in enumerate(labels):
        if i and lab != labels[i - 1]:
            tid += 1
        out.append(tid)
    return out


def _multihot(label: str, action_index: dict) -> np.ndarray:
    v = np.zeros(len(action_index), dtype=int)
    if label and label.lower() != "rest":
        for part in label.split("+"):
            part = part.strip()
            if part in action_index:
                v[action_index[part]] = 1
    return v


def _window_features(seg: np.ndarray, cfg) -> np.ndarray:
    win, step = cfg.window_samples, cfg.step_samples
    if seg.shape[0] < win:
        return np.empty((0, 0))
    if cfg.modality == "eeg":
        band_names = list(cfg.bands.keys())
        rows = []
        for i in range(0, seg.shape[0] - win + 1, step):
            bp, _ = features_eeg.band_powers(seg[i:i + win], cfg.sample_rate, cfg.bands)
            rows.append(np.concatenate([bp[b] for b in band_names]))
        return np.asarray(rows) if rows else np.empty((0, len(band_names) * seg.shape[1]))
    return features_emg.extract_batch(seg, win, step)


def build_dataset(sessions, cfg) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (X, Y multi-hot, groups) from sessions.

    ``sessions`` items are ``(signal, labels)`` or ``(signal, labels, trials)``. Every window
    is tagged with a **group** = its trial (a run of a constant label), unique across sessions,
    so the trainer can split by trial and never leak correlated windows across train/test.
    """
    action_index = {a: i for i, a in enumerate(cfg.labels)}
    X_parts, Y_parts, G_parts = [], [], []
    g_offset = 0
    for sess in sessions:
        signal, labels = sess[0], sess[1]
        trials = sess[2] if len(sess) > 2 and sess[2] is not None else _trials(labels)
        if signal.shape[0] == 0:
            continue
        filt = RealTimeFilter(cfg.sample_rate, signal.shape[1],
                              notch_hz=cfg.filter.notch_hz,
                              bandpass=(cfg.filter.bandpass_low, cfg.filter.bandpass_high),
                              enabled=cfg.filter.enabled)
        used = set()
        for s, e, lab in _segments(labels):
            seg = filt.filter_offline(signal[s:e])
            if seg.shape[0] > cfg.transient_samples:
                seg = seg[cfg.transient_samples:]
            if cfg.modality == "eeg" and cfg.car and seg.shape[1] > 1:
                seg = seg - seg.mean(axis=1, keepdims=True)
            feats = _window_features(seg, cfg)
            if feats.shape[0] == 0:
                continue
            gid = g_offset + int(trials[s])            # trial id, made unique across sessions
            used.add(int(trials[s]))
            target = _multihot(lab, action_index)
            X_parts.append(feats)
            Y_parts.append(np.tile(target, (feats.shape[0], 1)))
            G_parts.append(np.full(feats.shape[0], gid))
        g_offset += (max(used) + 1) if used else 0
    if not X_parts:
        return (np.empty((0, 0)), np.empty((0, len(cfg.labels)), dtype=int),
                np.empty((0,), dtype=int))
    return np.vstack(X_parts), np.vstack(Y_parts), np.concatenate(G_parts)


def _combo_ids(Y: np.ndarray) -> np.ndarray:
    """Map each multi-hot row to an int id for its action-combo (all-zeros = 'rest'). Used to
    stratify the grouped folds by label combination (every trial is one combo)."""
    seen, ids = {}, []
    for row in Y:
        key = tuple(int(v) for v in row)
        ids.append(seen.setdefault(key, len(seen)))
    return np.asarray(ids)


def _emg_folds(X, Y, groups, seed, max_folds=5):
    """Return (list_of_(train_idx,test_idx), split_label). Trial-grouped, class-stratified
    k-fold when there are enough trials per combo; else a single grouped/random holdout."""
    groups = np.asarray(groups)
    strat = _combo_ids(Y)
    per_combo = [len(np.unique(groups[strat == c])) for c in np.unique(strat)]
    k = min(max_folds, min(per_combo)) if per_combo else 0
    if k >= 2:
        try:
            from sklearn.model_selection import StratifiedGroupKFold
            sp = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        except Exception:                       # older sklearn
            from sklearn.model_selection import GroupKFold
            sp = GroupKFold(n_splits=k)
        return list(sp.split(X, strat, groups)), f"{k}-fold grouped CV"
    if len(np.unique(groups)) >= 2:             # too few trials to k-fold: one grouped holdout
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        return [next(gss.split(X, Y, groups))], "grouped holdout (too few trials for k-fold)"
    idx = np.arange(X.shape[0])                  # last resort: random holdout
    return [train_test_split(idx, test_size=0.2, random_state=seed, shuffle=True)], \
           "random holdout (too few trials)"


def train(X: np.ndarray, Y: np.ndarray, groups: np.ndarray, actions: List[str], model_type: str,
          seed: int = 0, max_folds: int = 5) -> Tuple[MultiLabelClassifier, dict]:
    """Train the multi-label model with **trial-grouped, class-stratified k-fold CV**.

    Whole trials (``groups``) are held out per fold so temporally correlated windows never
    straddle the split (random window splitting would inflate the score via leakage). Per-label
    precision/recall/f1/support come from the pooled out-of-fold predictions; the final model is
    refit on all data. Falls back to a single grouped (or random) holdout when trials are scarce.
    """
    if X.shape[0] < 10:
        raise ValueError(f"Only {X.shape[0]} windows — record more data before training.")
    Y = np.asarray(Y)
    n_trials = int(len(np.unique(np.asarray(groups)))) if np.asarray(groups).size else 0
    folds, split = _emg_folds(X, Y, groups, seed, max_folds)

    Yt_parts, Yp_parts, fold_acc = [], [], []
    for tr, te in folds:
        try:
            m = MultiLabelClassifier(actions, model_type).fit(X[tr], Y[tr])
            pred = (m.predict_proba(X[te]) >= 0.5).astype(int)
        except Exception:
            continue
        Yt_parts.append(Y[te]); Yp_parts.append(pred)
        fold_acc.append(float((pred == Y[te]).all(axis=1).mean()))
    if not fold_acc:
        raise ValueError("Could not evaluate — every fold failed to train (too little data?).")
    Yt, Yp = np.vstack(Yt_parts), np.vstack(Yp_parts)

    p, r, f, s = precision_recall_fscore_support(Yt, Yp, average=None, zero_division=0)
    per_class = {a: {"precision": round(float(p[i]), 3), "recall": round(float(r[i]), 3),
                     "f1": round(float(f[i]), 3), "support": int(s[i])}
                 for i, a in enumerate(actions)}
    model = MultiLabelClassifier(actions, model_type).fit(X, Y)   # final model on all data
    metrics = {
        "accuracy": round(float(np.mean(fold_acc)), 3),          # subset accuracy (mean over folds)
        "accuracy_std": round(float(np.std(fold_acc)), 3),
        "macro_f1": round(float(np.mean(f)), 3),
        "cv_folds": len(fold_acc),
        "per_class": per_class,
        "split": split,
        "n_trials": n_trials,
        "n_windows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
    }
    return model, metrics
