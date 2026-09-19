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

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

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


def build_dataset(sessions: List[Tuple[np.ndarray, List[str]]], cfg) -> Tuple[np.ndarray, np.ndarray]:
    """sessions: list of (signal (n,ch), per-sample labels). Returns (X, Y multi-hot)."""
    action_index = {a: i for i, a in enumerate(cfg.labels)}
    X_parts, Y_parts = [], []
    for signal, labels in sessions:
        if signal.shape[0] == 0:
            continue
        filt = RealTimeFilter(cfg.sample_rate, signal.shape[1],
                              notch_hz=cfg.filter.notch_hz,
                              bandpass=(cfg.filter.bandpass_low, cfg.filter.bandpass_high),
                              enabled=cfg.filter.enabled)
        for s, e, lab in _segments(labels):
            seg = filt.filter_offline(signal[s:e])
            if seg.shape[0] > cfg.transient_samples:
                seg = seg[cfg.transient_samples:]
            if cfg.modality == "eeg" and cfg.car and seg.shape[1] > 1:
                seg = seg - seg.mean(axis=1, keepdims=True)
            feats = _window_features(seg, cfg)
            if feats.shape[0] == 0:
                continue
            target = _multihot(lab, action_index)
            X_parts.append(feats)
            Y_parts.append(np.tile(target, (feats.shape[0], 1)))
    if not X_parts:
        return np.empty((0, 0)), np.empty((0, len(cfg.labels)), dtype=int)
    return np.vstack(X_parts), np.vstack(Y_parts)


def train(X: np.ndarray, Y: np.ndarray, actions: List[str], model_type: str,
          test_size: float = 0.2, seed: int = 0) -> Tuple[MultiLabelClassifier, dict]:
    if X.shape[0] < 10:
        raise ValueError(f"Only {X.shape[0]} windows — record more data before training.")
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=test_size, random_state=seed, shuffle=True)
    probe = MultiLabelClassifier(actions, model_type).fit(Xtr, Ytr)
    pred = (probe.predict_proba(Xte) >= 0.5).astype(int)
    accuracy = float((pred == Yte).all(axis=1).mean())      # subset (all-labels) accuracy
    per_class = {}
    for i, a in enumerate(actions):
        if Yte[:, i].sum() and (Yte[:, i] == 0).sum():
            per_class[a] = round(float(f1_score(Yte[:, i], pred[:, i], zero_division=0)), 3)
        else:
            per_class[a] = None
    model = MultiLabelClassifier(actions, model_type).fit(X, Y)   # final model on all data
    metrics = {
        "accuracy": round(accuracy, 3),
        "per_class_f1": per_class,
        "test_size": test_size,
        "n_windows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
    }
    return model, metrics
