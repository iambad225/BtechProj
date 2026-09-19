"""Sliding-window feature extraction.

Classic time-domain sEMG features per channel (cheap, robust, well-studied), plus
cross-channel correlation which captures the *spatial pattern* of muscle activity
across the electrode ring -- the part that tells gestures apart and stays meaningful
even if electrode placement shifts a little.

For C channels the vector is 6*C single-channel features + C*(C-1)/2 correlations.
With C=4 that is 30 features. Extraction is a few microseconds, so it is never the
latency bottleneck.
"""
from __future__ import annotations

import numpy as np

_SINGLE_CH = ("MAV", "RMS", "WL", "ZC", "SSC", "VAR")


def feature_names(n_channels: int) -> list[str]:
    names = [f"{feat}_ch{c}" for feat in _SINGLE_CH for c in range(n_channels)]
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            names.append(f"CORR_{i}{j}")
    return names


def feature_dim(n_channels: int) -> int:
    return len(_SINGLE_CH) * n_channels + n_channels * (n_channels - 1) // 2


def _zero_crossings(w: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    s = w[:-1] * w[1:]
    return np.sum(s < -eps, axis=0).astype(float)


def _slope_sign_changes(w: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    d = np.diff(w, axis=0)
    s = d[:-1] * d[1:]
    return np.sum(s < -eps, axis=0).astype(float)


def extract(window: np.ndarray) -> np.ndarray:
    """Feature vector for one window shaped (n_samples, n_channels)."""
    w = np.asarray(window, dtype=float)
    w = w - w.mean(axis=0, keepdims=True)          # per-window DC removal
    n, ch = w.shape

    mav = np.mean(np.abs(w), axis=0)
    rms = np.sqrt(np.mean(w ** 2, axis=0))
    wl = np.sum(np.abs(np.diff(w, axis=0)), axis=0)
    zc = _zero_crossings(w)
    ssc = _slope_sign_changes(w)
    var = np.var(w, axis=0)

    if ch > 1:
        corr = np.corrcoef(w, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)        # flat channel -> 0 correlation
        iu = np.triu_indices(ch, k=1)
        syn = corr[iu]
    else:
        syn = np.empty(0)

    return np.concatenate([mav, rms, wl, zc, ssc, var, syn])


def extract_batch(signal_2d: np.ndarray, window: int, step: int) -> np.ndarray:
    """Slide over a recording and return (n_windows, n_features)."""
    n = signal_2d.shape[0]
    if n < window:
        return np.empty((0, feature_dim(signal_2d.shape[1])))
    feats = [extract(signal_2d[i:i + window]) for i in range(0, n - window + 1, step)]
    return np.asarray(feats)
