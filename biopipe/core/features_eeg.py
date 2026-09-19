"""EEG features: per-channel band powers via Welch's PSD.

Band power is the simplest, most robust EEG feature (roadmap section 6.1) and it is
all we need for a signal monitor and an eyes-closed alpha test. The alpha *ratio*
(alpha power / broadband power) on occipital channels is the classic, training-free
check that the headset is picking up real brain activity: it jumps when the eyes close.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

DEFAULT_MONTAGE = ["Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2"]

# np.trapz was renamed to np.trapezoid in NumPy 2.0 (old name removed in 2.x).
_trapz = getattr(np, "trapezoid", None) or np.trapz


def band_powers(window: np.ndarray, fs: float, bands: dict):
    """Return ({band_name: power_per_channel}, broadband_power_per_channel).

    ``window`` is (n_samples, n_channels).
    """
    w = np.asarray(window, dtype=float)
    n = w.shape[0]
    nperseg = int(min(n, 256))
    f, psd = welch(w, fs=fs, axis=0, nperseg=nperseg)     # psd: (n_freq, n_channels)

    out = {}
    for name, rng in bands.items():
        lo, hi = float(rng[0]), float(rng[1])
        idx = (f >= lo) & (f <= hi)
        out[name] = _trapz(psd[idx], f[idx], axis=0) if idx.any() else np.zeros(w.shape[1])
    broadband = _trapz(psd, f, axis=0)
    return out, broadband


def alpha_ratio(bp: dict, broadband: np.ndarray, alpha_band: str, columns) -> float:
    """Mean alpha share of broadband over the given channel columns (0..1)."""
    if not columns:
        return 0.0
    cols = list(columns)
    alpha = np.asarray(bp[alpha_band])[cols]
    total = np.asarray(broadband)[cols]
    total = np.where(total <= 1e-12, 1e-12, total)
    return float(np.mean(alpha / total))
