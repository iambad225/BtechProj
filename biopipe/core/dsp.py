"""Causal, stateful real-time filtering.

A band-pass isolates the EMG band and kills DC drift; an optional notch removes
mains hum. Filter state (``zi``) is carried between calls so that streaming the
signal in small chunks gives exactly the same result as filtering it all at once
-- this is what makes training (offline) and inference (online) consistent.
"""
from __future__ import annotations

import numpy as np
from scipy import signal


class RealTimeFilter:
    def __init__(
        self,
        sample_rate: float,
        n_channels: int,
        notch_hz: float = 50.0,
        bandpass: tuple[float, float] = (20.0, 120.0),
        enabled: bool = True,
        order: int = 4,
    ):
        self.enabled = enabled
        self.n_channels = n_channels
        nyq = sample_rate / 2.0

        low, high = bandpass
        low = max(1.0, low)
        high = min(high, nyq * 0.99)
        self.sos_bp = signal.butter(
            order, [low / nyq, high / nyq], btype="band", output="sos"
        )

        if notch_hz and 0 < notch_hz < nyq:
            b, a = signal.iirnotch(notch_hz, Q=30.0, fs=sample_rate)
            self.sos_notch = signal.tf2sos(b, a)
        else:
            self.sos_notch = None

        self.reset()

    def reset(self) -> None:
        """Clear filter memory (call before each independent recording)."""
        self.zi_bp = np.zeros((self.sos_bp.shape[0], 2, self.n_channels))
        if self.sos_notch is not None:
            self.zi_notch = np.zeros((self.sos_notch.shape[0], 2, self.n_channels))

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """Filter a chunk shaped (n_samples, n_channels) and return the same shape."""
        if not self.enabled or chunk.shape[0] == 0:
            return chunk
        y = np.asarray(chunk, dtype=float)
        if self.sos_notch is not None:
            y, self.zi_notch = signal.sosfilt(self.sos_notch, y, axis=0, zi=self.zi_notch)
        y, self.zi_bp = signal.sosfilt(self.sos_bp, y, axis=0, zi=self.zi_bp)
        return y

    def filter_offline(self, signal_2d: np.ndarray) -> np.ndarray:
        """Filter a whole recording from a clean state (used when building the dataset)."""
        self.reset()
        return self.process(signal_2d)
