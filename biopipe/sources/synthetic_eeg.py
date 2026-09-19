"""Hardware-free EEG source.

Produces reddish (1/f-ish) noise so the band-composition display looks EEG-like, and
can inject a ~10 Hz occipital alpha rhythm on demand (``set_alpha``) to exercise the
eyes-closed alpha test with no headset attached.
"""
from __future__ import annotations

import time
import numpy as np

from sources.base import DataSource


class EEGSyntheticSource(DataSource):
    def __init__(self, n_channels: int, sample_rate: float = 250.0,
                 alpha_gain=None, seed: int = 0):
        self.n_channels = n_channels
        self.sample_rate = float(sample_rate)
        self._rng = np.random.default_rng(seed)
        self._last_t = None
        self._prev = np.zeros(n_channels)     # AR(1) state for 1/f-ish colouring
        self._phase = 0.0
        self._alpha = False
        # default: strongest alpha on the last two channels (occipital in the default montage)
        if alpha_gain is None:
            alpha_gain = np.full(n_channels, 0.3)
            if n_channels >= 2:
                alpha_gain[-2:] = 1.0
        self._alpha_gain = np.asarray(alpha_gain, dtype=float)

    def set_alpha(self, on: bool) -> None:
        self._alpha = bool(on)

    def start(self) -> None:
        self._last_t = time.perf_counter()

    def stop(self) -> None:
        self._last_t = None

    def get_data(self) -> np.ndarray:
        now = time.perf_counter()
        if self._last_t is None:
            self._last_t = now
        n = int((now - self._last_t) * self.sample_rate)
        if n <= 0:
            return np.zeros((0, self.n_channels))
        self._last_t += n / self.sample_rate

        # AR(1) reddened noise -> 1/f-like EEG background
        white = self._rng.standard_normal((n, self.n_channels)) * 6.0
        out = np.empty((n, self.n_channels))
        prev = self._prev
        for i in range(n):
            prev = 0.85 * prev + white[i]
            out[i] = prev
        self._prev = prev

        if self._alpha:
            t = (np.arange(n) + self._phase) / self.sample_rate
            wave = np.sin(2 * np.pi * 10.0 * t)[:, None]     # 10 Hz alpha
            out = out + wave * (18.0 * self._alpha_gain)[None, :]
            self._phase = (self._phase + n) % self.sample_rate
        return out
