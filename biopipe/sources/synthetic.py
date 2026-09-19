"""Hardware-free source for development and demos.

Generates band-limited noise in real time. When one or more actions are marked
active (``set_active``), each adds a distinctive amplitude pattern across the
channels, so the *whole* pipeline -- recording, training, multi-label inference --
can be exercised end-to-end with no electrodes attached. Because patterns add,
marking two actions active is a faithful test of simultaneous detection.
"""
from __future__ import annotations

import time
import numpy as np

from sources.base import DataSource


class SyntheticSource(DataSource):
    def __init__(self, n_channels: int, sample_rate: float = 250.0,
                 n_actions: int = 0, seed: int = 0):
        self.n_channels = n_channels
        self.sample_rate = float(sample_rate)
        self._rng = np.random.default_rng(seed)
        self._active: set[int] = set()
        self._last_t = None
        # A fixed, distinct channel gain pattern per action index.
        pat_rng = np.random.default_rng(1234)
        self._patterns = 0.5 + pat_rng.random((max(n_actions, 8), n_channels)) * 1.5

    def set_active(self, indices) -> None:
        self._active = set(int(i) for i in indices)

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

        baseline = 8.0
        sig = self._rng.standard_normal((n, self.n_channels)) * baseline
        for idx in self._active:
            gain = self._patterns[idx % self._patterns.shape[0]]
            burst = self._rng.standard_normal((n, self.n_channels)) * (40.0 * gain)
            sig = sig + burst
        return sig
