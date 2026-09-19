"""Real-time inference pipeline: raw chunks in, smoothed multi-label decisions out.

Flow per new chunk:
    filter -> append to rolling buffer -> (every step) extract features on the latest
    window -> per-action probabilities -> EMA smoothing -> per-action hysteresis.

Latency: the heavy compute (features + predict) is timed and reported; it is typically
1-3 ms, far under the 100 ms budget. End-to-end responsiveness is governed by
``window_ms``/``step_ms`` in the config (see README), not by compute.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
import numpy as np

from core.config import Config
from core.dsp import RealTimeFilter
from core.model import MultiLabelClassifier
from core import features_emg as F


@dataclass
class Decision:
    proba: np.ndarray          # smoothed P(active) per action
    active: np.ndarray         # bool per action (after hysteresis)
    actions: list[str]
    latency_ms: float          # compute time for this prediction
    channel_mav: np.ndarray    # per-channel activity, for display
    updated: bool = False      # True only on frames where a new prediction ran

    def active_names(self) -> list[str]:
        return [a for a, on in zip(self.actions, self.active) if on]


class InferencePipeline:
    def __init__(self, cfg: Config, model: MultiLabelClassifier):
        self.cfg = cfg
        self.model = model
        self.filt = RealTimeFilter(
            cfg.sample_rate, cfg.n_channels,
            notch_hz=cfg.filter.notch_hz,
            bandpass=(cfg.filter.bandpass_low, cfg.filter.bandpass_high),
            enabled=cfg.filter.enabled,
        )
        self._win = cfg.window_samples
        self._step = cfg.step_samples
        self._buffer = np.zeros((0, cfg.n_channels))
        self._since_pred = 0

        self._smoothed = np.zeros(cfg.n_actions)
        self._active = np.zeros(cfg.n_actions, dtype=bool)
        self._on = cfg.decision.on_threshold
        self._off = cfg.decision.off_threshold
        self._alpha = cfg.decision.ema_alpha
        self._last_mav = np.zeros(cfg.n_channels)
        self._last_latency = 0.0

    def reset(self) -> None:
        self.filt.reset()
        self._buffer = np.zeros((0, self.cfg.n_channels))
        self._since_pred = 0
        self._smoothed[:] = 0
        self._active[:] = False

    def process(self, new_samples: np.ndarray) -> Decision:
        updated = False
        if new_samples.shape[0]:
            filtered = self.filt.process(new_samples)
            self._buffer = np.vstack([self._buffer, filtered])
            self._since_pred += filtered.shape[0]

            # Run at most one prediction per call, on the most recent window.
            if self._since_pred >= self._step and self._buffer.shape[0] >= self._win:
                self._since_pred = 0
                t0 = time.perf_counter()
                window = self._buffer[-self._win:]
                feat = F.extract(window)[None, :]
                proba = self.model.predict_proba(feat)[0]
                self._smoothed = self._alpha * proba + (1 - self._alpha) * self._smoothed
                self._apply_hysteresis()
                self._last_mav = np.mean(np.abs(window - window.mean(0)), axis=0)
                self._last_latency = (time.perf_counter() - t0) * 1000.0
                updated = True

            # Keep the buffer bounded.
            if self._buffer.shape[0] > self._win * 3:
                self._buffer = self._buffer[-self._win * 2:]

        return Decision(
            proba=self._smoothed.copy(),
            active=self._active.copy(),
            actions=self.cfg.actions,
            latency_ms=self._last_latency,
            channel_mav=self._last_mav.copy(),
            updated=updated,
        )

    def _apply_hysteresis(self) -> None:
        for i in range(self.cfg.n_actions):
            if not self._active[i] and self._smoothed[i] >= self._on:
                self._active[i] = True
            elif self._active[i] and self._smoothed[i] < self._off:
                self._active[i] = False
