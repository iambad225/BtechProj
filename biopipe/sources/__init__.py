"""Acquisition sources + a factory that builds the one named in a Config.

A source is modality-agnostic hardware-wise; only the synthetic source differs (EMG
bursts vs EEG 1/f + alpha). ``create_source(cfg)`` returns the right object for
``cfg.source.type`` using the config's data-channel indices.
"""
from __future__ import annotations

from sources.base import DataSource
from sources.synthetic import SyntheticSource
from sources.synthetic_eeg import EEGSyntheticSource


def create_source(cfg) -> DataSource:
    kind = cfg.source.type.lower()
    channels = cfg.channel_indices
    if kind == "synthetic":
        if cfg.modality == "eeg":
            import numpy as np
            gain = np.full(cfg.n_channels, 0.3)
            for c in cfg.alpha_columns():
                if c < gain.size:
                    gain[c] = 1.0
            return EEGSyntheticSource(cfg.n_channels, cfg.sample_rate, alpha_gain=gain)
        return SyntheticSource(cfg.n_channels, cfg.sample_rate, n_actions=max(cfg.n_labels, 1))
    if kind == "cyton":
        from sources.cyton import CytonSource
        return CytonSource(cfg.source.serial_port, channels,
                           cfg.source.mac_address, cfg.sample_rate)
    if kind == "lsl":
        from sources.lsl import LSLSource
        return LSLSource(cfg.source.lsl_name, channels, cfg.sample_rate)
    raise ValueError(f"Unknown source type: {cfg.source.type!r}")


__all__ = ["DataSource", "SyntheticSource", "EEGSyntheticSource", "create_source"]
