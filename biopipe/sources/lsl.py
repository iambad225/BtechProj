"""Generic Lab Streaming Layer source.

Use this when the data arrives as an LSL stream -- e.g. the OpenBCI GUI is still
running, or any other device that publishes over LSL. Selects the configured
channel columns from each pulled chunk.
"""
from __future__ import annotations

import numpy as np

from sources.base import DataSource


class LSLSource(DataSource):
    def __init__(self, stream_name: str, channels: list[int],
                 sample_rate: float | None = None, resolve_timeout: float = 5.0):
        try:
            from pylsl import resolve_byprop, StreamInlet
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError(
                "pylsl is required for the LSL source. Install it with `pip install pylsl`."
            ) from e

        streams = resolve_byprop("name", stream_name, timeout=resolve_timeout)
        if not streams:
            raise RuntimeError(f"No LSL stream named '{stream_name}' found.")
        self._inlet = StreamInlet(streams[0])
        info = self._inlet.info()
        self._channels = channels
        self.n_channels = len(channels)
        self.sample_rate = float(sample_rate or info.nominal_srate() or 250.0)

    def start(self) -> None:
        self._inlet.flush()

    def stop(self) -> None:
        try:
            self._inlet.close_stream()
        except Exception:
            pass

    def get_data(self) -> np.ndarray:
        chunk, _ = self._inlet.pull_chunk()
        if not chunk:
            return np.zeros((0, self.n_channels))
        arr = np.asarray(chunk, dtype=float)
        return arr[:, self._channels]
