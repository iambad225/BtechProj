"""OpenBCI Cyton acquisition via BrainFlow -- no OpenBCI GUI required.

BrainFlow talks to the Cyton directly over the USB dongle, so Python owns the whole
acquisition path. ``get_data`` drains BrainFlow's ring buffer and returns only the
configured EMG channels.
"""
from __future__ import annotations

import numpy as np

from sources.base import DataSource


class CytonSource(DataSource):
    def __init__(self, serial_port: str, channels: list[int],
                 mac_address: str = "", sample_rate: float | None = None):
        try:
            from brainflow.board_shim import (
                BoardShim, BrainFlowInputParams, BoardIds,
            )
        except ImportError as e:  # pragma: no cover - depends on optional dep
            raise ImportError(
                "brainflow is required for the Cyton source. Install it with "
                "`pip install brainflow`."
            ) from e

        self._board_id = BoardIds.CYTON_BOARD.value
        params = BrainFlowInputParams()
        params.serial_port = serial_port
        if mac_address:
            params.mac_address = mac_address

        self._board = BoardShim(self._board_id, params)
        exg = BoardShim.get_exg_channels(self._board_id)   # board rows holding EMG
        try:
            self._rows = [exg[c] for c in channels]
        except IndexError as e:
            raise ValueError(
                f"channels {channels} out of range; board exposes {len(exg)} EXG channels"
            ) from e

        self.n_channels = len(channels)
        self.sample_rate = float(sample_rate or BoardShim.get_sampling_rate(self._board_id))

    def start(self) -> None:
        self._board.prepare_session()
        self._board.start_stream()

    def stop(self) -> None:
        try:
            self._board.stop_stream()
            self._board.release_session()
        except Exception:
            pass

    def get_data(self) -> np.ndarray:
        data = self._board.get_board_data()           # (rows, n_samples); clears buffer
        if data.size == 0:
            return np.zeros((0, self.n_channels))
        return data[self._rows, :].T
