"""Acquisition interface.

Every data source -- the OpenBCI Cyton today, something else tomorrow -- implements
this tiny contract. Nothing downstream knows or cares which one is plugged in.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class DataSource(ABC):
    sample_rate: float
    n_channels: int

    @abstractmethod
    def start(self) -> None:
        """Open the device / begin streaming."""

    @abstractmethod
    def stop(self) -> None:
        """Close the device / stop streaming."""

    @abstractmethod
    def get_data(self) -> np.ndarray:
        """Return all samples acquired since the previous call.

        Shape is (n_new_samples, n_channels); may be (0, n_channels) when no new
        data is available yet. Channels are already in the configured order.
        """

    def __enter__(self) -> "DataSource":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
