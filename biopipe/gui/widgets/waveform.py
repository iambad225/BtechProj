"""An expanding multi-channel waveform canvas (stacked traces, dark background)."""
from __future__ import annotations

import tkinter as tk
import numpy as np

_COLORS = ["#4a90d9", "#2ecc40", "#f39c12", "#e74c3c",
           "#9b59b6", "#1abc9c", "#e67e22", "#95a5a6"]


class Waveform(tk.Canvas):
    def __init__(self, master, **kw):
        super().__init__(master, bg="#111", highlightthickness=0, **kw)

    def draw(self, buf: np.ndarray | None) -> None:
        self.delete("wave")
        if buf is None or buf.shape[0] < 2:
            return
        W = self.winfo_width();  W = W if W > 1 else 480
        H = self.winfo_height(); H = H if H > 1 else 180
        ch = buf.shape[1]
        band = H / ch
        step = max(1, buf.shape[0] // W)
        idx = np.arange(0, buf.shape[0], step)
        xs = np.linspace(0, W, len(idx))
        for c in range(ch):
            seg = buf[idx, c] - buf[idx, c].mean()
            m = float(np.max(np.abs(seg)))
            amp = (band * 0.42) / m if m > 1e-6 else 0.0
            base = band * (c + 0.5)
            ys = base - seg * amp
            pts = []
            for x, y in zip(xs, ys):
                pts.extend([float(x), float(y)])
            if len(pts) >= 4:
                self.create_line(*pts, fill=_COLORS[c % len(_COLORS)], width=1, tags="wave")
