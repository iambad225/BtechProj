"""An expanding multi-channel waveform canvas (stacked traces, dark background).

Optional axes: pass ``fs`` and ``labels`` to ``draw`` and the canvas gains a left gutter
with each trace's electrode name + its current peak amplitude in uV, and a bottom time
axis in seconds. Each trace is auto-scaled independently, so the per-channel "+/-N uV"
label is what gives the vertical scale its real units.
"""
from __future__ import annotations

import tkinter as tk
import numpy as np

_COLORS = ["#4a90d9", "#2ecc40", "#f39c12", "#e74c3c",
           "#9b59b6", "#1abc9c", "#e67e22", "#95a5a6"]


class Waveform(tk.Canvas):
    def __init__(self, master, **kw):
        super().__init__(master, bg="#111", highlightthickness=0, **kw)

    def draw(self, buf: np.ndarray | None, fs: float | None = None, labels=None,
             flags=None) -> None:
        """``flags``: optional per-channel raw-quality status ('railed'/'flat'/'near'/'ok')."""
        self.delete("wave")
        if buf is None or buf.shape[0] < 2:
            return
        W = self.winfo_width();  W = W if W > 1 else 480
        H = self.winfo_height(); H = H if H > 1 else 180
        ch = buf.shape[1]
        axed = fs is not None
        left = 58 if axed else 0
        bot = 16 if axed else 0
        top = 4 if axed else 0
        plot_w = max(10, W - left - 8)
        plot_h = max(10, H - top - bot)
        band = plot_h / ch
        step = max(1, buf.shape[0] // plot_w)
        idx = np.arange(0, buf.shape[0], step)
        xs = np.linspace(left, left + plot_w, len(idx))
        for c in range(ch):
            status = flags[c] if flags and c < len(flags) else "ok"
            dead = status in ("railed", "flat")
            col = "#555" if dead else _COLORS[c % len(_COLORS)]   # a dead trace is a CAR artefact
            seg = buf[idx, c] - buf[idx, c].mean()
            m = float(np.max(np.abs(seg)))
            amp = (band * 0.42) / m if m > 1e-6 else 0.0
            base = top + band * (c + 0.5)
            ys = base - seg * amp
            if axed:
                self.create_line(left, base, left + plot_w, base,
                                 fill="#333", width=1, tags="wave")
                name = labels[c] if labels and c < len(labels) else f"ch{c}"
                self.create_text(left - 5, base - 5, text=name, fill=col, anchor="e",
                                 font=("", 8, "bold"), tags="wave")
                if dead:
                    sub, subcol, subfont = status.upper(), "#e74c3c", ("", 7, "bold")
                elif status == "near":
                    sub, subcol, subfont = "near rail", "#f39c12", ("", 7, "bold")
                else:
                    sub, subcol, subfont = f"±{m:.0f}µV", "#888", ("", 7)
                self.create_text(left - 5, base + 7, text=sub, fill=subcol, anchor="e",
                                 font=subfont, tags="wave")
            pts = []
            for x, y in zip(xs, ys):
                pts.extend([float(x), float(y)])
            if len(pts) >= 4:
                self.create_line(*pts, fill=col, width=1, tags="wave")
        if axed:
            dur = buf.shape[0] / float(fs)             # seconds shown (oldest..newest=now)
            y = top + plot_h + 2
            t = 0
            while t <= dur + 1e-6:
                x = left + plot_w - (t / dur) * plot_w if dur > 0 else left + plot_w
                self.create_line(x, top + plot_h, x, top + plot_h + 3, fill="#666", tags="wave")
                lbl = "0s" if t == 0 else f"-{t:.0f}"
                self.create_text(x, y, text=lbl, fill="#aaa", anchor="n",
                                 font=("", 7), tags="wave")
                t += 1
