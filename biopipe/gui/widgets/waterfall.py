"""Vertical scrolling spectrogram (waterfall) for the EEG monitor.

Newest spectrum is drawn as the top row; the image scrolls downward so time runs down the
Y axis and frequency runs across the X axis. Log-power maps to colour through a fixed LUT.
Scrolling is a one-row self-copy per frame (cheap in Tk); axes carry Hz and seconds.
"""
from __future__ import annotations

import time
import tkinter as tk
from tkinter import ttk
import numpy as np


def _build_lut(n: int = 256):
    stops = [(0.00, (10, 10, 30)), (0.25, (30, 60, 170)), (0.50, (0, 170, 180)),
             (0.70, (60, 200, 70)), (0.85, (240, 220, 40)), (1.00, (230, 40, 40))]
    lut = []
    for i in range(n):
        f = i / (n - 1)
        for k in range(len(stops) - 1):
            f0, c0 = stops[k]; f1, c1 = stops[k + 1]
            if f <= f1:
                t = (f - f0) / (f1 - f0) if f1 > f0 else 0.0
                r = int(c0[0] + (c1[0] - c0[0]) * t)
                g = int(c0[1] + (c1[1] - c0[1]) * t)
                b = int(c0[2] + (c1[2] - c0[2]) * t)
                lut.append(f"#{r:02x}{g:02x}{b:02x}"); break
        else:
            lut.append("#e62828")
    return lut


_LUT = _build_lut()
_BG = "#0a0a1e"
_LEFT, _BOT, _TOP, _RIGHT = 34, 18, 6, 8       # canvas margins around the plot


class Waterfall(ttk.Frame):
    def __init__(self, master, fmax: float = 45.0, dyn_range: float = 3.0, **kw):
        super().__init__(master, **kw)
        self.fmax = float(fmax)
        self.dyn = float(dyn_range)
        self.columnconfigure(0, weight=1); self.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(self, bg=_BG, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.img = None
        self.pw = self.ph = 0
        self._vmax = -12.0
        self._dt = 0.06          # seconds per row (EMA of real push interval)
        self._last = None
        self.canvas.bind("<Configure>", self._resize)

    def set_fmax(self, fmax: float):
        self.fmax = float(fmax)
        self._draw_freq_axis()

    def _resize(self, e):
        pw = max(48, int(e.width) - _LEFT - _RIGHT)
        ph = max(32, int(e.height) - _TOP - _BOT)
        if pw == self.pw and ph == self.ph:
            return
        self.pw, self.ph = pw, ph
        self.img = tk.PhotoImage(width=pw, height=ph)
        self.img.put(_BG, to=(0, 0, pw, ph))
        self.canvas.delete("all")
        self.canvas.create_image(_LEFT, _TOP, anchor="nw", image=self.img)
        self._draw_freq_axis()
        self._draw_time_axis()

    def _draw_freq_axis(self):
        self.canvas.delete("faxis")
        if self.pw < 2:
            return
        y = _TOP + self.ph + 2
        step = 10 if self.fmax <= 60 else 20
        hz = 0
        while hz <= self.fmax + 1e-6:
            x = _LEFT + (hz / self.fmax) * self.pw
            self.canvas.create_line(x, _TOP + self.ph, x, _TOP + self.ph + 3,
                                    fill="#888", tags="faxis")
            self.canvas.create_text(x, y, text=str(hz), fill="#aaa", anchor="n",
                                    font=("", 7), tags="faxis")
            hz += step
        self.canvas.create_text(_LEFT + self.pw, y + 10, text="Hz", fill="#aaa",
                                anchor="e", font=("", 7), tags="faxis")

    def _draw_time_axis(self):
        self.canvas.delete("taxis")
        if self.ph < 2:
            return
        span = self.ph * self._dt                     # seconds visible top->bottom
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = _TOP + frac * self.ph
            t = frac * span
            lbl = "0" if frac == 0 else f"-{t:.0f}"
            self.canvas.create_text(_LEFT - 4, y, text=lbl, fill="#aaa", anchor="e",
                                    font=("", 7), tags="taxis")
        self.canvas.create_text(2, _TOP, text="s", fill="#aaa", anchor="nw",
                                font=("", 7), tags="taxis")

    def clear(self):
        if self.img is not None:
            self.img.put(_BG, to=(0, 0, self.pw, self.ph))

    def push(self, psd, freqs):
        """Append one spectral column at the top and scroll everything down one row."""
        if self.img is None or self.pw < 2 or self.ph < 2:
            return
        now = time.perf_counter()
        if self._last is not None:
            self._dt = 0.9 * self._dt + 0.1 * (now - self._last)
        self._last = now

        freqs = np.asarray(freqs, dtype=float)
        psd = np.asarray(psd, dtype=float)
        mask = freqs <= self.fmax
        f, p = freqs[mask], psd[mask]
        if f.size < 2:
            return
        p = np.log10(p + 1e-12)
        self._vmax = max(self._vmax * 0.995, float(p.max()))
        lo = self._vmax - self.dyn
        cols = np.linspace(0.0, self.fmax, self.pw)   # left->right = 0..fmax Hz
        vals = np.interp(cols, f, p)
        norm = np.clip((vals - lo) / max(1e-9, self.dyn), 0.0, 1.0)
        idx = (norm * 255).astype(int)
        row = "{" + " ".join(_LUT[i] for i in idx) + "}"   # one row, pw cols
        try:
            self.img.tk.call(str(self.img), "copy", str(self.img),
                             "-from", 0, 0, self.pw, self.ph - 1, "-to", 0, 1)
            self.img.put(row, to=(0, 0))
        except tk.TclError:
            return
        self._draw_time_axis()
