"""Expanding horizontal bar widgets.

LevelBars   one bar per row (label · bar · value) that fills width; used for EMG per-channel
            RMS and EMG live action probabilities.
StackedBars one segmented bar per row; used for EEG band composition (delta/theta/alpha/beta).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class LevelBars(ttk.Frame):
    def __init__(self, master, row_labels, fill="#4a90d9", **kw):
        super().__init__(master, **kw)
        self.fill = fill
        self.rows = []
        self.columnconfigure(1, weight=1)
        for i, name in enumerate(row_labels):
            self.rowconfigure(i, weight=1)
            ttk.Label(self, text=name, width=18, anchor="e").grid(
                row=i, column=0, padx=6, pady=2, sticky="e")
            cv = tk.Canvas(self, height=22, bg="#eee", highlightthickness=1,
                           highlightbackground="#ccc")
            cv.grid(row=i, column=1, sticky="nsew", pady=2)
            rect = cv.create_rectangle(0, 0, 0, 22, fill=fill, width=0)
            val = ttk.Label(self, text="", width=7, anchor="e")
            val.grid(row=i, column=2, padx=6, sticky="e")
            self.rows.append((cv, rect, val))

    def set(self, i, frac, text="", on=None):
        cv, rect, val = self.rows[i]
        W = cv.winfo_width();  W = W if W > 1 else 300
        H = cv.winfo_height(); H = H if H > 1 else 22
        frac = max(0.0, min(1.0, float(frac)))
        cv.coords(rect, 0, 0, int(W * frac), H)
        if on is not None:
            cv.itemconfig(rect, fill="#2ecc40" if on else self.fill)
        val.config(text=text)


class StackedBars(ttk.Frame):
    """One segmented bar per row; segments given as {name: color} in draw order."""
    def __init__(self, master, row_labels, segments: dict, **kw):
        super().__init__(master, **kw)
        self.segments = segments
        self.rows = []
        self.columnconfigure(1, weight=1)
        for i, name in enumerate(row_labels):
            self.rowconfigure(i, weight=1)
            ttk.Label(self, text=name, width=8, anchor="e").grid(
                row=i, column=0, padx=(2, 6), pady=1, sticky="e")
            cv = tk.Canvas(self, width=240, height=18, bg="#f4f4f4",
                           highlightthickness=1, highlightbackground="#ccc")
            cv.grid(row=i, column=1, sticky="nsew", pady=1)
            self.rows.append(cv)

    def set(self, i, shares: dict):
        """shares: {segment_name: fraction_of_width}; drawn left to right in segments order."""
        cv = self.rows[i]
        cv.delete("seg")
        W = cv.winfo_width();  W = W if W > 1 else 240
        H = cv.winfo_height(); H = H if H > 1 else 18
        x = 0.0
        for name, color in self.segments.items():
            w = max(0.0, min(1.0, float(shares.get(name, 0.0)))) * W
            cv.create_rectangle(x, 0, x + w, H, fill=color, width=0, tags="seg")
            x += w
