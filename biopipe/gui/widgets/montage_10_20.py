"""Visual 10-20 / 10-10 electrode selector.

A head diagram with a clickable node per electrode. The classic 10-20 electrodes are drawn
with a solid outline; the extended 10-10 positions with a dotted outline. Electrodes that
were renamed between the systems show the old name in brackets (e.g. "T7 (T3)"). A mode
selector chooses what a click does: enable as Data (a box appears to type its board channel
0-7), mark as Ref (SRB) / Bias, or Clear. Reads back a list of ``core.config.Electrode``.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from core.montage import NODES, TEN_TWENTY, ALIASES
from core.config import Electrode

ROLE_COLORS = {
    "off": ("#ececec", "#a7a7a7"),
    "data": ("#4a90d9", "#204a70"),
    "ref": ("#f39c12", "#a35f00"),
    "bias": ("#e74c3c", "#8a1c11"),
}


class Montage1020(ttk.Frame):
    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self.roles = {n: "off" for n in NODES}
        self.chan_vars = {n: tk.StringVar() for n in NODES}
        self.mode = tk.StringVar(value="data")

        top = ttk.Frame(self)
        top.grid(row=0, column=0, sticky="w", pady=(0, 4))
        ttk.Label(top, text="Click sets:").grid(row=0, column=0, padx=(0, 6))
        for i, (m, txt) in enumerate([("data", "Data"), ("ref", "Ref (SRB)"),
                                      ("bias", "Bias"), ("off", "Clear")]):
            ttk.Radiobutton(top, text=txt, value=m, variable=self.mode).grid(row=0, column=i + 1, padx=2)

        self.canvas = tk.Canvas(self, bg="white", highlightthickness=1, highlightbackground="#ccc")
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self._entries: dict[str, tk.Entry] = {}
        self.canvas.bind("<Configure>", lambda e: self._redraw())
        self.after(60, self._redraw)

    def _node_xy(self, name, W, H):
        nx, ny = NODES[name]
        size = min(W, H) - 60
        return W / 2 + (nx - 0.5) * size, H / 2 + (ny - 0.5) * size

    def _redraw(self):
        cv = self.canvas
        cv.delete("all")
        W, H = cv.winfo_width(), cv.winfo_height()
        if W < 20 or H < 20:
            return
        size = min(W, H) - 60
        cx, cy, r = W / 2, H / 2, size / 2 * 1.16
        cv.create_oval(cx - r, cy - r, cx + r, cy + r, outline="#cccccc", width=2)
        cv.create_polygon(cx - 11, cy - r + 2, cx + 11, cy - r + 2, cx, cy - r - 15,
                          outline="#cccccc", fill="", width=2)                       # nose

        for w in self._entries.values():
            w.destroy()
        self._entries = {}
        rad = max(10, int(min(W, H) * 0.023))
        for name in NODES:
            x, y = self._node_xy(name, W, H)
            role = self.roles[name]
            fill, outline = ROLE_COLORS[role]
            solid = name in TEN_TWENTY
            cv.create_oval(x - rad, y - rad, x + rad, y + rad, fill=fill, outline=outline,
                           width=(2 if solid else 1), dash=(() if solid else (2, 2)),
                           tags=("node", name))
            tcol = "white" if role != "off" else "#333333"
            alias = ALIASES.get(name)
            if alias:
                cv.create_text(x, y - 4, text=name, font=("", 7, "bold"), fill=tcol, tags=("node", name))
                cv.create_text(x, y + 5, text=f"({alias})", font=("", 6), fill=tcol, tags=("node", name))
            else:
                cv.create_text(x, y, text=name, font=("", 7, "bold"), fill=tcol, tags=("node", name))
            cv.tag_bind(name, "<Button-1>", lambda e, n=name: self._on_click(n))
            if role == "data":
                ent = tk.Entry(cv, width=2, textvariable=self.chan_vars[name], justify="center")
                cv.create_window(x, y + rad + 9, window=ent)
                self._entries[name] = ent

        cv.create_text(10, H - 22, anchor="w", font=("", 8), fill="#999",
                       text="solid = 10-20   ··· dotted = 10-10 extended   ·   () = alt. designation")
        cv.create_text(10, H - 8, anchor="w", font=("", 8), fill="#999",
                       text="blue = data (enter channel)   orange = ref (SRB)   red = bias")

    def _on_click(self, name):
        mode = self.mode.get()
        if mode == "off" or self.roles[name] == mode:
            self.roles[name] = "off"
            self.chan_vars[name].set("")
        else:
            self.roles[name] = mode
            if mode == "data" and not self.chan_vars[name].get():
                self.chan_vars[name].set(str(self._next_free_channel()))
        self._redraw()

    def _next_free_channel(self):
        used = {int(v.get()) for n, v in self.chan_vars.items()
                if self.roles[n] == "data" and v.get().isdigit()}
        for c in range(64):
            if c not in used:
                return c
        return 0

    def get_electrodes(self) -> list[Electrode]:
        out = []
        for name, role in self.roles.items():
            if role == "data":
                s = self.chan_vars[name].get()
                out.append(Electrode(channel=int(s) if s.isdigit() else None, name=name, role="data"))
            elif role in ("ref", "bias"):
                out.append(Electrode(channel=None, name=name, role=role))
        return out

    def set_electrodes(self, electrodes) -> None:
        self.roles = {n: "off" for n in NODES}
        for v in self.chan_vars.values():
            v.set("")
        for e in electrodes:
            if e.name in self.roles:
                self.roles[e.name] = e.role if e.role in ("data", "ref", "bias") else "off"
                if e.role == "data" and e.channel is not None:
                    self.chan_vars[e.name].set(str(e.channel))
        self._redraw()
