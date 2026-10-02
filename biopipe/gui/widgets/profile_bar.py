"""A small bar for saving/loading named configuration profiles.

Embedded at the top of each modality's Config tab. ``get_cfg`` returns the current Config
(reading the tab's widgets, may raise on invalid input); ``on_load`` applies a loaded Config
back to the tab.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox

from core import config as C


class ProfileBar(ttk.Frame):
    def __init__(self, master, modality, get_cfg, on_load, **kw):
        super().__init__(master, **kw)
        self.modality = modality
        self.get_cfg = get_cfg
        self.on_load = on_load

        ttk.Label(self, text="Profile:").grid(row=0, column=0, padx=(0, 4))
        self.var = tk.StringVar()
        self.combo = ttk.Combobox(self, textvariable=self.var, state="readonly", width=18)
        self.combo.grid(row=0, column=1)
        ttk.Button(self, text="Load", width=6, command=self._load).grid(row=0, column=2, padx=3)
        ttk.Button(self, text="Delete", width=7, command=self._delete).grid(row=0, column=3)
        ttk.Label(self, text="Save as:").grid(row=0, column=4, padx=(14, 4))
        self.name = ttk.Entry(self, width=16)
        self.name.grid(row=0, column=5)
        ttk.Button(self, text="Save", width=6, command=self._save).grid(row=0, column=6, padx=3)
        self.status = ttk.Label(self, text="", foreground="#0a7")
        self.status.grid(row=1, column=0, columnspan=7, sticky="w", pady=(2, 0))
        self.refresh()

    def refresh(self):
        names = C.list_profiles(self.modality)
        self.combo["values"] = names
        if names and self.var.get() not in names:
            self.combo.current(0)

    def _load(self):
        name = self.var.get()
        if not name:
            return
        try:
            cfg = C.load_profile(self.modality, name)
            self.on_load(cfg)
            self.status.config(text=f"Loaded profile '{name}'.", foreground="#0a7")
        except Exception as e:
            messagebox.showerror("Load profile", str(e))

    def _save(self):
        nm = self.name.get().strip()
        if not nm:
            messagebox.showinfo("Save profile", "Enter a name to save as."); return
        try:
            cfg = self.get_cfg()
        except Exception as e:
            messagebox.showerror("Invalid config", str(e)); return
        C.save_profile(cfg, nm)
        self.name.delete(0, tk.END)
        self.refresh()
        self.var.set(C._profile_slug(nm))
        self.status.config(text=f"Saved profile '{nm}'.", foreground="#0a7")

    def _delete(self):
        name = self.var.get()
        if not name:
            return
        if not messagebox.askyesno("Delete profile", f"Delete profile '{name}'?"):
            return
        C.delete_profile(self.modality, name)
        self.var.set("")
        self.refresh()
        self.status.config(text=f"Deleted profile '{name}'.", foreground="#a00")
