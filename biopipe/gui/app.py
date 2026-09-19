"""Main window: one window with EMG, EEG, and Train tabs."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from gui.emg_tab import EMGTab
from gui.eeg_tab import EEGTab
from gui.train_tab import TrainTab


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("biopipe — hybrid EEG–EMG interface")
        root.geometry("1000x780")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        nb = ttk.Notebook(root)
        nb.grid(sticky="nsew")
        emg_f = ttk.Frame(nb)
        eeg_f = ttk.Frame(nb)
        train_f = ttk.Frame(nb)
        for f in (emg_f, eeg_f, train_f):
            f.columnconfigure(0, weight=1)
            f.rowconfigure(0, weight=1)
        nb.add(emg_f, text="   EMG   ")
        nb.add(eeg_f, text="   EEG   ")
        nb.add(train_f, text="   Train   ")

        self.emg = EMGTab(emg_f)
        self.eeg = EEGTab(eeg_f)
        self.train = TrainTab(train_f)
        self._nb = nb
        nb.bind("<<NotebookTabChanged>>", lambda e: self._on_tab())
        root.protocol("WM_DELETE_WINDOW", self._close)

    def _on_tab(self):
        # refresh the Train list when its tab is opened (new recordings/models appear)
        try:
            if self._nb.tab(self._nb.select(), "text").strip() == "Train":
                self.train._refresh()
        except Exception:
            pass

    def _close(self):
        for t in (self.emg, self.eeg, self.train):
            try:
                t.stop()
            except Exception:
                pass
        self.root.destroy()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()
