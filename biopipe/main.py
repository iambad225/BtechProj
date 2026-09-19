#!/usr/bin/env python
"""biopipe — hybrid EEG/EMG acquisition, recording, training and inference.

The only file in the project root. Run it to open the single-window GUI (EMG / EEG /
Train tabs). All logic lives in the sub-packages: core/, storage/, sources/, gui/.

Config files (config/*.yaml), recordings (data/) and models (models/) are resolved
relative to this folder, so the app behaves the same wherever it is launched from.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)     # make core/ storage/ sources/ gui/ importable
os.chdir(HERE)               # config/, data/, models/ live under this folder

from gui.app import main     # noqa: E402

if __name__ == "__main__":
    main()
