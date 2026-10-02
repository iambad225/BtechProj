"""Electrode positions for the visual selector: the 10-20 set plus the extended 10-10 set.

Coordinates are normalized to a top-down head view (nose up): x in [0,1] left->right
(subject's left hemisphere on the left), y in [0,1] nasion(top)->inion(bottom).

Each node is tagged in TEN_TWENTY (the classic 10-20 electrodes) or not (the extended
10-10 positions). The GUI draws 10-20 nodes with a solid outline and 10-10 nodes with a
dotted outline. ALIASES gives the older 10-20 designation for electrodes that were renamed
in the 10-10 system (e.g. T7 was T3), shown in brackets in the UI.
"""
from __future__ import annotations

# ---- classic 10-20 (drawn solid) ----
_TEN20 = {
    "Fp1": (0.40, 0.12), "Fp2": (0.60, 0.12),
    "F7": (0.21, 0.28), "F3": (0.38, 0.29), "Fz": (0.50, 0.27), "F4": (0.62, 0.29), "F8": (0.79, 0.28),
    "T7": (0.13, 0.50), "C3": (0.34, 0.50), "Cz": (0.50, 0.50), "C4": (0.66, 0.50), "T8": (0.87, 0.50),
    "P7": (0.21, 0.72), "P3": (0.38, 0.71), "Pz": (0.50, 0.73), "P4": (0.62, 0.71), "P8": (0.79, 0.72),
    "O1": (0.40, 0.88), "O2": (0.60, 0.88),
    "A1": (0.04, 0.52), "A2": (0.96, 0.52),
}

# ---- extended 10-10 additions (drawn dotted) ----
_TEN10 = {
    "Fpz": (0.50, 0.115),
    "AF7": (0.27, 0.19), "AF3": (0.40, 0.205), "AFz": (0.50, 0.195), "AF4": (0.60, 0.205), "AF8": (0.73, 0.19),
    "F5": (0.295, 0.285), "F1": (0.44, 0.283), "F2": (0.56, 0.283), "F6": (0.705, 0.285),
    "FT7": (0.15, 0.39), "FC5": (0.27, 0.40), "FC3": (0.37, 0.40), "FC1": (0.435, 0.395),
    "FCz": (0.50, 0.385), "FC2": (0.565, 0.395), "FC4": (0.63, 0.40), "FC6": (0.73, 0.40), "FT8": (0.85, 0.39),
    "C5": (0.24, 0.50), "C1": (0.42, 0.50), "C2": (0.58, 0.50), "C6": (0.76, 0.50),
    "TP7": (0.15, 0.61), "CP5": (0.27, 0.60), "CP3": (0.37, 0.60), "CP1": (0.435, 0.605),
    "CPz": (0.50, 0.615), "CP2": (0.565, 0.605), "CP4": (0.63, 0.60), "CP6": (0.73, 0.60), "TP8": (0.85, 0.61),
    "P5": (0.295, 0.715), "P1": (0.44, 0.717), "P2": (0.56, 0.717), "P6": (0.705, 0.715),
    "PO7": (0.30, 0.81), "PO3": (0.42, 0.81), "POz": (0.50, 0.81), "PO4": (0.58, 0.81), "PO8": (0.70, 0.81),
    "Oz": (0.50, 0.90),
}

# name -> (x, y) for every selectable node
NODES: dict[str, tuple[float, float]] = {**_TEN20, **_TEN10}

# electrodes that belong to the classic 10-20 set (solid outline)
TEN_TWENTY = set(_TEN20.keys())

# 10-10 name -> older 10-20 designation (shown as "T7 (T3)")
ALIASES = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}

# electrodes over occipital cortex (best alpha / SSVEP)
OCCIPITAL = ["O1", "O2", "Oz"]

DEFAULT_REF = "A1"
DEFAULT_BIAS = "A2"


def node_names() -> list[str]:
    return list(NODES.keys())


def is_ten_twenty(name: str) -> bool:
    return name in TEN_TWENTY
