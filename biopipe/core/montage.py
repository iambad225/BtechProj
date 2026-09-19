"""Standard 10-20 electrode positions for the visual selector.

Coordinates are normalized to a top-down head view (nose up): x in [0,1] left->right
(subject's left hemisphere on the left), y in [0,1] nasion(top)->inion(bottom). The GUI
widget scales these onto its canvas and draws the head outline itself.

A1/A2 are the ear/mastoid sites, offered as the usual reference/bias positions (but any
node can be tagged ref or bias in the UI).
"""
from __future__ import annotations

# name -> (x, y) in [0, 1]
NODES_10_20: dict[str, tuple[float, float]] = {
    "Fp1": (0.40, 0.12), "Fp2": (0.60, 0.12),
    "F7": (0.21, 0.28), "F3": (0.38, 0.29), "Fz": (0.50, 0.27), "F4": (0.62, 0.29), "F8": (0.79, 0.28),
    "T7": (0.14, 0.50), "C3": (0.35, 0.50), "Cz": (0.50, 0.50), "C4": (0.65, 0.50), "T8": (0.86, 0.50),
    "P7": (0.21, 0.72), "P3": (0.38, 0.71), "Pz": (0.50, 0.73), "P4": (0.62, 0.71), "P8": (0.79, 0.72),
    "O1": (0.40, 0.88), "O2": (0.60, 0.88),
    "A1": (0.05, 0.52), "A2": (0.95, 0.52),
}

# 10-20 names that also carry the older label, for the tooltip/label
ALIASES = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}

# electrodes over occipital cortex (best alpha / SSVEP)
OCCIPITAL = ["O1", "O2", "Oz"]

# reasonable default reference/bias sites
DEFAULT_REF = "A1"
DEFAULT_BIAS = "A2"


def node_names() -> list[str]:
    return list(NODES_10_20.keys())
