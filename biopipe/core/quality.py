"""Per-channel signal quality from RAW samples (before any filtering).

The Cyton's ADS1299 saturates at +/-187,500 uV (4.5 V / gain 24). A channel stuck there is
*railed*: its electrode isn't reading (no scalp contact, a floating input or a loose wire). Always
check the raw samples: band-pass filtering turns a railed channel into ~0, and common-average
reference then paints it with the other channels' average, so on a filtered display it can look
alive.
"""
from __future__ import annotations

import numpy as np

RAIL_UV = 187500.0          # Cyton full scale at gain 24


def channel_status(raw, rail: float = RAIL_UV) -> list[str]:
    """Per column of ``raw`` (n_samples, n_channels): 'railed' | 'flat' | 'near' | 'ok'.

    railed: mostly pinned at the input limit. flat: no variation at all. near: DC offset above
    75% of the range - still recording, but close to clipping (reseat that electrode).
    """
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 2 or raw.shape[0] == 0:
        return []
    out = []
    for i in range(raw.shape[1]):
        x = raw[:, i]
        if np.mean(np.abs(x) >= 0.99 * rail) > 0.5:
            out.append("railed")
        elif np.std(x) < 1e-6:
            out.append("flat")
        elif np.median(np.abs(x)) >= 0.75 * rail:
            out.append("near")
        else:
            out.append("ok")
    return out


def dead_channels(raw, names) -> list[str]:
    """Names of channels that carry no signal (railed or flat)."""
    return [n for n, s in zip(names, channel_status(raw)) if s in ("railed", "flat")]


def check_sessions(sessions, names, session_names=None) -> None:
    """Raise ValueError naming any dead channels, so training fails with an actionable cause."""
    problems = []
    for k, sess in enumerate(sessions):
        bad = dead_channels(sess[0], names)
        if bad:
            tag = session_names[k] if session_names and k < len(session_names) else f"recording {k + 1}"
            problems.append(f"{tag}: {', '.join(bad)}")
    if problems:
        raise ValueError(
            "dead channels - stuck at the Cyton's ±187,500 µV input limit or flat, so "
            "those electrodes weren't reading and the data can't train a model ("
            + "; ".join(problems)
            + "). Fix electrode contact/wiring (watch for RAILED on the Monitor) and re-record.")
