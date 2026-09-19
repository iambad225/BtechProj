"""Recording folders: save/list/load + merge validation.

A recording session is one folder ``<modality>_<subject>_<stamp>/`` holding a CSV
(``t, <electrode cols…>, label``) and a JSON sidecar (:class:`RecordingMeta`). Labelled
sessions live under ``data/training/``; raw/live captures under ``data/acquired/``.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np

from storage.schema import RecordingMeta, ChannelInfo, now_iso

TRAINING_ROOT = Path("data/training")
ACQUIRED_ROOT = Path("data/acquired")


def _root(training: bool) -> Path:
    return TRAINING_ROOT if training else ACQUIRED_ROOT


def save_recording(cfg, signal: np.ndarray, sample_labels: List[str],
                   description: str = "", training: bool = True) -> Path:
    """Persist one session. ``signal`` is (n_samples, n_channels) in electrode order;
    ``sample_labels`` is the per-sample class label ('rest' or an action, or 'a+b')."""
    meta = RecordingMeta(
        modality=cfg.modality, subject=cfg.subject, created=now_iso(), board=cfg.board,
        sample_rate=cfg.sample_rate,
        channels=[ChannelInfo(channel=e.channel, electrode=e.name, role="data")
                  for e in cfg.data_electrodes()],
        reference=cfg.reference_names, bias=cfg.bias_names,
        filter={"enabled": cfg.filter.enabled, "notch_hz": cfg.filter.notch_hz,
                "bandpass": [cfg.filter.bandpass_low, cfg.filter.bandpass_high]},
        labels=list(cfg.labels),
        n_samples=int(signal.shape[0]),
        duration_s=round(signal.shape[0] / max(1, cfg.sample_rate), 2),
        description=description,
    )
    base = meta.folder_name()
    folder = _root(training) / base
    folder.mkdir(parents=True, exist_ok=True)

    names = cfg.electrode_names
    with open(folder / f"{base}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t"] + names + ["label"])
        for i in range(signal.shape[0]):
            w.writerow([round(i / cfg.sample_rate, 5)]
                       + [f"{v:.4f}" for v in signal[i]] + [sample_labels[i]])
    meta.to_json(folder / f"{base}.json")
    return folder


def list_recordings(modality: str, training: bool = True) -> List[dict]:
    """Return [{folder, csv, meta}, …] for a modality, newest first."""
    root = _root(training)
    out = []
    if not root.exists():
        return out
    for folder in root.glob(f"{modality}_*"):
        j, c = folder / f"{folder.name}.json", folder / f"{folder.name}.csv"
        if j.exists() and c.exists():
            try:
                out.append({"folder": folder, "csv": c, "meta": RecordingMeta.from_json(j)})
            except Exception:
                continue
    out.sort(key=lambda e: e["meta"].created, reverse=True)
    return out


def load_session(folder: str | Path) -> Tuple[np.ndarray, List[str], RecordingMeta]:
    folder = Path(folder)
    meta = RecordingMeta.from_json(folder / f"{folder.name}.json")
    nch = len(meta.channels)
    sig, labels = [], []
    with open(folder / f"{folder.name}.csv", newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r, None)  # header
        for row in r:
            if len(row) < nch + 2:
                continue
            sig.append([float(x) for x in row[1:1 + nch]])
            labels.append(row[-1])
    return np.asarray(sig, dtype=float), labels, meta


def validate_merge(metas: List[RecordingMeta]) -> Tuple[bool, str]:
    """Recordings may be pooled only if their electrode→channel map, sample rate, and
    label set all match (see RecordingMeta.signature)."""
    if not metas:
        return False, "No recordings selected."
    sig0 = metas[0].signature()
    for m in metas[1:]:
        if m.signature() != sig0:
            return False, ("Incompatible selection - every recording must share the same "
                           "electrode-to-channel map, sample rate, and label set.")
    return True, "Compatible."
