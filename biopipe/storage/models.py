"""Trained-model folders: save/list/load.

Each model is a folder ``<modality>_<subject>_<model>_acc<score>_<stamp>/`` holding the
serialized classifier (``<name>.joblib``) and a JSON sidecar (:class:`ModelMeta`) recording
the source files, electrode map, labels, and train/test metrics.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import joblib
from storage.schema import ModelMeta

MODELS_ROOT = Path("models")


def save_model(model, meta: ModelMeta) -> Path:
    base = meta.folder_name()
    folder = MODELS_ROOT / base
    folder.mkdir(parents=True, exist_ok=True)
    model.save(folder / f"{base}.joblib")
    meta.to_json(folder / f"{base}.json")
    return folder


def list_models(modality: str | None = None) -> List[dict]:
    """Return [{folder, joblib, meta}, …], newest first; optionally filter by modality."""
    out = []
    if not MODELS_ROOT.exists():
        return out
    pattern = f"{modality}_*" if modality else "*"
    for folder in MODELS_ROOT.glob(pattern):
        if not folder.is_dir():
            continue
        j = folder / f"{folder.name}.json"
        m = folder / f"{folder.name}.joblib"
        if j.exists() and m.exists():
            try:
                out.append({"folder": folder, "joblib": m, "meta": ModelMeta.from_json(j)})
            except Exception:
                continue
    out.sort(key=lambda e: e["meta"].created, reverse=True)
    return out


def load_model(folder: str | Path):
    """Load any saved model (EMG MultiLabelClassifier or EEG MI/ERD/ErrP) + its metadata."""
    folder = Path(folder)
    meta = ModelMeta.from_json(folder / f"{folder.name}.json")
    model = joblib.load(folder / f"{folder.name}.joblib")
    return model, meta
