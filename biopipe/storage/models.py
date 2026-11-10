"""Trained-model folders: save/list/load.

Each model is a folder ``models/<name>/`` holding the serialized classifier (``<name>.joblib``)
and a JSON sidecar (:class:`ModelMeta`) recording the modality, source files, electrode map,
labels, and train/test metrics. The Train tab names it ``<subject>_<name>_<scores>`` (subject
required, name and scores optional).
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import joblib
from storage.schema import ModelMeta, name_token

MODELS_ROOT = Path("models")


def save_model(model, meta: ModelMeta, name: str) -> Path:
    """Save under ``models/<name>/``, replacing a model of that name."""
    base = name_token(name)
    if not base:
        raise ValueError("A model needs a name.")
    folder = MODELS_ROOT / base
    folder.mkdir(parents=True, exist_ok=True)
    model.save(folder / f"{base}.joblib")
    meta.to_json(folder / f"{base}.json")
    return folder


def model_exists(name: str) -> bool:
    return bool(name_token(name)) and (MODELS_ROOT / name_token(name)).exists()


def list_models(modality: str | None = None) -> List[dict]:
    """Return [{folder, joblib, meta}, …], newest first; optionally only one modality (read
    from each model's JSON, since names no longer start with it)."""
    out = []
    if not MODELS_ROOT.exists():
        return out
    for folder in MODELS_ROOT.iterdir():
        if not folder.is_dir():
            continue
        j = folder / f"{folder.name}.json"
        m = folder / f"{folder.name}.joblib"
        if j.exists() and m.exists():
            try:
                meta = ModelMeta.from_json(j)
            except Exception:
                continue
            if modality is None or meta.modality == modality:
                out.append({"folder": folder, "joblib": m, "meta": meta})
    out.sort(key=lambda e: e["meta"].created, reverse=True)
    return out


def load_model(folder: str | Path):
    """Load any saved model (EMG MultiLabelClassifier or EEG MI/ERD/ErrP) + its metadata."""
    folder = Path(folder)
    meta = ModelMeta.from_json(folder / f"{folder.name}.json")
    model = joblib.load(folder / f"{folder.name}.joblib")
    return model, meta
