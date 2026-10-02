"""Metadata schema for recordings and models, with folder-naming helpers.

A *recording* folder and a *model* folder each hold exactly two files that share the
folder's base name: the payload (``.csv`` / ``.joblib``) and a ``.json`` sidecar described
here. The JSON captures which electrode fed which channel, so recordings can be validated
and only compatible ones pooled for training (see ``recordings.validate_merge``).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List

SCHEMA_VERSION = 1


def now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def stamp_from_iso(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%Y%m%d_%H%M%S")
    except Exception:
        return datetime.now().strftime("%Y%m%d_%H%M%S")


def slug(text: str) -> str:
    """Filesystem-safe token: lowercase alnum + dashes."""
    s = re.sub(r"[^0-9A-Za-z]+", "-", str(text).strip().lower()).strip("-")
    return s or "unnamed"


@dataclass
class ChannelInfo:
    channel: int
    electrode: str
    role: str = "data"


@dataclass
class RecordingMeta:
    modality: str
    subject: str
    created: str
    board: str
    sample_rate: int
    channels: List[ChannelInfo]
    reference: List[str]
    bias: List[str]
    filter: dict
    labels: List[str]
    n_samples: int
    duration_s: float
    paradigm: str = ""              # eeg: "mi" | "errp" | "" (generic/monitor)
    n_trials: int = 0               # number of labelled trials (runs of a constant label)
    description: str = ""
    schema_version: int = SCHEMA_VERSION

    def folder_name(self) -> str:
        return f"{self.modality}_{slug(self.subject)}_{stamp_from_iso(self.created)}"

    def signature(self):
        """What must match to pool recordings: data map + rate + label set."""
        chans = tuple(sorted((c.channel, c.electrode) for c in self.channels))
        return (chans, int(self.sample_rate), tuple(sorted(self.labels)))

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict) -> "RecordingMeta":
        d = dict(d)
        d["channels"] = [ChannelInfo(**c) for c in d.get("channels", [])]
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_json(cls, path: str | Path) -> "RecordingMeta":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class ModelMeta:
    modality: str
    subject: str
    created: str
    model_type: str
    source_files: List[str]
    channels: List[ChannelInfo]
    sample_rate: int
    labels: List[str]
    features: str
    window_ms: int
    step_ms: int
    metrics: dict
    paradigm: str = ""              # eeg: "mi" | "erd" | "errp"
    description: str = ""
    schema_version: int = SCHEMA_VERSION

    def accuracy(self) -> float:
        return float(self.metrics.get("accuracy", 0.0))

    def folder_name(self) -> str:
        return (f"{self.modality}_{slug(self.subject)}_{self.model_type}"
                f"_acc{self.accuracy():.2f}_{stamp_from_iso(self.created)}")

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict) -> "ModelMeta":
        d = dict(d)
        d["channels"] = [ChannelInfo(**c) for c in d.get("channels", [])]
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelMeta":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
