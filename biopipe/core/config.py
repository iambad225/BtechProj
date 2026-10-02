"""Runtime configuration for a modality (EMG or EEG).

One ``Config`` object drives a modality's live behaviour. Electrodes are first-class:
each carries a board ``channel`` (0-7) and a human ``name`` (EEG: a 10-20 label such as
``C3``; EMG: a muscle label such as ``flexor carpi radialis``) and a ``role``
(``data`` / ``ref`` / ``bias``). The same electrode list is stamped into every recording's
JSON so data can be validated and pooled later.

A single dataclass covers both modalities (EEG-only fields like ``bands`` simply go unused
for EMG) which keeps load/save and the downstream pipeline uniform.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import yaml

STANDARD_MONTAGE = ["Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2"]


@dataclass
class Electrode:
    channel: Optional[int]      # board channel 0-7 for data electrodes; None for ref/bias
    name: str                   # EEG 10-20 label or EMG muscle label
    role: str = "data"          # "data" | "ref" | "bias"


@dataclass
class SourceConfig:
    type: str = "cyton"         # cyton | lsl | synthetic
    serial_port: str = "COM3"
    lsl_name: str = "obci_eeg1"
    mac_address: str = ""


@dataclass
class FilterConfig:
    enabled: bool = True
    notch_hz: float = 50.0
    bandpass_low: float = 20.0
    bandpass_high: float = 120.0


@dataclass
class DecisionConfig:
    on_threshold: float = 0.60
    off_threshold: float = 0.40
    ema_alpha: float = 0.40


@dataclass
class Config:
    modality: str = "emg"                      # "emg" | "eeg"
    subject: str = "subject"
    board: str = "cyton"
    sample_rate: int = 250

    source: SourceConfig = field(default_factory=SourceConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    electrodes: List[Electrode] = field(default_factory=list)

    window_ms: int = 150
    step_ms: int = 40
    transient_ms: int = 400
    labels: List[str] = field(default_factory=list)

    # --- EMG ---
    model_type: str = "lda"
    decision: DecisionConfig = field(default_factory=DecisionConfig)

    # --- EEG ---
    car: bool = True
    bands: dict = field(default_factory=lambda: {
        "delta": [1.0, 4.0], "theta": [4.0, 8.0],
        "alpha": [8.0, 12.0], "beta": [13.0, 30.0], "gamma": [30.0, 45.0]})
    alpha_band: str = "alpha"
    alpha_electrodes: List[str] = field(default_factory=lambda: ["O1", "O2"])
    alpha_threshold: float = 0.40
    paradigm: str = "mi"             # eeg record/train paradigm: "mi" | "errp" | "monitor"

    # ---------- derived views over the electrode map ----------
    def data_electrodes(self) -> List[Electrode]:
        """Data electrodes with an assigned channel, ordered by channel number."""
        ds = [e for e in self.electrodes if e.role == "data" and e.channel is not None]
        return sorted(ds, key=lambda e: e.channel)

    @property
    def channel_indices(self) -> List[int]:
        return [e.channel for e in self.data_electrodes()]

    @property
    def electrode_names(self) -> List[str]:
        return [e.name for e in self.data_electrodes()]

    @property
    def reference_names(self) -> List[str]:
        return [e.name for e in self.electrodes if e.role == "ref"]

    @property
    def bias_names(self) -> List[str]:
        return [e.name for e in self.electrodes if e.role == "bias"]

    @property
    def n_channels(self) -> int:
        return len(self.data_electrodes())

    @property
    def n_labels(self) -> int:
        return len(self.labels)

    @property
    def window_samples(self) -> int:
        return max(1, round(self.sample_rate * self.window_ms / 1000))

    @property
    def step_samples(self) -> int:
        return max(1, round(self.sample_rate * self.step_ms / 1000))

    @property
    def transient_samples(self) -> int:
        return max(0, round(self.sample_rate * self.transient_ms / 1000))

    def alpha_columns(self) -> List[int]:
        """Column indices (within the captured array) of the alpha electrodes.

        Falls back to the most posterior available electrode(s) when the named
        ``alpha_electrodes`` aren't in the montage, so the eyes-closed alpha test
        never silently reads zero (alpha is a posterior rhythm).
        """
        names = self.electrode_names
        cols = [i for i, n in enumerate(names) if n in self.alpha_electrodes]
        if cols:
            return cols
        from core.montage import NODES     # local import avoids a module cycle
        posterior = sorted((i for i, n in enumerate(names) if n in NODES),
                           key=lambda i: NODES[names[i]][1], reverse=True)
        return posterior[:2]

    # aliases the EMG inference pipeline expects
    @property
    def channels(self) -> List[int]:
        return self.channel_indices

    @property
    def actions(self) -> List[str]:
        return self.labels

    @property
    def n_actions(self) -> int:
        return len(self.labels)

    # ---------- (de)serialisation ----------
    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        d = dict(d or {})
        src = SourceConfig(**(d.pop("source", {}) or {}))
        filt = FilterConfig(**(d.pop("filter", {}) or {}))
        dec = DecisionConfig(**(d.pop("decision", {}) or {}))
        elecs = [Electrode(**e) for e in (d.pop("electrodes", []) or [])]
        known = {f for f in cls.__dataclass_fields__
                 if f not in ("source", "filter", "decision", "electrodes")}
        for k in list(d.keys()):
            if k not in known:
                d.pop(k)
        return cls(**d, source=src, filter=filt, decision=dec, electrodes=elecs)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, default_flow_style=False)


def mi_errp_montage() -> List[Electrode]:
    """8-ch montage for the MI + ERD + ErrP build, using only physically available holes
    (excludes FCz/CP3/CP4/CPz/FC3/FC4, which are unavailable on this headset).

    Data: Fz, FC5, FC6, C3, Cz, C4, P3, P4 -> channels 0-7. Ref/bias at the ears.
    Serves MI (symmetric lateral pairs FC5/FC6 + C3/C4 + P3/P4 for CSP), the ERD gate
    (C3/Cz/C4), and ErrP (Fz frontal, Cz central, P3/P4 for the Pe).
    """
    names = ["Fz", "FC5", "FC6", "C3", "Cz", "C4", "P3", "P4"]
    elecs = [Electrode(channel=i, name=n, role="data") for i, n in enumerate(names)]
    elecs += [Electrode(channel=None, name="A1", role="ref"),
              Electrode(channel=None, name="A2", role="bias")]
    return elecs


def default_config(modality: str) -> "Config":
    """Sensible starting config for a modality (used when no file exists yet)."""
    modality = modality.lower()
    if modality == "eeg":
        elecs = [Electrode(channel=i, name=STANDARD_MONTAGE[i], role="data") for i in range(8)]
        elecs += [Electrode(channel=None, name="A1", role="ref"),
                  Electrode(channel=None, name="A2", role="bias")]
        return Config(
            modality="eeg", sample_rate=250, electrodes=elecs,
            filter=FilterConfig(enabled=True, notch_hz=50, bandpass_low=1, bandpass_high=40),
            window_ms=1000, step_ms=250, car=True,
            source=SourceConfig(type="cyton"),
            labels=[],
        )
    # EMG default
    muscles = ["flexor carpi radialis", "extensor digitorum",
               "pronator teres", "brachioradialis"]
    elecs = [Electrode(channel=i, name=muscles[i], role="data") for i in range(4)]
    return Config(
        modality="emg", sample_rate=250, electrodes=elecs,
        filter=FilterConfig(enabled=True, notch_hz=50, bandpass_low=20, bandpass_high=120),
        window_ms=150, step_ms=40,
        source=SourceConfig(type="cyton"),
        labels=["fist", "open_hand", "wrist_flexion"],   # actions only; 'rest' is implicit (all-zeros)
    )


# ---------------- named configuration profiles ----------------
PROFILES_ROOT = Path("config/profiles")


def _profile_slug(name: str) -> str:
    s = re.sub(r"[^0-9A-Za-z._ -]", "", str(name)).strip().replace(" ", "_")
    return s or "unnamed"


def profiles_dir(modality: str) -> Path:
    return PROFILES_ROOT / modality


def list_profiles(modality: str) -> List[str]:
    d = profiles_dir(modality)
    return sorted(p.stem for p in d.glob("*.yaml")) if d.exists() else []


def profile_path(modality: str, name: str) -> Path:
    return profiles_dir(modality) / f"{_profile_slug(name)}.yaml"


def save_profile(cfg: "Config", name: str) -> Path:
    """Save a Config under a user-given name (config/profiles/<modality>/<name>.yaml)."""
    path = profile_path(cfg.modality, name)
    cfg.save(path)
    return path


def load_profile(modality: str, name: str) -> "Config":
    return Config.load(profile_path(modality, name))


def delete_profile(modality: str, name: str) -> None:
    p = profile_path(modality, name)
    if p.exists():
        p.unlink()
