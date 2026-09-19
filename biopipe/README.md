# biopipe

A modular Python toolkit for **hybrid EEG / EMG brain–machine interfacing**: acquire from an
OpenBCI Cyton (or LSL, or a hardware-free synthetic source), **configure each electrode**,
**record** labelled sessions to a clean on-disk schema, **train** classifiers from one or
more sessions, and run **live** inference — all from one window.

- **EMG** is the full pipeline today: per-channel muscle electrodes → filtering → Hudgins
  time-domain features → a **multi-label** classifier (several actions can fire at once;
  "rest" is simply nothing active) → live decoding.
- **EEG** has a visual **10-20 electrode selector**, live **band-power monitor**, and an
  **eyes-closed alpha test** now. EEG *classification* (emotion/mental-state, motor imagery,
  ERD/ERS) is scaffolded but **flagged for a later iteration** — the Train tab's EEG path is
  provisional (generic band-power features).

Everything runs on the **synthetic source with no hardware**, so you can explore the whole
flow before wiring up the board.

---

## Quick start

```bash
cd biopipe
python -m venv .venv && .venv\Scripts\activate     # Windows (source .venv/bin/activate elsewhere)
pip install -r requirements.txt
python main.py
```

`main.py` is the only file in the root; it opens one window with three tabs: **EMG**, **EEG**,
**Train**. Config, recordings, and models are stored under this folder regardless of where you
launch from.

**Try it with no hardware** (default configs use `synthetic` after you switch, or set it in
Config): EMG tab → **Record** a session → **Train** tab → select it → Train → **Live** tab →
load the model and watch it classify.

For the real board, in each tab's **Config** set `source: cyton` and the correct COM port
(Windows: Device Manager → Ports), then **close the OpenBCI GUI** (one program owns the dongle).

---

## The window

```
EMG   →  Config · Monitor · Record · Live
EEG   →  Config · Monitor · Record
Train →  (modality selector: EMG / EEG)
```

### EMG tab
- **Config** — subject, source + port, sample rate, filter (notch / band-pass), window/step,
  the **electrode table** (board channel + free-text muscle label per channel), and the
  **actions** (classes; `rest` is implicit). *Apply & Save* writes `config/emg.yaml`.
- **Monitor** — live waveform (right) + per-channel RMS bars (left); no model needed.
- **Record** — guided capture: cycles `rest` + each action for the chosen seconds × reps into
  **one session**, with an optional **description**, saved under `data/training/`.
- **Live** — pick a trained model, Start, and see a probability bar per action (green when
  active, several at once) with the compute latency.

### EEG tab
- **Config** — a **visual 10-20 head**: click a node to enable it; the mode selector chooses
  whether a click marks it **Data** (a box appears to type its board channel 0-7),
  **Ref (SRB)**, or **Bias**; plus source/filter/CAR/window and the alpha threshold.
  Occipital data electrodes (O1/O2/Oz) are used automatically for the alpha test.
  *Apply & Save* writes `config/eeg.yaml`.
- **Monitor** — waveform (right) + per-channel **band composition** δ/θ/α/β (left) + the
  **alpha (eyes-closed) test**: the light turns green when occipital alpha crosses threshold.
  On the synthetic source a *simulate eyes-closed* toggle injects alpha for testing.
- **Record** — captures a session for future EEG training (baseline if no actions defined).

### Train tab
Pick the **modality**, select one or more recordings (Ctrl/Shift-click to pool), review each
one's metadata + description, choose the model and window/step, **Train** (train/test split,
per-class F1 + subset accuracy), then **Save** the model with a description. Pooling is allowed
**only when every selected recording shares the same electrode→channel map, sample rate, and
label set** — useful for combining sessions across people; incompatible picks are blocked.

---

## On-disk schema

### Recordings — `data/training/<modality>_<subject>_<YYYYMMDD_HHMMSS>/`
Two files sharing the folder's base name:

- `<name>.csv` — `t`, one column **named by each electrode**, then `label`
  (per-sample class; `rest`, an action, or `a+b` for a simultaneous combo).
- `<name>.json` — validation metadata, shown in the Train tab before you train:

```json
{
  "schema_version": 1, "modality": "emg", "subject": "amogh",
  "created": "2026-09-19T14:35:22", "board": "cyton", "sample_rate": 250,
  "channels": [{"channel": 0, "electrode": "flexor carpi radialis", "role": "data"}],
  "reference": ["A1"], "bias": ["A2"],
  "filter": {"enabled": true, "notch_hz": 50, "bandpass": [20, 120]},
  "labels": ["fist", "open_hand"], "n_samples": 30000, "duration_s": 120,
  "description": "optional note entered before saving"
}
```

Live/inference captures go under `data/acquired/` instead of `data/training/`.

### Models — `models/<modality>_<subject>_<model>_acc<score>_<YYYYMMDD_HHMMSS>/`
- `<name>.joblib` — the trained classifier.
- `<name>.json` — modality, subject, `model_type`, `source_files`, electrode `channels`,
  `sample_rate`, `labels`, `features`, `window_ms`/`step_ms`, `metrics`
  (accuracy, per-class F1, window count), and an optional `description`.

`data/` and `models/` are git-ignored (they're your captured data).

---

## Hardware notes (OpenBCI Cyton + Ultracortex)

- 8 EXG channels stream in; map electrodes to channels in the tab's Config.
- The **2 extra electrodes are reference and bias**, at the ears/mastoids: one to the Cyton's
  **SRB** pin (reference), one to **BIAS** — **not** to AVDD/AVSS/AGND (those are power rails).
  Mark them in the EEG 10-20 selector so they're recorded in the metadata.
- EEG default montage is the standard 8-ch layout (Fp1, Fp2, C3, C4, P7, P8, O1, O2); move a
  node in the selector to use, e.g., P3/P4 instead of P7/P8.
- India mains → keep `notch_hz: 50`.

---

## Configuration reference

`config/emg.yaml` and `config/eeg.yaml` (edited via each tab's Config, or by hand):
`modality`, `subject`, `board`, `sample_rate`, `source` (type/serial_port/lsl_name),
`filter` (enabled/notch_hz/bandpass_low/high), `electrodes` (channel/name/role),
`window_ms`/`step_ms`/`transient_ms`, `labels`; EMG adds `model_type`, `decision`
(on/off/ema thresholds); EEG adds `car`, `bands`, `alpha_band`, `alpha_electrodes`,
`alpha_threshold`.

---

## Project layout

```
biopipe/
  main.py                 # only root file — launches the GUI
  config/  emg.yaml  eeg.yaml
  core/                   # config, dsp, features_emg/eeg, model, pipeline, montage, training
  storage/                # schema, recordings, models  (folder + JSON conventions)
  sources/                # base, cyton, lsl, synthetic, synthetic_eeg + create_source()
  gui/                    # app, emg_tab, eeg_tab, train_tab
    widgets/              # montage_10_20, waveform, bars
  data/    (git-ignored)  training/   acquired/
  models/  (git-ignored)
```

---

## Extending

- **New device**: implement `sources/base.py:DataSource` (`start`/`stop`/`get_data → (n,ch)`),
  register it in `sources/__init__.py:create_source`, set `source.type` in the config.
- **New EMG feature/model**: `core/features_emg.py`, `core/model.py` (models are pluggable:
  `lda` / `logreg` / `rf` / `svm`).
- **EEG classification (later)**: the record/train/model plumbing already runs on band-power
  features; the paradigm-specific pipelines (emotion, MI, ERD/ERS) drop into
  `core/features_eeg.py` + `core/training.py` when designed.

## Notes
- `setuptools<81` is pinned because `brainflow` still imports `pkg_resources` (removed in
  setuptools 81+); otherwise the Cyton source raises `No module named 'pkg_resources'`.
- EMG compute latency is ~1–3 ms per prediction; responsiveness is set by `window_ms`/`step_ms`.
