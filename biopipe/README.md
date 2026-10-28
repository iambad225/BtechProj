# biopipe

A modular Python toolkit for **hybrid EEG / EMG brain–machine interfacing**: acquire from an
OpenBCI Cyton (or LSL, or a hardware-free synthetic source), **configure each electrode**,
**record** labelled sessions to a clean on-disk schema, **train** classifiers from one or
more sessions, and run **live** inference — all from one window.

- **EMG** — per-channel muscle electrodes → filtering → Hudgins time-domain features → a
  **multi-label** classifier (several actions can fire at once; "rest" is simply nothing active)
  → live decoding.
- **EEG** — a visual **10-20 / 10-10 electrode selector**, a live **band-power monitor**
  (δ/θ/α/β/γ + a toggleable **waterfall** spectrogram) with an **eyes-closed alpha test**, and
  two trainable decoders: **motor imagery** as a 3-class **rest / left / right** model (CSP + a
  pluggable classifier — the `rest` class folds the ERD/ERS gate in) and **ErrP** error detection.

Both modalities record to the same schema and are evaluated with **trial-grouped** splits (see
[On-disk schema](#on-disk-schema)), so accuracy isn't inflated by leaking correlated windows.

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
EEG   →  Config · Monitor · Record · Live
Train →  (modality selector: EMG / EEG)
```

Each Config tab has a **Profile** bar (save/load named configurations under
`config/profiles/<modality>/`). **Subject** is set per-recording in the **Record** tab.

### EMG tab
- **Config** — source + port, sample rate, filter (notch / band-pass), window/step, the
  **electrode table** (board channel + free-text muscle label per channel), the **actions**
  (classes; `rest` is implicit), and a **Profile** bar. *Apply & Save* writes `config/emg.yaml`.
- **Monitor** — live waveform with µV / seconds axes (right) + per-channel RMS bars (left).
- **Record** — **subject** + guided capture: cycles `rest` + each action for the chosen
  seconds × reps into **one session** (each block is one **trial**), with an optional
  **description**, saved under `data/training/`.
- **Live** — pick a trained model, Start, and see a probability bar per action (green when
  active, several at once) with the compute latency.

### EEG tab
- **Config** — a **visual 10-20 / 10-10 head**: click a node to enable it; the mode selector
  marks a click as **Data** (type its board channel 0-7), **Ref (SRB)**, or **Bias**; plus
  source/filter/CAR/window, the alpha threshold, a **Profile** bar, and a **Load MI/ErrP preset**
  button (the 8-ch Fz/FC5/FC6/C3/Cz/C4/P3/P4 montage). The alpha test uses occipital electrodes,
  or the most **posterior** available (e.g. P3/P4) when O1/O2/Oz aren't in the montage.
  *Apply & Save* writes `config/eeg.yaml`.
- **Monitor** — waveform with µV / Hz / seconds axes (right) + per-channel **band composition**
  δ/θ/α/β/γ (left) + the **alpha (eyes-closed) test**, plus a toggleable **waterfall**
  spectrogram (frequency × time). On the synthetic source a *simulate eyes-closed* toggle
  injects alpha for testing. Electrodes are flagged **RAILED** / **FLAT** / *near rail* from the
  **raw** samples (a railed Cyton input sits at ±187,500 µV): trust the flags, not the traces —
  with common-average reference on, a dead channel still draws a convincing trace. Recording
  warns if a run captured dead channels, and training refuses them with a clear message.
- **Record** — **subject** + a large subject-facing **cue** display with a corner countdown.
  Two paradigms: **MI** (rest / left / right cues, block-randomised: each class once per rep,
  equal counts, never the same cue twice in a row; cue length = **Seconds/class**) and two **ErrP**
  tasks that mark each outcome as error or correct:
  - **ErrP (cursor task)** — recommended. A target box lights up left or right; the subject
    presses **←/→** toward it and the cursor moves — the opposite way at **Error rate**. Their own
    command going wrong (an *interaction* ErrP) gives a far stronger response than watching: on our
    data the cursor task is decodable (0.78 balanced accuracy) and the arrow task is not. A key away
    from the target (a slip) or no key within **Max wait s** leaves the trial unlabelled and it is
    repeated.
  - **ErrP (arrow task)** — observation: a target arrow for 0.9 s, then the outcome arrow.
  - Right and wrong outcomes are drawn identically (same colour, no OK/WRONG text), so only the
    error itself can drive the response. **Outcome s** (default 1.0, ≥ 0.8 so the outcome never
    disappears inside an epoch) sets how long it stays up.

  Saved under `data/training/`.
- **Live** — run a trained MI model on the stream; a probability bar per class
  (rest / left / right) with the top class highlighted (a `rest` prediction = gate closed).
- **ErrP Live** — test a trained ErrP model on live EEG. ErrP is event-locked, so this runs the
  same cursor or arrow task as Record; at each outcome the −0.2…+0.8 s epoch is cut from the
  continuously filtered stream (same causal 1–10 Hz filter and baseline as training) and
  classified as soon as it ends (~0.8 s after the outcome), then scored against what actually
  happened: per-trial log, P(error) bar, running accuracy, errors caught, false alarms and a
  confusion matrix. The threshold is adjustable and re-scores every trial.
  - **Test** selector: *Cursor (press ←/→)*, *Arrow (watch)*, or *True / false facts*: easy statements
    (`core/errp_facts.py`, 100 true + 100 false, most false ones turning false on their last word)
    shown one at a time for **Fact time** seconds (default 15), with a 2–2.5 s fixation gap.
    *False share* sets the fraction of false statements. While a statement is up, the moment the
    error is noticed isn't known, so every point is treated as a possible event: the model's
    epoch around it is classified every 0.1 s (live P(error) trace). The statement's **Score** is
    the *peak* or *mean* of those values, thresholded like the other tests (TRUE / FALSE rows in
    the confusion matrix). True and false statements are drawn in the same colour.
  - **Max wait (s)** (cursor) / **Outcome (s)**: as in Record. A verdict needs the 0.8 s epoch
    after its outcome; with Outcome ≥ ~1.5 s it arrives while that outcome is still on screen
    (otherwise it lands in the next trial, but is still that trial's verdict — the log and the
    "last: #n" line name the trial). Models from the old ErrP pipeline ask to be retrained.
  - **Show results** toggle hides the verdict panel (last result, bar, trace, stats, log) so the
    subject doesn't see HIT/MISS feedback mid-test; scoring carries on and reappears when ticked.

**Timing.** Each outcome is painted first, then time-stamped, and mapped to its EEG sample through
a *stream clock* (the lower envelope of chunk arrival times against the 250 Hz sample index), so
USB/serial buffering bursts don't shift the marker — in Record and ErrP Live alike. Monitor, Live
and ErrP Live show a **data delay** readout (how stale the newest sample is) and warn when data
arrives in bursts. Training and the live tabs use the **same causal filters**, so live results
match training exactly.

### Train tab
Pick the **modality**; for EEG choose the **paradigm** (`MI (rest/L/R)` or `ErrP`) and, for MI,
the **classifier** (`lda` / `svm` / `logreg` / `rf` / `tree`) and Win/Step (ErrP has a fixed
pipeline and epoch, so neither is shown for it). Train ErrP on one task's recordings — cursor and
arrow responses differ. Select one or more recordings
(Ctrl/Shift-click to pool; **↻ Refresh** to pick up new ones), review each one's metadata +
description, then **Train** and **Save** with a description.

Evaluation never splits windows randomly (that would leak temporally-correlated windows across
the split and inflate the score):
- **EEG** uses **trial-grouped, class-stratified k-fold cross-validation** (up to 5 folds,
  `StratifiedGroupKFold`): whole trials are held out per fold, and the pooled out-of-fold
  predictions give **per-class precision / recall / f1 / support** plus balanced accuracy
  (mean ± std across folds) and macro-F1 — shown in the report box on the right.
- **EMG** (multi-label) uses the **same trial-grouped k-fold CV**, stratified by label-combination
  (each trial is one combo), reporting **per-class precision/recall/f1/support**, macro-F1, and
  subset accuracy (mean ± std across folds); a single grouped/random holdout only when trials are
  scarce.

Pooling is allowed **only when every selected recording shares the same electrode→channel map,
sample rate, and label set**; incompatible picks are blocked.

Two EEG models total: the 3-class **MI** decoder (its `rest` class is the ERD/ERS gate, so
there's no separate gate model) and **ErrP**.

**Why the ErrP model uses spatial filtering.** On this headset ~95% of the 1–10 Hz power is common
to all eight channels (the ear reference's own activity), and the error response is a small
per-channel difference under it. Time-binned amplitudes + shrinkage LDA on the raw channels stayed
at chance (0.55 on the cursor task). The ErrP model is now **xDAWN spatial filters (Rivet et al.
2009) + Riemannian tangent space + logistic regression** — the BCI Challenge @ NER 2015 winning
approach for feedback errors: per class, xDAWN finds the channel combinations that maximise the
evoked response; each epoch is stacked under the two filtered class averages, its shrunk
(Ledoit–Wolf) covariance is mapped to the tangent space at the training mean, and a balanced
logistic regression outputs P(error). On the 28 Oct cursor sessions (180 trials): 0.78 balanced
accuracy (5-fold grouped CV; label-shuffle p = 0.01), 0.73–0.81 on a held-out session, unchanged
without the eye-movement-sensitive FC5/FC6 and without blink-sized epochs; the error − correct
difference peaks at Cz (negative ~350 ms, positive ~420 ms).

---

## On-disk schema

### Recordings — `data/training/<modality>_<subject>_<YYYYMMDD_HHMMSS>/`
Two files sharing the folder's base name:

- `<name>.csv` — `t`, one column **named by each electrode**, then `trial`, then `label`:
  - `label` — per-sample class (`rest`, an action, or `a+b` for a simultaneous combo).
  - `trial` — the trial/segment index: it increments each time the label changes, so a run of
    one constant label (one `rest` block, one MI cue) is one trial. This is the **grouping unit
    for evaluation**: the trainer splits by `trial` so windows from the same trial never land on
    both sides of the train/test split. Splitting individual windows randomly would leak
    temporally-correlated (and overlapping) windows across the split and inflate accuracy.
- `<name>.json` — validation metadata, shown in the Train tab before you train:

```json
{
  "schema_version": 1, "modality": "emg", "subject": "amogh",
  "created": "2026-09-19T14:35:22", "board": "cyton", "sample_rate": 250,
  "channels": [{"channel": 0, "electrode": "flexor carpi radialis", "role": "data"}],
  "reference": ["A1"], "bias": ["A2"],
  "filter": {"enabled": true, "notch_hz": 50, "bandpass": [20, 120]},
  "labels": ["fist", "open_hand"], "n_samples": 30000, "duration_s": 120,
  "paradigm": "", "n_trials": 24,
  "description": "optional note entered before saving"
}
```

Both EMG and EEG use this schema (EEG `electrode` holds 10-20 names, `paradigm` is `mi`/`errp`).
Recordings that predate the `trial` column still load — their trials are derived from label runs.
Live/inference captures go under `data/acquired/` instead of `data/training/`.

### Models — `models/<modality>_<subject>_<model>_acc<score>_<YYYYMMDD_HHMMSS>/`
- `<name>.joblib` — the trained classifier.
- `<name>.json` — modality, subject, `model_type`, `paradigm` (EEG), `source_files`, electrode
  `channels`, `sample_rate`, `labels`, `features`, `window_ms`/`step_ms`, `metrics`, and an
  optional `description`. `metrics` carries a **`per_class`** map of precision/recall/f1/support
  for every class; EMG adds subset `accuracy`, `n_trials`, `split`; EEG adds `balanced_accuracy`
  (± `balanced_accuracy_std`), `macro_f1`, `cv_folds`, `classes`, and `classifier`.

`data/` and `models/` are git-ignored (they're your captured data).

---

## Hardware notes (OpenBCI Cyton + Ultracortex)

- 8 EXG channels stream in; map electrodes to channels in the tab's Config.
- The **2 extra electrodes are reference and bias**, at the ears/mastoids: one to the Cyton's
  **SRB** pin (reference), one to **BIAS** — **not** to AVDD/AVSS/AGND (those are power rails).
  Mark them in the EEG 10-20 selector so they're recorded in the metadata. **Both must be
  physically connected** — without them you record mains hum and drift, not EEG (the giveaway is
  a band composition swamped by delta). Keep electrode impedances low (hundreds of kΩ or less).
- EEG default montage is the standard 8-ch layout (Fp1, Fp2, C3, C4, P7, P8, O1, O2); move a
  node in the selector to use, e.g., P3/P4 instead of P7/P8.
- India mains → keep `notch_hz: 50`.
- **Dongle latency (Windows):** the FTDI USB-serial driver buffers 16 ms by default and can make
  data arrive in bursts. Set it to 1 ms: Device Manager → Ports (COM & LPT) → the dongle's COM
  port → Properties → Port Settings → Advanced → *Latency Timer* = 1. The Monitor's data-delay
  readout should then stay around 20 ms at worst.

---

## Configuration reference

`config/emg.yaml` and `config/eeg.yaml` (edited via each tab's Config, or by hand):
`modality`, `subject`, `board`, `sample_rate`, `source` (type/serial_port/lsl_name),
`filter` (enabled/notch_hz/bandpass_low/high), `electrodes` (channel/name/role),
`window_ms`/`step_ms`/`transient_ms`, `labels`; EMG adds `model_type`, `decision`
(on/off/ema thresholds); EEG adds `car`, `bands` (δ/θ/α/β/γ), `alpha_band`, `alpha_electrodes`,
`alpha_threshold`, `paradigm`. Named profiles are saved copies under
`config/profiles/<modality>/<name>.yaml`.

---

## Project layout

```
biopipe/
  main.py                 # only root file — launches the GUI
  config/  emg.yaml  eeg.yaml
  core/                   # config, dsp, features_emg/eeg, model, pipeline, montage, training,
                          #   csp, eeg_models, eeg_training
  storage/                # schema, recordings, models  (folder + JSON conventions)
  sources/                # base, cyton, lsl, synthetic, synthetic_eeg + create_source()
  gui/                    # app, emg_tab, eeg_tab, train_tab
    widgets/              # montage_10_20, waveform, bars, waterfall, profile_bar
  config/  profiles/      # named saved configs per modality
  data/    (git-ignored)  training/   acquired/
  models/  (git-ignored)
```

---

## Extending

- **New device**: implement `sources/base.py:DataSource` (`start`/`stop`/`get_data → (n,ch)`),
  register it in `sources/__init__.py:create_source`, set `source.type` in the config.
- **New EMG feature/model**: `core/features_emg.py`, `core/model.py` (models are pluggable:
  `lda` / `logreg` / `rf` / `svm` / `tree`).
- **EEG paradigms**: MI (CSP + pluggable classifier, one-vs-rest CSP for the 3-class case) and
  ErrP (xDAWN + Riemannian tangent space + logistic regression) live in `core/eeg_models.py` +
  `core/eeg_training.py`
  (`core/csp.py` for CSP); add a paradigm by extending `train_paradigm` and the Train tab's list.

## Notes
- `setuptools<81` is pinned because `brainflow` still imports `pkg_resources` (removed in
  setuptools 81+); otherwise the Cyton source raises `No module named 'pkg_resources'`.
- EMG compute latency is ~1–3 ms per prediction; responsiveness is set by `window_ms`/`step_ms`.
