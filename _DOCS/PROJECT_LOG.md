# biopipe — project documentation & task log

## About this log
- Running record of every change to the project, newest entries at the bottom of the timeline
- Updated whenever a change is made; bullet points only
- Project: B.Tech "Biopotential Controlled Aerial Vehicle" — hybrid EEG–EMG brain–machine interface (COEP)
- Guiding document: `BMI_Hybrid_EEG-EMG_Project_Roadmap.pdf` (repo root)
- Code: `biopipe/` (Python); repo: github.com/iambad225/BtechProj, branch `main`
- Hardware: OpenBCI Cyton (8 channels, 250 Hz, ADS1299) + Ultracortex headset; A1 ear clip → SRB2 (reference), A2 ear clip → BIAS

## Timeline

- **2026-06-27 — repository created**
  - Initial commits (`30cd183`, `5778a75`)

- **≈ 2026-08 → 09-19 — Python pipeline replaces OpenBCI GUI → MATLAB**
  - New package `biopipe/` replaces the OpenBCI-GUI → MATLAB workflow
  - Acquisition abstracted behind `DataSource`: Cyton (BrainFlow), LSL, synthetic
  - EMG classifier is multi-label (one-vs-rest): simultaneous actions detectable; "rest" = all actions off, not a class
  - Pluggable EMG models: lda / logreg / rf / svm
  - Latency target < 100 ms (measured compute ≈ 2–3 ms)
  - GUI evolved from separate windows → one window with EMG / EEG / Train tabs and subtabs; config editable in the GUI; Monitor with live raw waveform
  - `build_exe.bat`: builds an exe, copies it to the parent folder, deletes build/dist; `requirements.txt`; heavy unused modules excluded (exe 637 → 227 MB)
  - Fixes: numpy 2 `trapz` shim; `setuptools<81` pinned (BrainFlow needs `pkg_resources`); package folder named `storage/` (not `io/`, which shadows the stdlib)

- **2026-09-19 — full rework with a defined schema** (commit `7426309`)
  - Repo root of `biopipe/` holds only `main.py`; helpers in `core/`, `storage/`, `sources/`, `gui/`
  - Recording folder `data/training/<modality>_<subject>_<YYYYMMDD_HHMMSS>/` = CSV (t, electrode-named columns, label) + JSON sidecar
  - `data/acquired/` kept separate from training data
  - Model folder `models/<modality>_<subject>_<model>_acc<score>_<timestamp>/` = joblib + JSON (sources, channels, metrics)
  - Recordings may be pooled only if electrode→channel map, sample rate and label set all match
  - EMG electrodes: free-text muscle labels; EEG electrodes: clickable 10-20 head selector (Data / Ref (SRB) / Bias)
  - Optional user `description` stored in recording and model JSON
  - Separate Train tab with EEG/EMG selector; old data discarded

- **≈ 2026-09-19 → 09-21 — EEG paradigms, montage, first docs**
  - `_DOCS/EEG_Paradigms_8channel_Analysis.pdf`: emotion, motor imagery, ERD/ERS, SSVEP/P300/ErrP feasibility on 8 channels, with literature
  - Selector extended to the 10-10 system (dotted outlines, alternate names such as "T7 (T3)")
  - EEG paradigms chosen: ERD/ERS gate + 2-class MI (CSP + LDA) + ErrP (built-in cued feedback task, ERP + shrinkage LDA)
  - Montage iterated around physically unavailable holes (FCz, CP3/CP4, CPz, FC3/FC4) → final **Fz, FC5, FC6, C3, Cz, C4, P3, P4** (ch 0–7)
  - `_DOCS/ERD-ERS_MI_ErrP_Training_Methodology.pdf` revision 1 (21 Sep)
  - Named configuration profiles (save / load / delete) in the EMG and EEG Config tabs → `config/profiles/<modality>/<name>.yaml`

- **2026-09-24 — signal diagnosis and GUI improvements**
  - Diagnosed delta-dominated, noisy traces: SRB reference and BIAS not connected; electrode impedances far too high
  - After connecting SRB + BIAS: jaw clench and resting EEG verified as real signal
  - Alpha test fixed: it was reading FC5/FC6; now uses occipital electrodes or the most posterior pair (P3/P4)
  - New EEG **Live** subtab: run one or more trained models on the stream, per-class bars, DECISION line, compute latency
  - Record tab redesigned: large subject-facing cue display, dim corner countdown
  - Subject field moved from Config to the Record tab (EMG and EEG)
  - Train tab: Refresh button
  - Monitor: gamma band (30–45 Hz); toggleable waterfall spectrogram (later made vertical); µV / seconds / Hz axes on traces and waterfall
  - Live model list overflow fixed (expands, horizontal scrollbar)

- **2026-09-30 — model architecture and evaluation**
  - MI classifier made pluggable: lda / svm / logreg / rf / tree on CSP features
  - Decision: **two EEG models only** — 3-class MI {rest, mi_left, mi_right} (rest class = ERD/ERS gate, folded in) + ErrP; separate ERD model removed
  - 3-class MI uses one-vs-rest CSP (12 features)
  - EMG wrist clench planned as the system-wide gate (deferred; focus on 8-channel EEG)
  - Advised against decision-tree regressors on raw per-channel samples for MI (not phase-locked)
  - Recordings gained a `trial` column + `n_trials` in JSON; old recordings derive trials from label runs
  - EMG train/test split changed from random windows to trial-grouped (no leakage)
  - EEG and EMG training use trial-grouped, class-stratified **k-fold CV**; report per-class precision / recall / F1 / support, balanced accuracy ± std, macro-F1; shown in the Train tab report box
  - README updated throughout

- **2026-10-02 — documentation, cue randomisation, dead-channel detection, push**
  - Source localisation judged unnecessary (8 sparse channels, no head model; CSP already spatially filters)
  - Methodology PDF rewritten as **revision 2**: two-model paradigm, settings tables for every GUI control, control-loop and trial-timeline figures, metrics guide, pre-flight checklist, references; generator script kept in `_DOCS/source/`
  - MI guidance clarified: imagine the squeeze, never actually squeeze; optional unrecorded warm-up
  - MI cue order **block-randomised**: rest/left/right once per rep, equal counts, never the same cue twice in a row (adjacent identical blocks would merge into one trial)
  - Training crash ("leading minor … not positive definite") traced to **6 railed channels** (Fz, FC5, FC6, C3, P3, P4 stuck at ±187,500 µV); common-average reference had hidden them on the Monitor
  - New `core/quality.py`: raw-sample checks (railed / flat / near rail); Monitor flags and banner; Record warns; training refuses dead channels with a clear message
  - Pushed to GitHub (`ef96edf`): code, configs, profiles, README, both PDFs, PDF generator; OpenBCI GUI install (510 MB) git-ignored; only our own `_DOCS` files tracked
  - Electrode balance analysis (headset lifts because the montage is top-heavy): frame has 33 usable holes; proposed **Fz FC5 FC6 C3 C4 CP5 CP6 Pz** (move Cz→Pz, P3→CP5, P4→CP6) plus low comfort/anchor units at F7 F8 P7 P8 (+Fpz, Oz) — awaiting fit test, not applied

- **2026-10-13 — ErrP Live test screen**
  - New EEG **ErrP Live** subtab: runs the feedback task, classifies each outcome from live EEG, scores it against what was shown
  - Shows last-trial P(error) bar, running accuracy, balanced accuracy, errors caught, false alarms, confusion matrix, per-trial log; adjustable threshold re-scores all trials
  - Record and ErrP Live share one feedback-task implementation
  - Tab-change handling made robust to Tk's spurious startup event

- **2026-10-15 — user recording session** (no code changes)
  - Recorded ErrP trial 1 (six channels railed again), ErrP trial 2, MI trial 1
  - Trained ErrP model (balanced acc 0.60 ± 0.16) and MI svm model (0.62 ± 0.07)

- **2026-10-21 — timing and latency**
  - Concern addressed: headset/USB delay could make the classifier act late or on misaligned data
  - **Stream clock**: maps PC time ↔ sample index from the lower envelope of chunk arrival times; immune to buffering bursts and tracks clock drift
  - ErrP outcomes are painted first, then time-stamped, and mapped to their EEG sample through the stream clock (Record and ErrP Live)
  - Simulation: with 500 ms delivery bursts the old marker could be off by ≈ 490 ms; the new marker stays within ≈ 1.6 ms
  - **Causal filtering** in training and both live tabs (same forward-only filter): live results now equal training exactly (verified: ErrP max ΔP ≈ 2e-16, MI ΔP = 0)
  - ErrP Live verdict now ≈ 0.8 s after the outcome (was 1.8 s)
  - **Data-delay readout** (typical / worst) on Monitor, Live and ErrP Live; warning when data arrives in bursts; FTDI latency-timer fix documented (Windows: set to 1 ms)
  - Models trained before this change show a retrain warning; models now record `metrics.filter = "causal"`
  - Record ErrP lead-in raised to 3 s (filter settles before the first outcome, same as ErrP Live)
  - Re-checked the 15 Oct data with causal filtering: ErrP trial 2 0.59 ± 0.24, MI trial 1 0.56 ± 0.10 (same as before within noise); ErrP trial 1 refused (railed)
  - README and methodology PDF updated; this log created (`_DOCS/PROJECT_LOG.md`)
  - All work since `ef96edf` (ErrP Live, timing/causal filtering, docs, this log, config/profile edits) committed to local branch **`errp-live-timing`**; `main` left at `ef96edf`; nothing pushed

- **2026-10-28 — latency questions, stage 3 presentation, ErrP Live facts test**
  - Window/step latency check (no code change)
    - Step is a free latency cut: average wait = step / 2; Live reads it from Config, no retrain needed
    - Window trades speed for accuracy; on the 15 Oct MI data a 2 s window keeps reporting the previous state for about 2 s after each cue; 500 ms never became reliable
  - Advice recorded: neural networks not worthwhile yet (18 MI trials, 60 ErrP epochs); next upgrades are more clean data, a Riemannian classifier and filter-bank CSP; the CSP is currently unregularised
  - Stage 3 mid-sem deck (`_DOCS/project stage 3 presentation/`) verified against the code, recordings and datasheets
    - Flagged: "< 1 kΩ" impedance claim; "2-class MI" upcoming item vs the 3-class pipeline; terrestrial vs aerial vehicle; minor wording on Slides 6–7
    - Speaker report written: `mid_sem_speaker_report.md` (slide-by-slide, line-by-line script, key terms, likely questions, demo run-sheet, numbers cheat sheet)
  - ErrP Live: **Show results** toggle hides the verdict panel (last result, bar, trace, stats, log) so the subject doesn't see HIT/MISS mid-test; scoring continues
  - ErrP Live: new **Test** selector — *Arrow feedback* or *True / false facts*
    - `core/errp_facts.py`: 100 true + 100 false easy statements; most false ones turn false on their last word
    - Each statement shown for **Fact time** (default 15 s, 2–60 s), same colour for true and false, 2–2.5 s fixation gap; *False share* sets the false fraction
    - While a statement is up, the −0.2…+0.8 s epoch around every 0.1 s step is classified (live P(error) trace); the statement is scored on its **peak** or **mean** P(error) against the threshold
  - README and methodology PDF updated (still 7 pages); tested headless with the synthetic source
  - Committed to local branch `errp-live-timing` only — `main` has no ErrP Live tab; nothing pushed

## Current state

- **EEG tab subtabs:** Config · Monitor · Record · Live · ErrP Live (arrow feedback or true/false facts test)
- **EEG models:** 3-class MI (CSP + pluggable classifier) and ErrP (ERP + shrinkage LDA)
- **EMG:** multi-label classifier with Config · Monitor · Record · Live
- **Evaluation:** trial-grouped stratified k-fold CV with a per-class report
- **Signal quality:** raw-sample railed / flat / near-rail checks; data-delay readout
- **Timing:** stream-clock event alignment; causal filters shared by training and live
- **Docs:** `biopipe/README.md`; `_DOCS/ERD-ERS_MI_ErrP_Training_Methodology.pdf` (rev. 2, regenerate with `python _DOCS/source/build_training_methodology.py <out.pdf>`); `_DOCS/EEG_Paradigms_8channel_Analysis.pdf`; this log

## Open items

- Retrain the 15 Oct models (trained with the old filter); record new ErrP runs (old ones used the old event timing)
- Set the dongle's FTDI latency timer to 1 ms and confirm the Monitor delay readout stays ≈ 20 ms
- Fit-test the balanced montage + anchors; then update the MI/ErrP preset and docs
- Fix the recurring railed channels (Fz, FC5, FC6, C3, P3, P4) — contact / wiring
- Collect more data: ≥ 25–30 MI trials per class, ≥ 200 ErrP trials
- Deferred: EMG-gated control loop; confidence threshold and smoothing for drone commands
- Branches: `main` (local and GitHub) = `ef96edf`; newer work on local branch `errp-live-timing`, not pushed — merge/push when approved
- Housekeeping: stale root `eeg_config.yaml` (old pipeline) left untracked
- Before the stage 3 review: fix or defend the flagged slide claims; run the demo from `errp-live-timing` with a retrained ErrP model and a 1–45 Hz Monitor band-pass
- Facts test: the arrow-trained ErrP model may not transfer (false statements classically evoke N400/P600 rather than a feedback ErrP, at an unlocked time); consider recording facts sessions to train on
