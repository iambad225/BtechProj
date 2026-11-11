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
  - ErrP arrow task: **Target s / Outcome s** now adjustable in Record and ErrP Live (defaults 0.9 / 1.0 s; outcome ≥ 0.8 s so it never disappears inside an epoch)
    - Reported issue: verdicts seemed to belong to the previous step. They don't — each verdict carries its own trial label and epoch (verified); it only arrives ~0.8 s after the outcome, i.e. during the next fixation when Outcome is 1.0 s
    - Measured with Outcome 2.0 s: verdicts 0.80–0.87 s after onset, while the outcome is still shown; the "last:" line now names the trial (#n)
    - MI cue length was already adjustable (Record → Seconds/class)
  - Monitor band composition + waterfall blank: Config Window ms had been set to **1** (in `eeg.yaml` and the ErrP profile), so the spectrum came from a 1-sample window
    - Both files reset to 1000 ms; Config Apply / profile save and profile load now refuse Window < 250 ms or Step < 50 ms with a message
    - Note: Window/Step and the band-pass in Config only drive the Monitor (and Record display); ErrP uses its fixed epoch, MI Live the model's own window
  - New ErrP data: 4 sessions × 60 trials (240 epochs, 69 errors), all at chance (pooled 0.51)
    - Markers verified: one marker per outcome onset, 2.6–3.1 s apart; the trainer cuts the full −0.2…+epoch window from it (single-cell label is intentional)
    - Channels highly correlated (r 0.85–0.99 at 1–10 Hz); user's hardware checks (blink, press, jaw, alpha) all pass, so signals accepted as real
    - No error-vs-correct difference at any channel or time point out to 2 s (0 of 500 samples p<0.01 per channel)
  - ErrP epoch length now adjustable: Train tab shows **Epoch ms** for ErrP (default 2000, 500–3000; 800 = classic); bins stay ~83 ms (26 × 8 = 208 features at 2 s); stored on the model (`ErrPModel.tmax`), ErrP Live cuts the same window (old models: 0.8 s); Record keeps recording 3.2 s after the last outcome; facts gap ≥ epoch + 0.3 s
    - Same 240 epochs: 0.8 s 0.51 · 1.0 s 0.53 · 1.5 s 0.54 · 2.0 s 0.545 ± 0.04 — permutation test p = 0.13 (not above chance)
    - ErrP Live with a real 2 s model: verdicts 2.01–2.07 s after each outcome
  - 2 s ErrP epoch **reverted** at user request (fixed −0.2…+0.8 s again; Train tab back to Win/Step, ignored for ErrP)
  - ErrP task **replaced by the interaction (cursor) task** in Record and ErrP Live: target box left/right → subject presses ←/→ → cursor moves (opposite way at the error rate); both outcomes drawn identically (no red/green, no OK/WRONG); slips and no-press trials unlabelled and repeated; Max wait s (default 3) replaces Target s
    - Tested (simulated key presses): labels match the drawn movement in ErrP Live and Record; verdicts 0.81–0.86 s after the move
    - Keyboard focus moves to the task display each trial (focus_set, never focus_force)
  - **ErrP signal extraction — real-data comparison of published pipelines** (4 arrow + 3 cursor sessions, trial-grouped 5-fold CV)
    - Marker-free checks: residual after removing the shared signal has a 1/f EEG spectrum (signals are real); ~95% of 1–10 Hz power is common to all channels (ear-reference activity)
    - Arrow (watch) task: every method at chance (best xDAWN + LDA 0.58, p = 0.08) — no ErrP in observation data
    - Cursor (press) task: old bins + sLDA 0.55 → CAR 0.73 · spatial whitening (Farquhar & Hill 2013) 0.74 · xDAWN + bins (Rivet 2009) 0.74 · **xDAWN covariances + Riemannian tangent space + LR (BCI Challenge @ NER 2015 winner) 0.78, label-shuffle p = 0.01**; fronto-central subset and theta features did not help
    - Validity: error − correct difference peaks at Cz (− ~350 ms, + ~420 ms); unchanged without blink-sized epochs (0.77) and without FC5/FC6 (0.78 held-out); held-out sessions 0.73 / 0.79 / 0.81
  - **ErrP model replaced** by xDAWN (2 + 2 filters) + Ledoit–Wolf super-trial covariances + tangent space + balanced logistic regression (`core/eeg_models.ErrPModel`, `metrics.method = "xdawn+riemann"`); ErrP Live refuses old-pipeline models with a retrain message
  - **Arrow task restored** as an option next to the cursor task (Record: ErrP (cursor task) / ErrP (arrow task); ErrP Live: Cursor / Arrow / Facts), sharing one code path; outcome arrow now neutral colour, no OK/WRONG text
  - Clean-up: removed unused `ERDGate`, zero-phase `bandpass` (filtfilt — wrong for live use), `feature_names`, `node_names`, `is_ten_twenty`; Train tab hides Win/Step and the classifier for ErrP (fixed pipeline)
  - Tests on replayed real EEG (no simulated data): ErrP Live cursor + arrow — labels match drawn outcomes, live P(error) = offline to 1e-15
  - README, methodology PDF (revision 3) and speaker report updated
  - Stage 3 deck updated (now 15 slides; compiled locally with MiKTeX, installer disabled)
    - New Slide 9 after the demo: related work, Heim et al., *IEEE Access* 13:91232–91247 (2025) — title, hardware figure, results table, full citation (details checked on IEEE Xplore)
    - Slide 10 rewritten for the cursor task and xDAWN covariances + Riemannian tangent space + logistic regression (with equations); new Slide 11: method comparison on our data + validity checks
    - Slides 2, 6 and 7 updated to match; speaker report rewritten for Slides 9–11 and renumbered to 15 slides
    - Speaker report: new "Concepts explained simply" section (xDAWN, prototypes, covariance, Riemannian / tangent space, logistic regression, p-value) with analogies, linked from Slides 10–11
  - Branch check: cursor recordings (28 Oct 18:57–19:03) were made with the uncommitted working tree on `errp-live-timing`; `main` (`ef96edf`) has no cursor task, ErrP Live or stream clock — branch kept, not deleted
  - All work committed on `errp-live-timing`, `main` fast-forwarded to it and pushed to GitHub; the merged `errp-live-timing` branch deleted (local only, never pushed)

- **2026-10-29 — xDAWN visualisation** (no code changes)
  - Figures `_DOCS/errp_xdawn_single_trial.png` and `_DOCS/errp_xdawn_averages.png`: C3 / Cz / C4 without and with xDAWN (channels rebuilt from the 4 components), markers, baseline and training epoch, on the 3 real cursor sessions
  - xDAWN keeps ~20% of the epoch power; after it, error trials show a clear negativity ≈ 0.38 s (−1.0 to −1.3 µV) that correct trials don't, with much tighter error bands; filters fitted on the same trials, so the picture is in-sample (the 0.78 CV score is the honest measure)
  - Step-by-step figures `_DOCS/errp_step0_input.png`, `errp_step1_xdawn.png`, `errp_step23_riemann.png`, `errp_step4_logreg.png`: input, xDAWN recipes + signals, mean 12 × 12 covariances and their difference + tangent-space scatter, out-of-fold P(error) per trial
    - Cross-validated balanced accuracy after each step: raw channels 0.55 → after xDAWN 0.74 → + covariance / tangent space / logistic regression 0.78; out-of-fold 76% of errors caught, 80% of correct trials passed
  - **Stop recording** button on the EEG and EMG Record tabs (Monitor, Live and ErrP Live already had Start/Stop): the record button becomes "■ Stop recording" while a session runs; stopping asks Yes = save what was recorded, No = discard, Cancel = carry on; an ErrP task stops at once
    - Tested on replayed real EEG (EMG recorder fed 4 real EEG channels, as no EMG recordings exist): saved data identical to the streamed samples; discard saves nothing; Cancel keeps recording; ErrP stop after 5 trials saved exactly 5 markers
  - ErrP analysis-window test (real cursor data, 5 repeated grouped 5-fold CVs + held-out sessions; report only, not implemented): +0.4 s 0.75 · **+0.5 s 0.80 ± 0.02 (held-out 0.74 / 0.85 / 0.82; label-shuffle p = 0.01)** · +0.6 s 0.79 · +0.7 s 0.78 · +0.8 s (current) 0.78 · +1.0 s 0.78 — 0.5 s is as accurate and gives the verdict 0.3 s sooner; 0.4 s cuts off the dip
  - **ErrP window default changed to −0.2…+0.5 s** (was +0.8 s); no running-average wave added (user's choice)
    - Each model stores its epoch end (`ErrPModel.tmax`); ErrP Live and the facts test cut each model's own window, so older 0.8 s models still run correctly; outcome display minimum stays 0.8 s
    - Verified on real data: app training uses 175-sample epochs; ErrP Live on a replayed held-out session — 0.5 s model verdict 0.52–0.56 s after the move, 0.8 s model 0.83–0.87 s; live = offline (≈1e-15) for both
    - Same 180 trials, 5 repeated splits: raw channels 0.51 → after xDAWN 0.78 → full model 0.80
    - All ErrP figures re-rendered for the 0.5 s window; README, methodology PDF and speaker notes updated (method table on the results slide kept, labelled as the 0.8 s comparison)
  - Stage 3 deck: new slide 13 "What xDAWN Reveals: The Error Response" (C3 / Cz / C4 averages with and without xDAWN); deck now 20 slides
  - **Window reverted to −0.2…+0.8 s** at the user's request (they wanted the original averages figure); the per-model window (`ErrPModel.tmax`) is kept, so any 0.5 s model still runs correctly
    - Code, README, methodology PDF, all ErrP figures, deck text and speaker notes back on 0.8 s; slide 13 uses the averages figure as first shown (xDAWN fitted on 0.8 s epochs, dip at ≈0.38 s); step numbers back to 55% → 74% → 78% (the app's split, matching the results table)
    - Verified: training uses 250-sample epochs again, model.tmax 0.8
  - Live waveform viability (held-out session played in order): single-trial signal-to-noise ≈ 0.8, so one trial's wave is unreadable; a running error-vs-correct average shows the dip clearly after ~20 trials of each kind (2.6 standard errors) — figure `_DOCS/errp_live_wave_viability.png`
  - Stage 3 deck: four new slides (11–14), one step figure each with short plain-language bullets, between the algorithm and results slides; slide versions of the figures re-rendered with larger text on white (`slide_errp_step*.png`); deck now 19 slides, compiled and checked; speaker report given notes for Slides 11–14, renumbered, timing plan updated
  - Speaker report also produced as PDF (`mid_sem_speaker_report.pdf`, 12 pages, same styling as the methodology PDF); regenerate after editing the `.md` with `python _DOCS/source/build_speaker_report.py`

- **2026-11-09 — Two-stage MI model + public-data validation**
  - MI is now **two stages**, each trained and scored on its own: stage 1 rest vs MI (either hand), stage 2 left vs right (MI windows only); live decision = rest unless stage 1 says MI, then the likelier side; the planned EMG wrist clench will replace stage 1 later (stage 2 then runs alone)
    - Each stage: 2 s window's 8×8 covariance → Riemannian tangent space at the training mean → class-balanced logistic regression (Barachant et al. 2012); CSP + LDA still selectable
    - All MI classifiers weight classes equally (stage 1 sees twice as many MI windows as rest windows)
    - The old single 3-class one-vs-rest CSP model removed (no MI models had been saved)
    - Riemannian helpers shared with the ErrP model; ErrP result unchanged (0.777 on the cursor sessions)
  - Train tab: paradigm `MI (2-stage)`; new **features** picker (`riemann` default / `csp`); EEG defaults logreg, Win/Step 2000 / 500 ms (2 s beat 1 s on both stages); report shows stage 1, stage 2, then combined rest / left / right, each with a per-class table
  - Live tab: a two-stage model shows stage 1 (rest / MI) and stage 2 (left / right) bars and e.g. `→ mi_left  MI 72% · side 64%`
  - Record tab: MI **Reps** default 10 (≈ 3 min), up to 40 (was 6, max 15)
  - **Public-data validation** — new folder `_DOCS/mi_validation_physionet/` (short doc `README.md`, script, `results/`), tracked in git except the downloaded data
    - PhysioNet EEG Motor Movement/Imagery, subjects S001–S005, imagined left/right fist + rest (runs 4, 8, 12; 15 EDF files, 37 MB, downloaded with the user's permission), only our 8 electrodes, scored through the app's own `train_mi`
    - Mean of 5 subjects: stage 1 **0.74**, stage 2 **0.69**, rest / left / right 0.58 (old single 3-class model 0.55); trained on 2 runs and tested on the unseen 3rd: 0.71 / 0.69
    - Best subjects: 0.91 (stage 1, S004) and 0.83 (stage 2, S001); 85% is reached by individuals, not on average; S003 at chance for left/right
    - Left/right accuracy follows how lateralised a person's ERD is (C3 vs C4 power drop), almost in rank order
    - 1 s windows 0.68 / 0.64, 3 s 0.77 / 0.70; all 64 electrodes 0.82 / 0.62 (overfits left/right with ~22 trials per hand); C3 Cz C4 only 0.63 / 0.55; our 8-electrode montage is a good choice
  - Tested on real data through the GUI code: PhysioNet runs saved in the app's recording format, trained in the Train tab, model started from the Live tab and a held-out run replayed — live = offline exactly, held-out gate 0.71 / side 0.79

- **2026-11-10 — Model naming (EEG and EMG)**
  - Models are no longer named after the selected recordings: the Train tab has **Subject**, **Name** and **+ scores** fields, and the model saves as `<subject>_<name>_<scores>`; a live *Saves as:* line shows the name before saving
    - Subject is pre-filled from the selected recordings (their shared start when they differ, e.g. `amogh_errp_trial`); the user's own edit is kept when the selection changes; the subject is also written into the model's JSON
    - Scores: `gate0.74_side0.69` for two-stage MI, `bal0.78` for other EEG models, `acc0.91` for EMG; untick **+ scores** to leave them out
    - Letters, digits, `.`, `_`, `-` are kept (case too); other characters become `-`; saving over an existing name asks before replacing it; an empty subject and name is refused
  - The Live, ErrP Live and EMG Live lists now find models by the modality in each model's JSON instead of the `eeg_` / `emg_` folder prefix, so old automatically named models still appear
  - Tested on the real cursor recordings, copied into a scratch folder: pre-filled subject, preview, train, save, load, replace (No keeps / Yes replaces), scores off, empty name refused; the new and an old-style model both listed in Live and ErrP Live, none in EMG
  - Follow-up (user request): **Subject is now typed in by hand and required** (marked `Subject*`, no pre-fill from the recordings); **Save model** refuses an empty subject even when a Name is given
  - **Old automatic naming removed everywhere** (`ModelMeta.folder_name` / `accuracy` deleted; `save_model` needs a name); the user will retrain the models
    - Old model folders under `biopipe/models/` are still on disk and still listed (found by the modality in their JSON) until deleted
    - Recording folders keep their own `<modality>_<subject>_<stamp>` names
    - Tested on the real cursor recordings: subject empty after selecting, save refused without it (even with a Name), `Amogh_cursor_bal0.75` saved and listed with subject `Amogh` in its JSON
  - All work since 27a35c3 (stop recording, per-model ErrP window, two-stage MI, MI validation folder, model naming) committed as `0fdd858` on `main` and pushed to GitHub; the deleted roadmap PDF, the `4channel emg acquisition` submodule change and the root `eeg_config.yaml` left out

- **2026-11-11 — Recording ratings in the Train tab**
  - The recordings list is now a table (Recording · Labels · **Rating (1-5)**) with a scrollbar; clicking a rating cell drops down 1–5 or *not rated*, to record how much each dataset helps model accuracy
    - Clicking a rating leaves the selection alone, so Ctrl/Shift-click pools stay intact; pooled recordings are still used in list order
    - Saved as `rating` in the recording's JSON (0 = not rated; every other field kept); also shown in the *Selected recording* panel
    - The rating is a manual note only; training does not use it
  - Tested on real ErrP recordings copied into a scratch folder: rating saved and still there after refresh, JSON otherwise unchanged, dropdown choices set and clear the rating, pooling check and training on a table selection work; the on-screen click → dropdown wasn't exercised, so that no test window could take keyboard focus

- **2026-11-11 — Datasets and models in git**
  - Legacy-named models: none left (the four old `eeg_…_acc…_<stamp>` folders were already gone); `biopipe/models/` holds only new-style names (`Amogh_cursor_bal0.84`, `amogh_4datasets_bal0.80`, `amogh_5_sets_bal0.81`)
  - `biopipe/data/` and `biopipe/models/` removed from `biopipe/.gitignore` at the user's request; the 14 recordings (13 ErrP, 1 MI; ~60 MB) and 3 models committed and pushed to the public GitHub repo
  - The PhysioNet EDFs (`_DOCS/mi_validation_physionet/data/`) are still ignored: a public dataset that can be downloaded again

## Current state

- **EEG tab subtabs:** Config · Monitor · Record · Live · ErrP Live (arrow feedback or true/false facts test)
- **EEG models:** two-stage MI (rest vs MI, then left vs right; Riemannian tangent space or CSP + pluggable classifier) and ErrP (xDAWN + Riemannian tangent space + logistic regression)
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
- Record our own MI sessions (≥ 40 trials per hand over several sessions, kinesthetic imagery) and check the C3 / C4 ERD lateralisation before tuning; ≥ 200 ErrP trials
- Raise ErrP from 0.78 towards 85%: more cursor runs; then build the MI-driven ErrP task and the MI / ErrP reinforcement-learning loop
- Methodology PDF still describes MI as CSP-based: update it for the two-stage model
- Deferred: EMG-gated control loop; confidence threshold and smoothing for drone commands
- Housekeeping: stale root `eeg_config.yaml` (old pipeline) left untracked
- Before the stage 3 review: fix or defend the flagged slide claims; run the demo from `errp-live-timing` with a retrained ErrP model and a 1–45 Hz Monitor band-pass
- Retrain the ErrP model on the cursor sessions (old-pipeline models are refused); record more cursor runs (≥ 200 trials)
- Store per-event target / pressed / moved direction in recordings, to test the eye-movement confound directly
- Facts test: the arrow-trained ErrP model may not transfer (false statements classically evoke N400/P600 rather than a feedback ErrP, at an unlocked time); consider recording facts sessions to train on
