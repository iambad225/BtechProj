"""Validate biopipe's MI pipeline on public motor-imagery EEG (PhysioNet EEGMMIDB).

Data: Schalk et al. 2004 / Goldberger et al. 2000, https://physionet.org/content/eegmmidb/1.0.0/
(ODC-By 1.0). Runs 4, 8 and 12 = imagined left fist (T1) / right fist (T2) with rest (T0), 64 ch,
160 Hz. Only our 8 headset sites are used (Fz FC5 FC6 C3 Cz C4 P3 P4) unless stated.

Everything is scored through the app's own code (core.eeg_training.train_mi = the Train tab):
causal 8-30 Hz filter, 2 s windows / 0.5 s step after a 0.4 s transient, trial-grouped stratified
5-fold CV, balanced accuracy. Run from anywhere:

    python -I validate_mi_physionet.py [data_dir] [results_dir]     (~5 min)
    python -I validate_mi_physionet.py --figures                     (redraw figures only)
"""
from __future__ import annotations

import sys, json, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "biopipe"))

from scipy.stats import binom
from sklearn.metrics import balanced_accuracy_score as BA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from core.config import Config
from core.csp import CSP
from core.eeg_models import TwoStageMI, causal_bandpass
from core import eeg_training as et

SUBJECTS = ["S001", "S002", "S003", "S004", "S005"]
RUNS = ["R04", "R08", "R12"]
OURS = ["Fz", "FC5", "FC6", "C3", "Cz", "C4", "P3", "P4"]
EVENTS = {"T0": "rest", "T1": "mi_left", "T2": "mi_right"}


# ---------------- minimal EDF+ reader ----------------
def read_edf(path):
    """(signals {LABEL: uV array}, annotations [(onset_s, dur_s, text)], fs)."""
    with open(path, "rb") as f:
        h = f.read(256)
        n_hdr, dur, ns = int(h[184:192]), float(h[244:252]), int(h[252:256])
        sh, pos = f.read(ns * 256), 0

        def field(w):
            nonlocal pos
            out = [sh[pos + i * w: pos + (i + 1) * w].decode("ascii", "replace").strip()
                   for i in range(ns)]
            pos += w * ns
            return out
        labels = field(16); field(80); field(8)
        pmin, pmax = np.array(field(8), float), np.array(field(8), float)
        dmin, dmax = np.array(field(8), float), np.array(field(8), float)
        field(80); nsamp = np.array(field(8), int); field(32)
        f.seek(n_hdr)
        raw = np.fromfile(f, dtype="<i2")
    rec = int(nsamp.sum())
    raw = raw[: (raw.size // rec) * rec].reshape(-1, rec)
    sig, ann, off = {}, [], 0
    for i, lab in enumerate(labels):
        blk = raw[:, off:off + nsamp[i]]
        off += nsamp[i]
        if lab == "EDF Annotations":
            for tal in blk.tobytes().decode("latin-1").split("\x00"):
                parts = tal.split("\x14")
                onset, _, d = parts[0].partition("\x15")
                ann += [(float(onset), float(d or 0), t) for t in parts[1:] if t and onset]
        else:
            d = blk.reshape(-1).astype(float)
            sig[lab.rstrip(".").upper()] = ((d - dmin[i]) * (pmax[i] - pmin[i])
                                            / (dmax[i] - dmin[i]) + pmin[i])
    fs = nsamp[[i for i, l in enumerate(labels) if l != "EDF Annotations"][0]] / dur
    return sig, ann, fs


def load_run(path, elec):
    """(signal (n, len(elec)) in uV, per-sample labels rest/mi_left/mi_right, fs)."""
    sig, ann, fs = read_edf(path)
    X = np.column_stack([sig[e.upper()] for e in elec])
    lab = np.array(["rest"] * len(X), dtype=object)
    for on, d, t in ann:
        if t in EVENTS:
            lab[int(round(on * fs)):int(round((on + d) * fs))] = EVENTS[t]
    return X, list(lab), fs


def cfg(fs, win_ms=2000, step_ms=500, clf="logreg", feats="riemann"):
    c = Config(modality="eeg", sample_rate=int(fs), labels=["mi_left", "mi_right"],
               window_ms=win_ms, step_ms=step_ms, transient_ms=400)
    c.model_type, c.mi_features = clf, feats
    return c


class OldThreeClass:
    """The app's MI model before 2026-11-09: one 3-class model (one-vs-rest CSP + LDA)."""
    def fit(self, X, y):
        self.csps = [CSP(4).fit(X, (y == c).astype(int)) for c in range(3)]
        self.sc, self.clf = StandardScaler(), LDA()
        self.clf.fit(self.sc.fit_transform(self._f(X)), y)
        return self

    def _f(self, X):
        return np.concatenate([c.transform(X) for c in self.csps], axis=1)

    def predict_proba(self, X):
        return self.clf.predict_proba(self.sc.transform(self._f(X)))


def stage_scores(metrics):
    st = metrics["stages"]
    return {"gate": st["gate"]["balanced_accuracy"], "side": st["side"]["balanced_accuracy"],
            "overall": metrics["balanced_accuracy"],
            "gate_std": st["gate"]["balanced_accuracy_std"],
            "side_std": st["side"]["balanced_accuracy_std"],
            "per_class": {"gate": st["gate"]["per_class"], "side": st["side"]["per_class"],
                          "overall": metrics["per_class"]}}


def chance_threshold(n, p=0.05):
    """Smallest accuracy beating chance (0.5) at level p with n independent trials."""
    return float(binom.ppf(1 - p, n, 0.5) / n)


def leave_one_run_out(runs, fs):
    """Train on two runs, test on the unseen third (offline = live, verified sample-exact)."""
    out = []
    for k in range(len(runs)):
        train = [r for i, r in enumerate(runs) if i != k]
        ep, y, _, classes = et.collect_mi(train, cfg(fs))
        m = TwoStageMI(classes).fit(ep, y)
        te, yt, _, _ = et.collect_mi([runs[k]], cfg(fs))
        g, r = m.stages(te)
        mi = yt > 0
        out.append({"gate": BA(mi, g > 0.5), "side": BA(yt[mi] - 1, r[mi] > 0.5),
                    "overall": BA(yt, m.predict(te))})
    return {k: round(float(np.mean([o[k] for o in out])), 3) for k in out[0]}


def erd_curves(runs, fs, chans=("C3", "C4"), tmin=-1.0, tmax=4.5):
    """Trial-averaged 8-30 Hz power relative to the 1 s of rest before each cue (ERD %)."""
    idx = [OURS.index(c) for c in chans]
    a, b = int(round(tmin * fs)), int(round(tmax * fs))
    k = np.ones(int(0.5 * fs)) / int(0.5 * fs)
    acc = {"mi_left": [], "mi_right": []}
    for X, lab, _ in runs:
        P = causal_bandpass(X[:, idx], fs, 8.0, 30.0) ** 2
        P = np.column_stack([np.convolve(P[:, j], k, mode="same") for j in range(P.shape[1])])
        for i in range(1, len(lab)):
            if lab[i] in acc and lab[i - 1] != lab[i] and i + a >= 0 and i + b <= len(lab):
                acc[lab[i]].append(P[i + a:i + b])
    t = np.arange(a, b) / fs
    out = {}
    for c, eps in acc.items():
        m = np.mean(eps, axis=0)                               # average power over trials
        base = m[t < 0].mean(axis=0)
        out[c] = 100.0 * (m / base - 1.0)                      # ERD (-) / ERS (+) in %
    return t, out


def main():
    data = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "data"
    res_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    allch = sorted(read_edf(data / "S001R04.edf")[0])
    montages = {"ours_8": OURS, "all_64": allch, "C3_Cz_C4": ["C3", "Cz", "C4"],
                "alt_8_CP5_CP6_Pz": ["Fz", "FC5", "FC6", "C3", "C4", "CP5", "CP6", "Pz"]}
    R = {"subjects": {}, "setup": {
        "dataset": "PhysioNet EEGMMIDB 1.0.0, runs R04/R08/R12 (imagined left/right fist + rest)",
        "electrodes": OURS, "band_hz": [8, 30], "filter": "causal Butterworth (as live)",
        "window_s": 2.0, "step_s": 0.5, "transient_s": 0.4,
        "cv": "trial-grouped stratified 5-fold (app's _grouped_cv)",
        "default_model": "TwoStageMI riemann+logreg"}}
    t0 = time.time()
    for s in SUBJECTS:
        runs = {name: [load_run(data / f"{s}{r}.edf", el) for r in RUNS]
                for name, el in montages.items()}
        fs = runs["ours_8"][0][2]
        ours = [(X, lab) for X, lab, _ in runs["ours_8"]]
        r = R["subjects"][s] = {}
        _, m = et.train_mi(ours, cfg(fs))
        r["two_stage_riemann"] = stage_scores(m)
        ep, y, g, classes = et.collect_mi(ours, cfg(fs))
        r["n_trials"] = {c: int(len(np.unique(g[y == i]))) for i, c in enumerate(classes)}
        r["two_stage_csp_lda"] = stage_scores(et.train_mi(ours, cfg(fs, clf="lda", feats="csp"))[1])
        cv = et._grouped_cv(OldThreeClass, ep, y, g, 3)
        r["old_single_3class"] = {"overall": round(float(np.mean(cv[2])), 3)}
        for w, st in ((1000, 250), (3000, 500)):
            r[f"window_{w}ms"] = stage_scores(et.train_mi(ours, cfg(fs, w, st))[1])
        for name in ("all_64", "C3_Cz_C4", "alt_8_CP5_CP6_Pz"):
            mm = [(X, lab) for X, lab, _ in runs[name]]
            r[f"montage_{name}"] = stage_scores(et.train_mi(mm, cfg(fs))[1])
        r["leave_one_run_out"] = leave_one_run_out(ours, fs)
        t, erd = erd_curves(runs["ours_8"], fs)
        e = r["erd_mean_0p5_4s"] = {c: {ch: round(float(v[(t >= 0.5) & (t <= 4.0), j].mean()), 1)
                                        for j, ch in enumerate(("C3", "C4"))} for c, v in erd.items()}
        # contralateral index: (C3 - C4) in right-hand MI minus (C3 - C4) in left-hand MI.
        # Negative = textbook pattern (right hand lowers C3 power more, left hand lowers C4 more).
        r["lateralisation_index"] = round((e["mi_right"]["C3"] - e["mi_right"]["C4"])
                                          - (e["mi_left"]["C3"] - e["mi_left"]["C4"]), 1)
        r["_erd_curves"] = {c: v.tolist() for c, v in erd.items()}
        R["_erd_t"] = t.tolist()
        nt = r["n_trials"]
        r["chance_p05"] = {"gate": round(chance_threshold(nt["rest"] + nt["mi_left"] + nt["mi_right"]), 3),
                           "side": round(chance_threshold(nt["mi_left"] + nt["mi_right"]), 3)}
        print(f"{s}  gate {r['two_stage_riemann']['gate']}  side {r['two_stage_riemann']['side']}  "
              f"overall {r['two_stage_riemann']['overall']}  LORO {r['leave_one_run_out']}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    # ---- summary: mean over subjects ----
    def mean_of(key, field):
        vals = [R["subjects"][s][key].get(field) for s in SUBJECTS]
        return None if any(v is None for v in vals) else round(float(np.mean(vals)), 3)
    keys = ["two_stage_riemann", "two_stage_csp_lda", "window_1000ms", "window_3000ms",
            "montage_all_64", "montage_C3_Cz_C4", "montage_alt_8_CP5_CP6_Pz", "leave_one_run_out"]
    R["summary"] = {k: {f: mean_of(k, f) for f in ("gate", "side", "overall")} for k in keys}
    R["summary"]["old_single_3class"] = {"overall": mean_of("old_single_3class", "overall")}
    R["summary"]["n_subjects_reaching_85"] = {
        f: sum(R["subjects"][s]["two_stage_riemann"][f] >= 0.85 for s in SUBJECTS)
        for f in ("gate", "side")}
    curves = {s: R["subjects"][s].pop("_erd_curves") for s in SUBJECTS}
    t = np.array(R.pop("_erd_t"))
    (res_dir / "results.json").write_text(json.dumps(R, indent=1), encoding="utf-8")
    np.savez(res_dir / "erd_curves.npz", t=t, **{f"{s}_{c}": np.asarray(v)
                                                for s in curves for c, v in curves[s].items()})
    print(json.dumps(R["summary"], indent=1))
    make_figures(R, curves, t, res_dir)


# ---------------- figures (reference palette: blue / orange; ink + hairline greys) ----------------
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def make_figures(R, curves, t, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Segoe UI", "text.color": INK, "axes.labelcolor": INK2})
    S = R["subjects"]

    # 1) ERD at C3 / C4 for a lateralised subject and a non-lateralised one
    lat = {s: S[s]["lateralisation_index"] for s in S}
    pair = [min(lat, key=lat.get), max(lat, key=lat.get)]          # most / least contralateral
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True, sharey=True, facecolor="white")
    for i, s in enumerate(pair):
        for j, ch in enumerate(("C3 (left hemisphere)", "C4 (right hemisphere)")):
            ax = axes[i, j]
            _style(ax)
            ax.axvspan(0, 4.1, color="#f0efec", zorder=0)
            ax.axhline(0, color=AXIS, linewidth=1)
            for cls, col, name in (("mi_left", BLUE, "left-hand imagery"),
                                   ("mi_right", ORANGE, "right-hand imagery")):
                ax.plot(t, np.asarray(curves[s][cls])[:, j], color=col, linewidth=2, label=name)
            ax.set_title(f"{s}  ·  {ch}", fontsize=10, color=INK, loc="left")
        axes[i, 0].set_ylabel("8-30 Hz power vs rest (%)")
        axes[i, 1].text(1.0, 1.08, f"left/right accuracy {S[s]['two_stage_riemann']['side']:.2f}"
                        f"  ·  index {lat[s]:+.1f}", transform=axes[i, 1].transAxes,
                        ha="right", fontsize=9, color=INK2)
    for ax in axes[1]:
        ax.set_xlabel("time from cue (s)  ·  shaded = imagery")
    axes[1, 0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Left/right works when imagery lowers power on the opposite side",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out / "mi_erd_examples.png", dpi=150)
    plt.close(fig)

    # 2) lateralisation index vs left/right accuracy
    fig, ax = plt.subplots(figsize=(5.6, 3.8), facecolor="white")
    _style(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    xs = [lat[s] for s in S]; ys = [S[s]["two_stage_riemann"]["side"] for s in S]
    ax.axhline(0.5, color=MUTED, linewidth=1, linestyle=":")
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.scatter(xs, ys, s=70, color=ORANGE, edgecolors="white", linewidths=2, zorder=3)
    for s, x, y in zip(S, xs, ys):
        ax.annotate(s, (x, y), xytext=(7, 4), textcoords="offset points", fontsize=8, color=INK2)
    ax.text(0.01, 0.5, "chance", transform=ax.get_yaxis_transform(), ha="left", va="bottom",
            fontsize=8, color=MUTED)
    ax.set_xlabel("lateralisation index (negative = contralateral ERD)")
    ax.set_ylabel("left vs right accuracy")
    ax.set_ylim(0.4, 0.9)
    ax.set_title("Stage 2 accuracy follows lateralisation", fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out / "mi_lateralisation_vs_accuracy.png", dpi=150)
    plt.close(fig)

    # 3) per-subject accuracy of each stage
    subs = list(S)
    gate = [S[s]["two_stage_riemann"]["gate"] for s in subs] + [R["summary"]["two_stage_riemann"]["gate"]]
    side = [S[s]["two_stage_riemann"]["side"] for s in subs] + [R["summary"]["two_stage_riemann"]["side"]]
    names = subs + ["mean"]
    x = np.arange(len(names)); w = 0.36
    fig, ax = plt.subplots(figsize=(8.4, 3.8), facecolor="white")
    _style(ax)
    ax.bar(x - w / 2 - 0.01, gate, w, color=BLUE, label="stage 1: rest vs MI")
    ax.bar(x + w / 2 + 0.01, side, w, color=ORANGE, label="stage 2: left vs right")
    for y, txt, col, ls in ((0.85, "85% target", INK2, "--"), (0.5, "chance", MUTED, ":")):
        ax.axhline(y, color=col, linewidth=1, linestyle=ls)
        ax.text(1.01, y, txt, transform=ax.get_yaxis_transform(), color=col, fontsize=8,
                ha="left", va="center")
    for xi, v in ((x[-1] - w / 2, gate[-1]), (x[-1] + w / 2, side[-1])):
        ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=9, color=INK)
    ax.axvline(len(names) - 1.5, color=GRID, linewidth=1)
    ax.set_xticks(x, names)
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("balanced accuracy (5-fold CV)")
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    ax.set_title("Two-stage MI on public data, our 8 electrodes", fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out / "mi_stage_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    if "--figures" in sys.argv:          # redraw from saved results without recomputing
        out = HERE / "results"
        R = json.loads((out / "results.json").read_text(encoding="utf-8"))
        z = np.load(out / "erd_curves.npz")
        make_figures(R, {s: {c: z[f"{s}_{c}"] for c in ("mi_left", "mi_right")} for s in R["subjects"]},
                     z["t"], out)
    else:
        main()
