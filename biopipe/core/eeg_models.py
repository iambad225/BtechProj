"""EEG paradigm models: MI (two-stage: rest vs MI, then left vs right) and ErrP (xDAWN +
Riemannian tangent space).

Both are small, CPU-fast classical pipelines. They consume *epochs* (lists of
(n_times, n_channels) arrays) that the training code has already band-passed for the
paradigm's band. Each saves/loads via joblib.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import joblib
from scipy.linalg import eigh
from scipy.signal import butter, sosfilt, sosfilt_zi
from sklearn.covariance import ledoit_wolf
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from core.csp import CSP

# MI features: "riemann" = window covariance in the Riemannian tangent space (default; best on
# the PhysioNet check, see _DOCS/mi_validation_physionet) or "csp" = CSP log-variances.
MI_FEATURES = ("riemann", "csp")
# Classifier choices for the MI model (all expose predict_proba). logreg is the default with
# riemann features; lda is the CSP workhorse; rf/tree usually overfit these features.
MI_CLASSIFIERS = ("logreg", "lda", "svm", "rf", "tree")


def make_classifier(name: str, n_classes: int = 2):
    """Classes are weighted equally: the rest-vs-MI stage sees twice as many MI windows (left +
    right) as rest windows, and an unweighted model would lean towards MI."""
    name = (name or "logreg").lower()
    if name == "lda":
        return LDA(priors=np.full(n_classes, 1.0 / n_classes))
    if name == "svm":
        return SVC(kernel="linear", probability=True, class_weight="balanced")
    if name == "logreg":
        return LogisticRegression(max_iter=2000, class_weight="balanced")
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1,
                                      class_weight="balanced")
    if name == "tree":
        return DecisionTreeClassifier(random_state=0, class_weight="balanced")
    raise ValueError(f"Unknown MI classifier: {name!r}")


# ---------------- shared signal helpers ----------------
class StreamBandpass:
    """Causal (forward-only) Butterworth band-pass for a live stream, fed chunk by chunk.

    Training runs the very same filter over each whole recording (``causal_bandpass``), so a live
    window/epoch is processed exactly like its training counterpart and nothing waits on future
    samples. The state starts in steady state for the first sample, so the Cyton's large DC
    offset causes no start-up transient.
    """

    def __init__(self, fs: float, lo: float, hi: float, n_ch: int, order: int = 4):
        nyq = fs / 2.0
        self.sos = butter(order, [lo / nyq, min(hi, nyq * 0.99) / nyq], btype="band", output="sos")
        self.n_ch = n_ch
        self.zi = None

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.shape[0] == 0:
            return np.zeros((0, self.n_ch))
        if self.zi is None:
            self.zi = sosfilt_zi(self.sos)[:, :, None] * x[0][None, None, :]
        y, self.zi = sosfilt(self.sos, x, axis=0, zi=self.zi)
        return y


def causal_bandpass(sig: np.ndarray, fs: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    """Causal band-pass of a whole recording - sample-for-sample what StreamBandpass gives live."""
    sig = np.asarray(sig, dtype=float)
    return StreamBandpass(fs, lo, hi, sig.shape[1] if sig.ndim == 2 else 1, order).process(sig)


def epoch_signal(sig: np.ndarray, fs: float, onsets, tmin: float, tmax: float,
                 baseline=(-0.2, 0.0)) -> list[np.ndarray]:
    """Cut epochs [tmin, tmax] s around each onset sample; baseline-correct if given."""
    a, b = int(round(tmin * fs)), int(round(tmax * fs))
    out = []
    for o in onsets:
        s, e = o + a, o + b
        if s < 0 or e > sig.shape[0]:
            continue
        ep = sig[s:e].astype(float)
        if baseline is not None:
            bs = int(round((baseline[0] - tmin) * fs))
            be = int(round((baseline[1] - tmin) * fs))
            if be > bs:
                ep = ep - ep[bs:be].mean(axis=0, keepdims=True)
        out.append(ep)
    return out


# ---------------- Riemannian helpers (MI and ErrP) ----------------
def inv_sqrt(C: np.ndarray) -> np.ndarray:
    d, V = eigh(C)
    return (V / np.sqrt(np.maximum(d, 1e-12))) @ V.T


def shrunk_covs(X) -> np.ndarray:
    """Ledoit-Wolf covariance of each (n_times, n_channels) window."""
    return np.array([ledoit_wolf(e)[0] for e in X])


def tangent_space(covs: np.ndarray, iref: np.ndarray) -> np.ndarray:
    """Covariances -> tangent vectors at the reference R (``iref`` = R^-1/2): the upper triangle
    of log(R^-1/2 C R^-1/2), off-diagonal terms scaled by sqrt(2)."""
    n = covs.shape[1]
    iu = np.triu_indices(n)
    w = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    out = []
    for C in covs:
        d, V = eigh(iref @ C @ iref)
        out.append(((V * np.log(np.maximum(d, 1e-12))) @ V.T)[iu] * w)
    return np.asarray(out)


# ---------------- MI ----------------
class MIModel:
    """One binary MI decoder: band-passed window -> spatial features -> classifier.

    ``riemann`` features: the window's covariance (8x8) mapped to the tangent space at the
    training mean (36 numbers; Barachant et al. 2012). It holds each channel's mu/beta power and
    how the channels co-vary, so it sees the contralateral ERD without hand-made features.
    ``csp`` features: log-variance of 4 CSP-filtered signals.
    """
    kind = "mi"

    def __init__(self, classes, clf: str = "logreg", features: str = "riemann", n_csp: int = 4):
        self.classes = list(classes)              # two classes, e.g. ["mi_left", "mi_right"]
        self.clf_name = (clf or "logreg").lower()
        self.features = (features or "riemann").lower()
        if self.features not in MI_FEATURES:
            raise ValueError(f"Unknown MI features: {features!r}")
        self.n_csp = n_csp
        # CSP log-variances need standardising; tangent vectors are already centred and whitened
        # by the reference (scaling them cost ~3 points on the PhysioNet check)
        self.scaler = StandardScaler() if self.features == "csp" else None
        self.clf = make_classifier(self.clf_name, len(self.classes))

    def fit(self, epochs, y):
        y = np.asarray(y)
        if self.features == "csp":
            self.csp_ = CSP(self.n_csp).fit(epochs, y)
        else:
            self.iref_ = inv_sqrt(shrunk_covs(epochs).mean(axis=0))
        F = self._transform(epochs)
        self.clf.fit(self.scaler.fit_transform(F) if self.scaler else F, y)
        return self

    def _transform(self, epochs):
        if self.features == "csp":
            return self.csp_.transform(epochs)
        return tangent_space(shrunk_covs(epochs), self.iref_)

    def predict_proba(self, epochs):
        F = self._transform(epochs)
        return self.clf.predict_proba(self.scaler.transform(F) if self.scaler else F)

    @staticmethod
    def decide(proba):
        """Class index per row of predict_proba."""
        return np.argmax(np.atleast_2d(proba), axis=1)

    def predict(self, epochs):
        return self.decide(self.predict_proba(epochs))

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)


class TwoStageMI(MIModel):
    """Stage 1 (gate): rest vs MI with either hand. Stage 2 (side): left vs right, trained on MI
    windows only. Decision: rest unless the gate says MI, then the side.

    Each stage is a binary MIModel and is scored on its own. The EMG wrist clench is meant to
    replace the gate later; the side stage then runs alone.
    """

    def __init__(self, classes, clf: str = "logreg", features: str = "riemann"):
        self.classes = list(classes)              # ["rest", "mi_left", "mi_right"]
        self.clf_name, self.features = clf, features
        self.gate = MIModel(["rest", "mi"], clf, features)
        self.side = MIModel(self.classes[1:], clf, features)

    def fit(self, epochs, y):
        y = np.asarray(y)
        self.gate.fit(epochs, (y > 0).astype(int))
        mi = np.flatnonzero(y > 0)
        self.side.fit([epochs[i] for i in mi], y[mi] - 1)
        return self

    def stages(self, epochs):
        """(P(MI), P(right | MI)) per window."""
        return self.gate.predict_proba(epochs)[:, 1], self.side.predict_proba(epochs)[:, 1]

    def predict_proba(self, epochs):
        """[P(rest), P(MI) P(left | MI), P(MI) P(right | MI)] per window."""
        g, r = self.stages(epochs)
        return np.column_stack([1.0 - g, g * (1.0 - r), g * r])

    @staticmethod
    def decide(proba):
        """Rest when P(rest) > 0.5 (the gate is closed), otherwise the likelier side."""
        p = np.atleast_2d(proba)
        return np.where(p[:, 0] > 0.5, 0, 1 + np.argmax(p[:, 1:], axis=1))


class ErrPModel:
    """xDAWN spatial filters + Riemannian tangent space + logistic regression.

    The approach of the BCI Challenge @ NER 2015 winners (Barachant et al.) for feedback-error
    detection. On this headset ~95% of the 1-10 Hz power is common to all channels (the ear
    reference's own activity), which buries the error response; xDAWN (Rivet et al. 2009) learns
    spatial filters that pull out the class-specific evoked response. Each epoch is stacked under
    the two class prototypes (filtered averages), its shrunk covariance is mapped to the tangent
    space at the training mean, and a balanced logistic regression gives P(error).
    """
    kind = "errp"
    method = "xdawn+riemann"
    tmax = 0.8                                     # epoch end of models saved before it was stored

    def __init__(self, n_filters: int = 2, tmax: float = 0.8):
        self.n_filters = n_filters                 # xDAWN filters per class
        self.tmax = tmax                           # epoch end (s) it was trained on; live cuts the same
        self.clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)

    def _xdawn(self, X, y):
        C = np.mean([e.T @ e for e in X], axis=0)
        C = C + 1e-9 * np.trace(C) * np.eye(len(C))
        W = []
        for c in (0, 1):                           # maximise evoked / total power, per class
            A = X[y == c].mean(axis=0)
            d, V = eigh(A.T @ A, C)
            W.append(V[:, np.argsort(d)[::-1][:self.n_filters]])
        return np.hstack(W)

    def _covs(self, X):
        return np.array([ledoit_wolf(np.vstack([self.P_, (e @ self.W_).T]).T)[0] for e in X])

    def _tangent(self, covs):
        return tangent_space(covs, self.iref_)

    def fit(self, epochs, y):                      # y: 0 correct, 1 error
        X, y = np.asarray(epochs, dtype=float), np.asarray(y)
        self.W_ = self._xdawn(X, y)
        self.P_ = np.vstack([(X[y == c].mean(axis=0) @ self.W_).T for c in (0, 1)])
        covs = self._covs(X)
        self.iref_ = inv_sqrt(covs.mean(axis=0))
        self.clf.fit(self._tangent(covs), y)
        return self

    def proba_error(self, epochs):
        return self.predict_proba(epochs)[:, 1]

    def predict_proba(self, epochs):
        return self.clf.predict_proba(self._tangent(self._covs(np.asarray(epochs, dtype=float))))

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
