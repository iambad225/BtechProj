"""EEG paradigm models: MI (CSP + pluggable classifier) and ErrP (xDAWN + Riemannian tangent space).

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

# Classifier choices for the MI model (all expose predict_proba). LDA is the default
# MI-literature workhorse; svm/logreg are linear; rf/tree are the tree-based options
# (a single tree is offered for comparison but usually overfits CSP features).
MI_CLASSIFIERS = ("lda", "svm", "logreg", "rf", "tree")


def make_classifier(name: str):
    name = (name or "lda").lower()
    if name == "lda":
        return LDA()
    if name == "svm":
        return SVC(kernel="linear", probability=True)
    if name == "logreg":
        return LogisticRegression(max_iter=2000)
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
    if name == "tree":
        return DecisionTreeClassifier(random_state=0)
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


# ---------------- MI: CSP + LDA (2-class) ----------------
class MIModel:
    """CSP + a pluggable classifier.

    2 classes use one CSP; >2 classes use one-vs-rest CSP (a CSP per class vs the rest, log-var
    features concatenated). This lets one model do ``{rest, mi_left, mi_right}`` so the ERD/ERS
    gate is folded in as the ``rest`` class rather than a separate model.
    """
    kind = "mi"

    def __init__(self, classes, clf: str = "lda", n_csp: int = 4):
        self.classes = list(classes)              # e.g. ["rest", "mi_left", "mi_right"]
        self.clf_name = (clf or "lda").lower()
        self.n_csp = n_csp
        self.scaler = StandardScaler()
        self.clf = make_classifier(self.clf_name)
        self._binary = None
        self._csp = None                          # single CSP (2-class)
        self.csps = None                          # one CSP per class (one-vs-rest, >2 classes)

    def fit(self, epochs, y):
        y = np.asarray(y)
        self._binary = (len(self.classes) == 2)
        if self._binary:
            self._csp = CSP(self.n_csp).fit(epochs, y)
        else:
            self.csps = [CSP(self.n_csp).fit(epochs, (y == c).astype(int))
                         for c in range(len(self.classes))]
        self.clf.fit(self.scaler.fit_transform(self._transform(epochs)), y)
        return self

    def _transform(self, epochs):
        if self._binary:
            return self._csp.transform(epochs)
        return np.concatenate([csp.transform(epochs) for csp in self.csps], axis=1)

    def predict_proba(self, epochs):
        return self.clf.predict_proba(self.scaler.transform(self._transform(epochs)))

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)


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

    def __init__(self, n_filters: int = 2):
        self.n_filters = n_filters                 # xDAWN filters per class
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
        n = covs.shape[1]
        iu = np.triu_indices(n)
        w = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
        out = []
        for C in covs:
            d, V = eigh(self.iref_ @ C @ self.iref_)
            out.append(((V * np.log(np.maximum(d, 1e-12))) @ V.T)[iu] * w)
        return np.asarray(out)

    def fit(self, epochs, y):                      # y: 0 correct, 1 error
        X, y = np.asarray(epochs, dtype=float), np.asarray(y)
        self.W_ = self._xdawn(X, y)
        self.P_ = np.vstack([(X[y == c].mean(axis=0) @ self.W_).T for c in (0, 1)])
        covs = self._covs(X)
        d, V = eigh(covs.mean(axis=0))
        self.iref_ = (V / np.sqrt(np.maximum(d, 1e-12))) @ V.T
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
