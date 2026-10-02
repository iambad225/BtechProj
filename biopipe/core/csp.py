"""Common Spatial Patterns (2-class), NumPy only.

CSP learns spatial filters that maximise the variance ratio between two classes of
band-passed EEG. On the sensorimotor band it turns left/right motor imagery into a small,
highly separable feature vector (log-variance of the projected components). Used for the
2-class MI decoder and reusable for the rest-vs-active ERD gate.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh


def _trial_cov(X: np.ndarray) -> np.ndarray:
    """Trace-normalised covariance of one epoch X (n_times, n_ch)."""
    X = X - X.mean(axis=0, keepdims=True)
    c = (X.T @ X) / max(1, X.shape[0] - 1)
    tr = np.trace(c)
    return c / tr if tr > 1e-12 else c


class CSP:
    def __init__(self, n_components: int = 4):
        self.n_components = n_components
        self.filters_ = None

    def fit(self, epochs: list[np.ndarray], y: np.ndarray) -> "CSP":
        """epochs: list of (n_times, n_ch); y: 0/1 labels."""
        y = np.asarray(y)
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("CSP needs exactly 2 classes.")
        c0 = np.mean([_trial_cov(epochs[i]) for i in np.where(y == classes[0])[0]], axis=0)
        c1 = np.mean([_trial_cov(epochs[i]) for i in np.where(y == classes[1])[0]], axis=0)
        # generalised eigenproblem: c0 w = lambda (c0 + c1) w
        d, V = eigh(c0, c0 + c1)
        order = np.argsort(d)[::-1]
        V = V[:, order]
        m = max(1, self.n_components // 2)
        n = V.shape[1]
        idx = list(range(m)) + list(range(n - m, n))
        self.filters_ = V[:, idx].T          # (n_components, n_ch)
        return self

    def transform(self, epochs: list[np.ndarray]) -> np.ndarray:
        feats = [self._features(X) for X in epochs]
        return np.asarray(feats)

    def _features(self, X: np.ndarray) -> np.ndarray:
        X = X - X.mean(axis=0, keepdims=True)
        Z = X @ self.filters_.T              # (n_times, n_components)
        v = np.var(Z, axis=0)
        v = v / (np.sum(v) + 1e-12)
        return np.log(v + 1e-12)
