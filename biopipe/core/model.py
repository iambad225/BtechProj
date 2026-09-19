"""Multi-label gesture classifier.

The key design choice: this is *multi-label*, not multi-class. Each action gets its
own one-vs-rest binary classifier with calibrated probabilities, so several actions
can be reported active at the same instant -- exactly the "two actions at once"
requirement. "Rest" is not a class; it is simply every action sitting below threshold.

Models are pluggable (LDA / logistic regression / random forest / RBF-SVM). LDA is
the default: it is the sEMG-literature workhorse, gives smooth posteriors that
threshold cleanly, trains on little calibration data, and predicts in well under a
millisecond.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def _base_estimator(kind: str):
    kind = kind.lower()
    if kind == "lda":
        return LinearDiscriminantAnalysis()
    if kind == "logreg":
        return LogisticRegression(max_iter=2000, class_weight="balanced")
    if kind == "rf":
        return RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight="balanced")
    if kind == "svm":
        return SVC(kernel="rbf", probability=True, class_weight="balanced")
    raise ValueError(f"Unknown model type: {kind!r}")


class MultiLabelClassifier:
    def __init__(self, actions: list[str], model_type: str = "lda"):
        self.actions = list(actions)
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.clf = OneVsRestClassifier(_base_estimator(model_type))
        self._fitted = False

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "MultiLabelClassifier":
        """X: (n_windows, n_features). Y: (n_windows, n_actions) binary multi-hot."""
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=int)
        if X.shape[0] == 0:
            raise ValueError("No training windows -- record some data first.")
        if Y.shape[1] != len(self.actions):
            raise ValueError("Label columns do not match number of actions.")
        Xs = self.scaler.fit_transform(X)
        self.clf.fit(Xs, Y)
        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Independent P(active) per action. Shape (n_samples, n_actions)."""
        Xs = self.scaler.transform(np.asarray(X, dtype=float))
        proba = np.asarray(self.clf.predict_proba(Xs))
        # OneVsRestClassifier returns one column per label for multilabel targets;
        # guard the degenerate single-action case.
        if proba.ndim == 1:
            proba = proba[:, None]
        return proba

    # --- persistence -----------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "MultiLabelClassifier":
        return joblib.load(Path(path))
