"""NeuroFoldNet and baseline model definitions.

NeuroFoldNet is a fixed-hyperparameter, soft-voting ensemble that combines two complementary
views of each emission spectrum:
  * the full normalised spectrum (shrinkage LDA, RBF-SVM, random forest), and
  * a compact set of physically interpretable spectral-shape descriptors (peak position,
    centroid, width, FWHM, skewness, blue/red asymmetry, derivative extrema) fed to a
    regularised logistic regression.
All hyperparameters are fixed a priori (LDA shrinkage is set analytically by Ledoit-Wolf),
so the ensemble involves no data-driven tuning that could leak across folds. Baselines are
tuned by inner, group-aware cross-validation inside every outer training fold.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 20260801


class SpectralDescriptors(BaseEstimator, TransformerMixin):
    """Stateless transform: spectrum (n_samples x n_wavelengths) -> shape descriptors."""

    names = [
        "peak_nm", "centroid_nm", "width_sd_nm", "fwhm_nm", "skewness",
        "blue_red_area_ratio", "total_area", "max_rise_nm", "max_fall_nm",
    ]

    def __init__(self, wavelengths=None):
        self.wavelengths = wavelengths

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.clip(np.asarray(X, dtype=float), 0.0, None)
        wl = np.asarray(self.wavelengths, dtype=float)
        area = np.trapezoid(X, wl, axis=1) + 1e-12
        w = X / X.sum(axis=1, keepdims=True).clip(1e-12)
        centroid = (w * wl).sum(axis=1)
        sd = np.sqrt((w * (wl - centroid[:, None]) ** 2).sum(axis=1))
        skew = (w * (wl - centroid[:, None]) ** 3).sum(axis=1) / np.maximum(sd, 1e-9) ** 3
        peak_idx = X.argmax(axis=1)
        peak = wl[peak_idx]
        fwhm = np.array([self._fwhm(row, wl) for row in X])
        blue = np.array([np.trapezoid(r[: i + 1], wl[: i + 1]) if i > 0 else 0.0 for r, i in zip(X, peak_idx)])
        red = area - blue
        deriv = np.gradient(X, wl, axis=1)
        rise = wl[deriv.argmax(axis=1)]
        fall = wl[deriv.argmin(axis=1)]
        return np.column_stack([peak, centroid, sd, fwhm, skew, (blue + 1e-6) / (red + 1e-6), area, rise, fall])

    @staticmethod
    def _fwhm(row, wl):
        half = row.max() / 2.0
        above = np.flatnonzero(row >= half)
        if above.size == 0:
            return 0.0
        lo, hi = above[0], above[-1]
        left = wl[lo] if lo == 0 else np.interp(half, [row[lo - 1], row[lo]], [wl[lo - 1], wl[lo]])
        right = wl[hi] if hi == len(row) - 1 else np.interp(half, [row[hi + 1], row[hi]], [wl[hi + 1], wl[hi]])
        return float(right - left)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.names)


def neurofoldnet(wavelengths) -> VotingClassifier:
    return VotingClassifier(
        estimators=[
            ("spectrum_lda", Pipeline([("scale", StandardScaler()),
                                       ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))])),
            ("spectrum_svm", Pipeline([("scale", StandardScaler()),
                                       ("clf", SVC(C=1.0, gamma="scale", probability=True, random_state=SEED))])),
            ("spectrum_rf", RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features="sqrt",
                                                   random_state=SEED, n_jobs=-1)),
            ("descriptor_lr", Pipeline([("desc", SpectralDescriptors(wavelengths)), ("scale", StandardScaler()),
                                        ("clf", LogisticRegression(C=1.0, max_iter=5000))])),
        ],
        voting="soft",
    )


def baseline_grids(wavelengths) -> dict[str, tuple[Pipeline, list[dict]]]:
    """Single-family baselines, each tuned by inner group-aware CV."""
    scale = ("scale", StandardScaler())
    return {
        "logistic_regression": (Pipeline([scale, ("clf", LogisticRegression(max_iter=5000))]),
                                [{"clf__C": [0.01, 0.1, 1.0, 10.0]}]),
        "shrinkage_lda": (Pipeline([scale, ("clf", LinearDiscriminantAnalysis(solver="lsqr"))]),
                          [{"clf__shrinkage": ["auto", 0.1, 0.5, 0.9]}]),
        "svm_rbf": (Pipeline([scale, ("clf", SVC(probability=True, random_state=SEED))]),
                    [{"clf__C": [0.1, 1.0, 10.0], "clf__gamma": ["scale", 0.01, 0.1]}]),
        "random_forest": (Pipeline([("clf", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1))]),
                          [{"clf__min_samples_leaf": [1, 3, 5], "clf__max_features": ["sqrt", 0.5]}]),
        "descriptors_only_lr": (Pipeline([("desc", SpectralDescriptors(wavelengths)), scale,
                                          ("clf", LogisticRegression(max_iter=5000))]),
                                [{"clf__C": [0.01, 0.1, 1.0, 10.0]}]),
    }


def small_n_grids() -> dict[str, tuple[Pipeline, list[dict]]]:
    """For low-dimensional, very small assays (e.g. Fig 4 seeding, n=18)."""
    scale = ("scale", StandardScaler())
    return {
        "logistic_regression": (Pipeline([scale, ("clf", LogisticRegression(max_iter=5000))]),
                                [{"clf__C": [0.1, 1.0, 10.0]}]),
        "shrinkage_lda": (Pipeline([scale, ("clf", LinearDiscriminantAnalysis(solver="lsqr"))]),
                          [{"clf__shrinkage": ["auto", 0.5]}]),
    }
