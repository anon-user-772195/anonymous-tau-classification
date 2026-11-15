"""Sensitivity analysis: does spectral preprocessing or model family raise single-spectrum accuracy?

Grid: preprocessing {raw, SNV, Savitzky-Golay 1st derivative, raw + derivative} x classifier
{shrinkage LDA, RBF-SVM, histogram gradient boosting, extra trees}, each evaluated with the same
block-grouped 5-fold CV as the primary analysis. All 16 cells are reported; the best cell is an
optimistic upper bound (it is selected on the evaluation data) and is not used as a headline result.

Usage:  python scripts/run_spectral_preprocessing_sensitivity.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurofoldnet import assays as A  # noqa: E402
from neurofoldnet import evaluation as E  # noqa: E402

warnings.filterwarnings("ignore")
CLASSES = list(A.DISEASES)


def snv(X):
    return (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)


def deriv(X):
    return savgol_filter(X, 7, 2, deriv=1, axis=1)


def raw_deriv(X):
    return np.hstack([X, deriv(X)])


PRE = {"raw": None, "snv": snv, "deriv1": deriv, "raw+deriv1": raw_deriv}
CLF = {
    "shrinkage_lda": lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    "svm_rbf": lambda: SVC(probability=True, random_state=0),
    "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                                                     max_iter=200, random_state=0),
    "extra_trees": lambda: ExtraTreesClassifier(500, min_samples_leaf=2, random_state=0, n_jobs=-1),
}


def main() -> None:
    rows = []
    for dye in A.DYE_BLOCKS:
        s = A.load_spectra(dye)
        w = A.wavelength_columns(s)
        X = s[w].to_numpy()
        y = s["label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
        g = E.contiguous_blocks(s["label"].to_numpy(), s["spectrum_index"].to_numpy(), 10)
        for pn, pf in PRE.items():
            for cn, make in CLF.items():
                steps = ([("pre", FunctionTransformer(pf))] if pf else []) + [("scale", StandardScaler()),
                                                                               ("clf", make())]
                r = E.repeated_nested_cv(f"{pn}/{cn}", Pipeline(steps), None, X, y, g, CLASSES, repeats=1, seed=1)
                m = E.metric_panel(y, r.mean_proba, CLASSES)
                rows.append({"dye": dye, "preprocessing": pn, "classifier": cn,
                             "balanced_accuracy": m["balanced_accuracy"], "macro_auc_ovr": m["macro_auc_ovr"]})
        print(f"{dye} done", flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "v2" / "spectra_preprocessing_sensitivity.csv"
    df.to_csv(out, index=False)
    print(df.groupby("dye")["balanced_accuracy"].agg(["min", "median", "max"]).round(3))


if __name__ == "__main__":
    main()
