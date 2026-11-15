"""Leakage-safe evaluation utilities.

* Repeated, stratified, group-aware outer CV; every fitted step (scaling, tuning, model)
  sees training-fold data only.
* Full metric panel: accuracy, balanced accuracy, macro-F1, macro one-vs-rest ROC-AUC,
  log loss, Brier score, and per-class sensitivity / specificity / precision / F1.
* Cluster (group) bootstrap confidence intervals on out-of-fold predictions.
* Group-level label-permutation test of the complete pipeline.
* Nadeau-Bengio corrected resampled t-test for paired model comparison.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss,
    precision_recall_fscore_support, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold


# ----------------------------------------------------------------------------- metrics
def metric_panel(y: np.ndarray, proba: np.ndarray, classes: list[str]) -> dict[str, float]:
    k = len(classes)
    pred = proba.argmax(axis=1)
    onehot = np.eye(k)[y]
    out = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", labels=range(k), zero_division=0),
        "macro_auc_ovr": roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=range(k))
        if len(np.unique(y)) == k else np.nan,
        "log_loss": log_loss(y, np.clip(proba, 1e-15, 1), labels=range(k)),
        "brier": float(((proba - onehot) ** 2).sum(axis=1).mean()),
    }
    prec, rec, f1, _ = precision_recall_fscore_support(y, pred, labels=range(k), zero_division=0)
    cm = confusion_matrix(y, pred, labels=range(k))
    for i, c in enumerate(classes):
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        out[f"{c}_sensitivity"] = rec[i]
        out[f"{c}_specificity"] = tn / (tn + fp) if (tn + fp) else np.nan
        out[f"{c}_precision"] = prec[i]
        out[f"{c}_f1"] = f1[i]
    return out


# ----------------------------------------------------------------------------- CV core
@dataclass
class CVResult:
    name: str
    classes: list[str]
    oof_proba: np.ndarray            # repeats x n x k
    fold_scores: pd.DataFrame        # one row per (repeat, fold)
    selections: list[dict] = field(default_factory=list)

    @property
    def mean_proba(self) -> np.ndarray:
        return self.oof_proba.mean(axis=0)

    def repeat_metrics(self, y: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame([metric_panel(y, p, self.classes) for p in self.oof_proba])


def _fit(estimator, grid, X, y, groups, inner_splits, seed):
    if grid is None:
        return clone(estimator).fit(X, y), {}
    inner = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    search = GridSearchCV(clone(estimator), grid, cv=inner, scoring="neg_log_loss", n_jobs=1, refit=True)
    search.fit(X, y, groups=groups)
    return search.best_estimator_, search.best_params_


def repeated_nested_cv(name, estimator, grid, X, y, groups, classes, outer_splits=5, inner_splits=3,
                       repeats=5, seed=0) -> CVResult:
    X = np.asarray(X)
    n, k = len(y), len(classes)
    proba = np.full((repeats, n, k), np.nan)
    rows, selections = [], []
    for r in range(repeats):
        outer = StratifiedGroupKFold(n_splits=outer_splits, shuffle=True, random_state=seed + r)
        for f, (tr, te) in enumerate(outer.split(X, y, groups)):
            model, params = _fit(estimator, grid, X[tr], y[tr], groups[tr], inner_splits, seed + 1000 * r + f)
            p = model.predict_proba(X[te])
            proba[r, te] = p
            m = metric_panel(y[te], p, classes)
            rows.append({"repeat": r, "fold": f, "n_test": len(te), **m})
            selections.append({"repeat": r, "fold": f, **{kk: str(v) for kk, v in params.items()}})
    if np.isnan(proba).any():
        raise RuntimeError("Some samples never received an out-of-fold prediction")
    return CVResult(name, list(classes), proba, pd.DataFrame(rows), selections)


# ----------------------------------------------------------------------------- inference
def cluster_bootstrap_ci(y, proba, groups, classes, n_boot=2000, seed=0, alpha=0.05) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    point = metric_panel(y, proba, classes)
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in pick])
        if len(np.unique(y[idx])) < len(classes):
            continue
        draws.append(metric_panel(y[idx], proba[idx], classes))
    d = pd.DataFrame(draws)
    return pd.DataFrame({
        "estimate": pd.Series(point),
        "ci_low": d.quantile(alpha / 2),
        "ci_high": d.quantile(1 - alpha / 2),
    })


def group_permutation_test(estimator, grid, X, y, groups, classes, n_perm=200, outer_splits=5, inner_splits=3,
                           seed=0, metric="balanced_accuracy") -> dict[str, float]:
    """Shuffle labels at the group level (every group keeps one label) and rerun the whole pipeline."""
    observed = repeated_nested_cv("obs", estimator, grid, X, y, groups, classes, outer_splits, inner_splits, 1, seed)
    obs = metric_panel(y, observed.mean_proba, classes)[metric]
    uniq = np.unique(groups)
    group_label = np.array([y[groups == g][0] for g in uniq])
    rng = np.random.default_rng(seed)
    null = []
    for i in range(n_perm):
        perm = dict(zip(uniq, rng.permutation(group_label)))
        y_perm = np.array([perm[g] for g in groups])
        res = repeated_nested_cv("perm", estimator, grid, X, y_perm, groups, classes, outer_splits, inner_splits,
                                 1, seed + 7919 * (i + 1))
        null.append(metric_panel(y_perm, res.mean_proba, classes)[metric])
    null = np.array(null)
    return {"metric": metric, "observed": obs, "null_mean": null.mean(), "null_sd": null.std(ddof=1),
            "null_95th": np.quantile(null, 0.95), "n_perm": n_perm,
            "p_value": (1 + (null >= obs).sum()) / (1 + n_perm)}


def corrected_resampled_ttest(a: np.ndarray, b: np.ndarray, n_train: int, n_test: int) -> dict[str, float]:
    """Nadeau & Bengio (2003) corrected repeated-CV t-test on paired per-fold scores."""
    d = np.asarray(a) - np.asarray(b)
    j = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return {"mean_diff": d.mean(), "t": np.nan, "p_value": 1.0 if d.mean() == 0 else 0.0}
    t = d.mean() / np.sqrt((1 / j + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), df=j - 1)
    return {"mean_diff": d.mean(), "t": t, "p_value": p}


def contiguous_blocks(labels: np.ndarray, index: np.ndarray, block_size: int) -> np.ndarray:
    """Group ID = label + contiguous block of the within-label acquisition index.

    Neighbouring spectra in the workbook may come from the same field of view or session; keeping
    each block on one side of every split prevents near-duplicate spectra from inflating accuracy.
    """
    return np.array([f"{l}_{i // block_size}" for l, i in zip(labels, index)])
