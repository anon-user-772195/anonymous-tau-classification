

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import (
    RANDOM_STATE,
    build_classifier,
    build_preprocessor,
    feature_columns,
    load_training_table,
)


CLASSIFIERS = [
    "logreg_balanced",
    "svc_linear_balanced",
    "svc_rbf_balanced",
    "rf_balanced",
]
K_VALUES = [None, 2, 3, 5, 8]


def make_estimator(data: pd.DataFrame, cols: list[str], mode: str, classifier: str, k: int | None) -> Pipeline:
    steps = [
        ("preprocessor", build_preprocessor(data, cols, mode)),
        ("variance_filter", VarianceThreshold()),
    ]
    if k is not None:
        steps.append(("feature_selector", SelectKBest(f_classif, k=k)))
    steps.append(("classifier", build_classifier(classifier)))
    return Pipeline(steps=steps)


def score_candidate(
    data: pd.DataFrame,
    cols: list[str],
    y: np.ndarray,
    groups: np.ndarray,
    mode: str,
    train_idx: np.ndarray,
    classifier: str,
    k: int | None,
) -> dict[str, float]:
    inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    metrics = []
    train_data = data.iloc[train_idx]
    train_y = y[train_idx]
    train_groups = groups[train_idx]
    for inner_train_rel, inner_val_rel in inner.split(train_data[cols], train_y, train_groups):
        inner_train_idx = train_idx[inner_train_rel]
        inner_val_idx = train_idx[inner_val_rel]
        estimator = make_estimator(data.iloc[inner_train_idx], cols, mode, classifier, k)
        estimator.fit(data.iloc[inner_train_idx][cols], y[inner_train_idx])
        pred = estimator.predict(data.iloc[inner_val_idx][cols])
        proba = estimator.predict_proba(data.iloc[inner_val_idx][cols])
        metrics.append(
            {
                "accuracy": accuracy_score(y[inner_val_idx], pred),
                "macro_f1": f1_score(y[inner_val_idx], pred, average="macro"),
                "log_loss": log_loss(y[inner_val_idx], proba, labels=np.arange(len(np.unique(y)))),
            }
        )
    frame = pd.DataFrame(metrics)
    return {
        "inner_accuracy": frame["accuracy"].mean(),
        "inner_macro_f1": frame["macro_f1"].mean(),
        "inner_log_loss": frame["log_loss"].mean(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-mode", default="tau_full_supplement_plus_geo")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "nested_model_selection_results.csv")
    parser.add_argument("--allow-disease-level-aux", action="store_true")
    args = parser.parse_args()

    data = load_training_table(args.feature_mode)
    data = data[data["disease"].isin(["AD", "DLB", "PSP"])].copy()
    cols = feature_columns(
        data,
        args.feature_mode,
        allow_disease_level_aux=args.allow_disease_level_aux,
    )
    labels = LabelEncoder()
    y = labels.fit_transform(data["disease"].astype(str))
    groups = data["group"].astype(str).to_numpy()
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rows: list[dict[str, object]] = []
    for fold, (train_idx, val_idx) in enumerate(outer.split(data[cols], y, groups), start=1):
        candidates = []
        for classifier in CLASSIFIERS:
            for k in K_VALUES:
                result = score_candidate(data, cols, y, groups, args.feature_mode, train_idx, classifier, k)
                candidates.append({"classifier": classifier, "select_k": k, **result})
        candidate_frame = pd.DataFrame(candidates).sort_values(
            ["inner_macro_f1", "inner_accuracy", "inner_log_loss"],
            ascending=[False, False, True],
        )
        best = candidate_frame.iloc[0]
        estimator = make_estimator(
            data.iloc[train_idx],
            cols,
            args.feature_mode,
            str(best["classifier"]),
            None if pd.isna(best["select_k"]) else int(best["select_k"]),
        )
        estimator.fit(data.iloc[train_idx][cols], y[train_idx])
        pred = estimator.predict(data.iloc[val_idx][cols])
        proba = estimator.predict_proba(data.iloc[val_idx][cols])
        rows.append(
            {
                "feature_mode": args.feature_mode,
                "fold": fold,
                "selected_classifier": best["classifier"],
                "selected_k": "none" if pd.isna(best["select_k"]) else int(best["select_k"]),
                "inner_macro_f1": best["inner_macro_f1"],
                "outer_accuracy": accuracy_score(y[val_idx], pred),
                "outer_macro_f1": f1_score(y[val_idx], pred, average="macro"),
                "outer_log_loss": log_loss(y[val_idx], proba, labels=np.arange(len(labels.classes_))),
            }
        )

    result = pd.DataFrame(rows)
    summary = {
        "accuracy_mean": result["outer_accuracy"].mean(),
        "accuracy_sd": result["outer_accuracy"].std(ddof=1),
        "macro_f1_mean": result["outer_macro_f1"].mean(),
        "macro_f1_sd": result["outer_macro_f1"].std(ddof=1),
        "log_loss_mean": result["outer_log_loss"].mean(),
        "log_loss_sd": result["outer_log_loss"].std(ddof=1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved nested model-selection results: {args.output}")
    print(result.to_string(index=False))
    print("Summary:")
    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    main()
