from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neurofoldnet.model import NeuroFoldNet
from train import build_preprocessor, feature_columns


MODE_TO_PATH = {
    "tau_full_supplement": PROJECT_ROOT / "data" / "processed" / "tau_full_supplement_features.csv",
    "tau_full_supplement_plus_geo": PROJECT_ROOT / "data" / "processed" / "tau_full_supplement_plus_geo_features.csv",
}


def make_models(random_state: int):
    return {
        "NeuroFoldNet": lambda: NeuroFoldNet(n_folds=5, random_state=random_state),
        "LogReg": lambda: LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=random_state),
        "SVC": lambda: SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, class_weight="balanced", random_state=random_state),
        "RF": lambda: RandomForestClassifier(n_estimators=200, max_depth=3, class_weight="balanced", random_state=random_state),
        "GB": lambda: GradientBoostingClassifier(n_estimators=60, max_depth=2, learning_rate=0.05, random_state=random_state),
        "XGB": lambda: xgb.XGBClassifier(
            n_estimators=60,
            max_depth=2,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=random_state,
        ),
    }


def evaluate_mode(mode: str, k_values: list[int | None], random_state: int) -> list[dict[str, object]]:
    data = pd.read_csv(MODE_TO_PATH[mode])
    data = data[data["disease"].isin(["AD", "DLB", "PSP"])].copy()
    cols = feature_columns(data, mode)
    class_names = sorted(data["disease"].unique())
    y = pd.Categorical(data["disease"], categories=class_names).codes
    groups = data["group"].astype(str).to_numpy()
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    rows = []

    for k in k_values:
        for model_name, make_model in make_models(random_state).items():
            fold_metrics = []
            for train_idx, val_idx in cv.split(data[cols], y, groups):
                preprocessor = build_preprocessor(data.iloc[train_idx], cols, mode)
                steps = [("preprocessor", preprocessor)]
                if k is not None:
                    steps.append(("select", SelectKBest(f_classif, k=min(k, len(cols)))))
                steps.append(("classifier", make_model()))
                estimator = Pipeline(steps)
                estimator.fit(data.iloc[train_idx][cols], y[train_idx])
                pred = estimator.predict(data.iloc[val_idx][cols])
                proba = estimator.predict_proba(data.iloc[val_idx][cols])
                fold_metrics.append(
                    (
                        accuracy_score(y[val_idx], pred),
                        f1_score(y[val_idx], pred, average="macro"),
                        log_loss(y[val_idx], proba, labels=np.arange(len(class_names))),
                    )
                )
            fold_array = np.asarray(fold_metrics)
            rows.append(
                {
                    "mode": mode,
                    "model": model_name,
                    "k": "all" if k is None else k,
                    "accuracy_mean": fold_array[:, 0].mean(),
                    "macro_f1_mean": fold_array[:, 1].mean(),
                    "log_loss_mean": fold_array[:, 2].mean(),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Search simple classifiers on expanded full-supplement feature modes.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "expanded_model_search.csv")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    rows = []
    for mode in MODE_TO_PATH:
        if not MODE_TO_PATH[mode].exists():
            raise FileNotFoundError(f"Missing {MODE_TO_PATH[mode]}; run prepare/merge scripts first.")
        rows.extend(evaluate_mode(mode, [None, 10, 20, 40, 80, 120], args.random_state))

    result = pd.DataFrame(rows).sort_values(["accuracy_mean", "macro_f1_mean"], ascending=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved expanded model search: {args.output}")
    print(result.head(20).to_string(index=False))
    print(
        "WARNING: 100% scores here arise from disease-level auxiliary summary features. "
        "Use this as an exploratory sensitivity analysis, not as the main matched-sample model claim."
    )


if __name__ == "__main__":
    main()
