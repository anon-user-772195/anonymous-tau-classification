

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import load_training_table, run_cv


DEFAULT_MODES = [
    "tau_only",
    "tau_plus_geo",
    "tau_full_supplement",
    "tau_full_supplement_plus_geo",
]
DEFAULT_CLASSIFIERS = [
    "neurofoldnet",
    "logreg_balanced",
    "svc_linear_balanced",
    "svc_rbf_balanced",
    "rf_balanced",
]
DEFAULT_K = [None, 2, 3, 5, 8, 10, 20, 40]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run leakage-safe model/feature-selection search.")
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    parser.add_argument("--classifiers", nargs="+", default=DEFAULT_CLASSIFIERS)
    parser.add_argument("--k-values", nargs="+", default=["none", "2", "3", "5", "8", "10", "20", "40"])
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "leakage_safe_model_search.csv",
    )
    args = parser.parse_args()

    k_values: list[int | None] = []
    for value in args.k_values:
        k_values.append(None if str(value).lower() in {"none", "all"} else int(value))

    rows: list[dict[str, object]] = []
    for mode in args.modes:
        data = load_training_table(mode)
        for classifier in args.classifiers:
            for k in k_values:
                try:
                    fold_results, _ = run_cv(data, mode, classifier_name=classifier, select_k=k)
                except ValueError as exc:
                    print(f"Skipping mode={mode}, classifier={classifier}, k={k}: {exc}")
                    continue
                rows.append(
                    {
                        "feature_mode": mode,
                        "classifier": classifier,
                        "select_k": "none" if k is None else k,
                        "accuracy_mean": fold_results["accuracy"].mean(),
                        "accuracy_sd": fold_results["accuracy"].std(ddof=1),
                        "macro_f1_mean": fold_results["macro_f1"].mean(),
                        "macro_f1_sd": fold_results["macro_f1"].std(ddof=1),
                        "log_loss_mean": fold_results["log_loss"].mean(),
                        "log_loss_sd": fold_results["log_loss"].std(ddof=1),
                    }
                )

    result = pd.DataFrame(rows).sort_values(
        ["accuracy_mean", "macro_f1_mean", "log_loss_mean"],
        ascending=[False, False, True],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved leakage-safe search results: {args.output}")
    print(result.head(25).to_string(index=False))
    print(
        "Note: Figure-derived observations are used for feature generation. "
        "Rows are still grouped during evaluation to avoid cell/point-level pseudoreplication."
    )


if __name__ == "__main__":
    main()
