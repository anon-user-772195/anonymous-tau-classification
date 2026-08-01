

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import RANDOM_STATE, build_preprocessor, feature_columns, load_training_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-mode", default="tau_full_supplement_plus_geo")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "selected_features_top3.csv",
    )
    args = parser.parse_args()

    data = load_training_table(args.feature_mode)
    data = data[data["disease"].isin(["AD", "DLB", "PSP"])].copy()
    cols = feature_columns(data, args.feature_mode)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data["disease"].astype(str))
    groups = data["group"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rows: list[dict[str, object]] = []
    for fold, (train_idx, _) in enumerate(splitter.split(data[cols], y, groups), start=1):
        train_frame = data.iloc[train_idx]
        preprocessor = build_preprocessor(train_frame, cols, args.feature_mode)
        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("variance_filter", VarianceThreshold()),
                ("feature_selector", SelectKBest(f_classif, k=args.k)),
            ]
        )
        pipe.fit(train_frame[cols], y[train_idx])

        pre_names = pipe.named_steps["preprocessor"].get_feature_names_out()
        variance_mask = pipe.named_steps["variance_filter"].get_support()
        variable_names = np.asarray(pre_names)[variance_mask]
        selector = pipe.named_steps["feature_selector"]
        selected_names = variable_names[selector.get_support()]
        selected_scores = selector.scores_[selector.get_support()]

        for rank, (name, score) in enumerate(
            sorted(zip(selected_names, selected_scores), key=lambda item: item[1], reverse=True),
            start=1,
        ):
            rows.append(
                {
                    "feature_mode": args.feature_mode,
                    "fold": fold,
                    "rank": rank,
                    "feature": name,
                    "f_score": score,
                }
            )

    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved selected features: {args.output}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
