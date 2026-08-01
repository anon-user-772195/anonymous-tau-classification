from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from neurofoldnet.model import NeuroFoldNet
from neurofoldnet.tau_data import load_tau_feature_table
from neurofoldnet.full_supplement import build_tau_full_supplement_table


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
TAU_FEATURES = PROJECT_ROOT / "data" / "processed" / "tau_features.csv"
TAU_PLUS_GEO = PROJECT_ROOT / "data" / "processed" / "tau_plus_figure7_features.csv"
TAU_PLUS_ONE_FIG7_NONLEAKING = PROJECT_ROOT / "data" / "processed" / "tau_plus_one_fig7_nonleaking_feature.csv"
TAU_FULL_SUPPLEMENT = PROJECT_ROOT / "data" / "processed" / "tau_full_supplement_features.csv"
TAU_FULL_SUPPLEMENT_PLUS_GEO = PROJECT_ROOT / "data" / "processed" / "tau_full_supplement_plus_geo_features.csv"
RANDOM_STATE = 42
META_COLUMNS = {
    "sample_id",
    "group",
    "disease",
    "replicate",
    "geo_auxiliary_level",
    "geo_feature_level",
    "genotype",
    "is_untreated_control",
    "supplement_auxiliary_level",
    "one_fig7_feature_name",
    "one_fig7_auxiliary_level",
    "one_fig7_note",
    "fig7_context_feature_name",
    "fig7_context_level",
    "fig7_context_note",
}


def load_training_table(feature_mode: str) -> pd.DataFrame:
    if feature_mode == "tau_only":
        if TAU_FEATURES.exists():
            data = pd.read_csv(TAU_FEATURES)
            print(f"Loaded tau-only features: {TAU_FEATURES}")
        else:
            data = load_tau_feature_table()
            TAU_FEATURES.parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(TAU_FEATURES, index=False)
            print(f"Saved reconstructed tau-only features: {TAU_FEATURES}")
        return data

    if feature_mode == "tau_full_supplement":
        if TAU_FULL_SUPPLEMENT.exists():
            data = pd.read_csv(TAU_FULL_SUPPLEMENT)
            print(f"Loaded tau + full supplement features: {TAU_FULL_SUPPLEMENT}")
        else:
            data = build_tau_full_supplement_table()
            TAU_FULL_SUPPLEMENT.parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(TAU_FULL_SUPPLEMENT, index=False)
            print(f"Saved tau + full supplement features: {TAU_FULL_SUPPLEMENT}")
        print(
            "WARNING: full supplement features are disease-level summaries from unmatched "
            "figure blocks. Treat this as exploratory feature augmentation, not matched-sample validation."
        )
        return data

    if feature_mode == "tau_plus_one_fig7_nonleaking":
        if not TAU_PLUS_ONE_FIG7_NONLEAKING.exists():
            raise FileNotFoundError(
                f"Missing tau + non-leaking one Fig 7 feature table: {TAU_PLUS_ONE_FIG7_NONLEAKING}\n"
                "Run:\n"
                "  python scripts/create_tau_plus_one_fig7_nonleaking_feature.py"
            )
        data = pd.read_csv(TAU_PLUS_ONE_FIG7_NONLEAKING)
        print(f"Loaded tau + one non-leaking Figure 7 context feature: {TAU_PLUS_ONE_FIG7_NONLEAKING}")
        if "fig7_context_feature_name" in data.columns:
            print(f"Figure 7 context feature: {data['fig7_context_feature_name'].iloc[0]}")
            print(
                "NOTE: this feature is disease-blind and assigned without knowing disease labels. "
                "It is expected to behave as context, not as a predictive matched transcriptomic measurement."
            )
        return data

    if feature_mode == "tau_full_supplement_plus_geo":
        if not TAU_FULL_SUPPLEMENT_PLUS_GEO.exists():
            raise FileNotFoundError(
                f"Missing full supplement + GEO table: {TAU_FULL_SUPPLEMENT_PLUS_GEO}\n"
                "Run:\n"
                "  python scripts/prepare_full_supplement_features.py\n"
                "  python scripts/merge_tau_and_geo_features.py --tau-input data/processed/tau_full_supplement_features.csv --output data/processed/tau_full_supplement_plus_geo_features.csv"
            )
        data = pd.read_csv(TAU_FULL_SUPPLEMENT_PLUS_GEO)
        print(f"Loaded tau + full supplement + Figure 7 GEO features: {TAU_FULL_SUPPLEMENT_PLUS_GEO}")
        print(
            "WARNING: full supplement and GEO additions are disease-level auxiliary summaries. "
            "This mode is exploratory and should not be framed as matched-sample multimodal learning."
        )
        if "geo_auxiliary_level" in data.columns:
            print("GEO auxiliary level distribution:")
            print(data["geo_auxiliary_level"].value_counts(dropna=False).to_string())
        return data

    if not TAU_PLUS_GEO.exists():
        raise FileNotFoundError(
            f"Missing merged tau+GEO table: {TAU_PLUS_GEO}\n"
            "Run:\n"
            "  python scripts/prepare_figure7_geo_features.py\n"
            "  python scripts/merge_tau_and_geo_features.py"
        )
    data = pd.read_csv(TAU_PLUS_GEO)
    print(f"Loaded tau+Figure 7 GEO features: {TAU_PLUS_GEO}")
    if "geo_auxiliary_level" in data.columns:
        print("GEO auxiliary level distribution:")
        print(data["geo_auxiliary_level"].value_counts(dropna=False).to_string())
        if data["geo_auxiliary_level"].astype(str).eq("disease_mean").any():
            print(
                "WARNING: GEO features are disease-level auxiliary summaries because exact tau/GEO "
                "sample IDs did not match. This is suitable for research exploration only and "
                "must not be interpreted as a deployable diagnostic setting."
            )
    return data


def feature_columns(
    data: pd.DataFrame,
    feature_mode: str,
    allow_disease_level_aux: bool = False,
) -> list[str]:
    blocked = set(META_COLUMNS)
    if "geo" in feature_mode:
        blocked.update(c for c in data.columns if c.startswith("geo_pca_global_"))
        if (
            not allow_disease_level_aux
            and "geo_auxiliary_level" in data.columns
            and data["geo_auxiliary_level"].astype(str).eq("disease_mean").any()
        ):
            geo_blocked = [c for c in data.columns if c.startswith("geo_")]
            blocked.update(geo_blocked)
            print(
                "Ignoring Figure 7 GEO disease-level summary features "
                f"({len(geo_blocked)} columns)."
            )
    if (
        not allow_disease_level_aux
        and "supplement_auxiliary_level" in data.columns
        and data["supplement_auxiliary_level"].astype(str).eq("disease_summary").any()
    ):
        supplement_blocked = [c for c in data.columns if c.startswith("fig")]
        blocked.update(supplement_blocked)
        print(
            "Ignoring Figure 1-6 disease-level supplement summary features "
            f"({len(supplement_blocked)} columns)."
        )
    cols = [
        col
        for col in data.columns
        if col not in blocked and col != "disease" and pd.api.types.is_numeric_dtype(data[col])
    ]
    if not cols:
        raise ValueError("No numeric feature columns found.")
    return cols


def build_preprocessor(train_frame: pd.DataFrame, cols: list[str], feature_mode: str) -> ColumnTransformer:
    if feature_mode in {"tau_only", "tau_full_supplement", "tau_plus_one_fig7_nonleaking"}:
        return ColumnTransformer(
            transformers=[
                (
                    "all_numeric",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    cols,
                )
            ],
            remainder="drop",
        )

    geo_expr_cols = [c for c in cols if c.startswith("geo_expr_")]
    other_cols = [c for c in cols if c not in geo_expr_cols]
    transformers = []
    if other_cols:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                other_cols,
            )
        )
    if geo_expr_cols:
        n_train = train_frame.shape[0]
        n_components = min(5, len(geo_expr_cols), max(1, n_train - 1))
        transformers.append(
            (
                "geo_expr_fold_pca",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("pca", PCA(n_components=n_components, random_state=RANDOM_STATE)),
                    ]
                ),
                geo_expr_cols,
            )
        )
        print(
            f"Fold-local preprocessing: PCA({n_components}) will be fit on training-fold "
            f"GEO expression columns only ({len(geo_expr_cols)} columns)."
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def assert_no_group_leakage(groups_train: np.ndarray, groups_val: np.ndarray, fold: int) -> None:
    overlap = set(groups_train).intersection(set(groups_val))
    if overlap:
        raise AssertionError(f"Group leakage in fold {fold}: {sorted(overlap)[:10]}")
    print(f"Fold {fold}: group leakage check passed ({len(groups_train)} train rows, {len(groups_val)} validation rows).")


def build_classifier(classifier_name: str):
    if classifier_name == "neurofoldnet":
        return NeuroFoldNet(n_folds=5, random_state=RANDOM_STATE)
    if classifier_name == "logreg_balanced":
        return LogisticRegression(
            max_iter=2000,
            C=0.5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    if classifier_name == "svc_linear_balanced":
        return SVC(
            C=0.5,
            kernel="linear",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    if classifier_name == "svc_rbf_balanced":
        return SVC(
            C=1.0,
            kernel="rbf",
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    if classifier_name == "rf_balanced":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=2,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"Unknown classifier: {classifier_name}")


def run_cv(
    data: pd.DataFrame,
    feature_mode: str,
    classifier_name: str = "neurofoldnet",
    select_k: int | None = None,
    allow_disease_level_aux: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"disease", "group"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Training table missing required columns: {sorted(missing)}")

    data = data[data["disease"].isin(["AD", "DLB", "PSP"])].copy()
    cols = feature_columns(data, feature_mode, allow_disease_level_aux=allow_disease_level_aux)
    print(f"Using {len(cols)} numeric feature columns for {feature_mode}.")
    print(f"Classifier: {classifier_name}")
    if allow_disease_level_aux:
        print("EXPLORATORY MODE: disease-level auxiliary summary features are allowed.")
    if select_k is not None:
        print(f"Fold-local feature selection enabled: SelectKBest(k={select_k}).")
    print("Fold-local variance filtering enabled: constant transformed features are removed using training folds only.")
    print("Feature columns:")
    for col in cols:
        print(f"  - {col}")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data["disease"].astype(str))
    groups = data["group"].astype(str).to_numpy()

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_rows = []
    prediction_rows = []

    for fold, (train_idx, val_idx) in enumerate(splitter.split(data[cols], y, groups), start=1):
        train_groups = groups[train_idx]
        val_groups = groups[val_idx]
        assert_no_group_leakage(train_groups, val_groups, fold)

        train_frame = data.iloc[train_idx]
        val_frame = data.iloc[val_idx]
        preprocessor = build_preprocessor(train_frame, cols, feature_mode)
        steps = [
            ("preprocessor", preprocessor),
            ("variance_filter", VarianceThreshold()),
        ]
        if select_k is not None:
            steps.append(("feature_selector", SelectKBest(f_classif, k=select_k)))
        steps.append(("classifier", build_classifier(classifier_name)))
        estimator = Pipeline(steps=steps)

        print(f"Fold {fold}: fitting preprocessing and model on training fold only.")
        estimator.fit(train_frame[cols], y[train_idx])
        y_pred = estimator.predict(val_frame[cols])
        y_proba = estimator.predict_proba(val_frame[cols])

        acc = accuracy_score(y[val_idx], y_pred)
        f1 = f1_score(y[val_idx], y_pred, average="macro")
        loss = log_loss(y[val_idx], y_proba, labels=np.arange(len(label_encoder.classes_)))
        fold_rows.append(
            {
                "feature_mode": feature_mode,
                "classifier": classifier_name,
                "select_k": "none" if select_k is None else select_k,
                "fold": fold,
                "n_train": len(train_idx),
                "n_validation": len(val_idx),
                "accuracy": acc,
                "macro_f1": f1,
                "log_loss": loss,
            }
        )
        print(f"Fold {fold}: Accuracy={acc:.4f}, Macro-F1={f1:.4f}, LogLoss={loss:.4f}")

        for row_position, true_encoded, pred_encoded, proba in zip(val_idx, y[val_idx], y_pred, y_proba):
            row = {
                "feature_mode": feature_mode,
                "classifier": classifier_name,
                "select_k": "none" if select_k is None else select_k,
                "fold": fold,
                "sample_id": data.iloc[row_position].get("sample_id", row_position),
                "group": data.iloc[row_position]["group"],
                "true_label": label_encoder.inverse_transform([true_encoded])[0],
                "predicted_label": label_encoder.inverse_transform([pred_encoded])[0],
            }
            for class_idx, class_name in enumerate(label_encoder.classes_):
                row[f"prob_{class_name}"] = proba[class_idx]
            prediction_rows.append(row)

    fold_results = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    return fold_results, predictions


def update_summary(feature_mode: str, fold_results: pd.DataFrame, run_name: str) -> pd.DataFrame:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "model_comparison_summary.csv"
    row = {
        "feature_mode": run_name,
        "accuracy_mean": fold_results["accuracy"].mean(),
        "accuracy_sd": fold_results["accuracy"].std(ddof=1),
        "macro_f1_mean": fold_results["macro_f1"].mean(),
        "macro_f1_sd": fold_results["macro_f1"].std(ddof=1),
        "log_loss_mean": fold_results["log_loss"].mean(),
        "log_loss_sd": fold_results["log_loss"].std(ddof=1),
        "n_folds": fold_results.shape[0],
    }
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        summary = summary[summary["feature_mode"] != run_name]
        summary = pd.concat([summary, pd.DataFrame([row])], ignore_index=True)
    else:
        summary = pd.DataFrame([row])
    summary = summary.sort_values("feature_mode")
    summary.to_csv(summary_path, index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate NeuroFoldNet feature modes.")
    parser.add_argument(
        "--feature_mode",
        choices=[
            "tau_only",
            "tau_plus_one_fig7_nonleaking",
            "tau_plus_geo",
            "tau_full_supplement",
            "tau_full_supplement_plus_geo",
        ],
        default="tau_only",
    )
    parser.add_argument(
        "--classifier",
        choices=[
            "neurofoldnet",
            "logreg_balanced",
            "svc_linear_balanced",
            "svc_rbf_balanced",
            "rf_balanced",
        ],
        default="neurofoldnet",
    )
    parser.add_argument(
        "--select_k",
        type=int,
        default=None,
        help="Optional fold-local SelectKBest feature count after preprocessing and variance filtering.",
    )
    parser.add_argument(
        "--allow_disease_level_aux",
        action="store_true",
        help=(
            "Opt in to disease-level Figure 1-7 auxiliary summaries. "
            "Default training ignores these columns."
        ),
    )
    args = parser.parse_args()

    data = load_training_table(args.feature_mode)
    fold_results, predictions = run_cv(
        data,
        args.feature_mode,
        args.classifier,
        args.select_k,
        allow_disease_level_aux=args.allow_disease_level_aux,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = args.feature_mode
    if args.classifier != "neurofoldnet" or args.select_k is not None:
        suffix = f"{args.feature_mode}_{args.classifier}"
        if args.select_k is not None:
            suffix = f"{suffix}_top{args.select_k}"
    if args.allow_disease_level_aux:
        suffix = f"{suffix}_exploratory_disease_aux"
    fold_path = RESULTS_DIR / f"cv_results_{suffix}.csv"
    pred_path = RESULTS_DIR / f"cv_predictions_{suffix}.csv"
    fold_results.to_csv(fold_path, index=False)
    predictions.to_csv(pred_path, index=False)
    summary = update_summary(args.feature_mode, fold_results, suffix)

    print("=" * 80)
    print(f"Saved fold results: {fold_path}")
    print(f"Saved per-sample predictions: {pred_path}")
    print(f"Saved/updated comparison summary: {RESULTS_DIR / 'model_comparison_summary.csv'}")
    print("Summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
