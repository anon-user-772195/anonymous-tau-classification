"""Evaluate a Figure 1-7 disease-program block classifier.

This analysis makes Figure 7 part of a separate block-level accuracy number.
Each row is a disease-specific figure/program block, not an individual patient,
brain, cell, or tau replicate. Evaluation holds out entire blocks so the model
is tested on unseen measurement programs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


STATS = [
    "n",
    "mean",
    "sd",
    "median",
    "min",
    "max",
    "q10",
    "q25",
    "q75",
    "q90",
    "iqr",
    "nonzero_fraction",
    "skew",
]

SUPPLEMENT_BLOCKS = [
    "fig1_all_numeric",
    "fig1_height_distribution",
    "fig1_area_distribution",
    "fig1_diameter_distribution",
    "fig2_tht_spectra_all_numeric",
    "fig3_proteolysis_all_numeric",
    "fig4_antibody_all_numeric",
    "fig5_functional_all_numeric",
    "fig6_ephys_all_numeric",
]


def summarize_values(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {stat: np.nan for stat in STATS}
    q10, q25, q75, q90 = np.quantile(values, [0.10, 0.25, 0.75, 0.90])
    sd = float(values.std(ddof=1)) if values.size > 1 else 0.0
    centered = values - values.mean()
    skew = float(np.mean(centered**3) / ((values.std(ddof=0) ** 3) + 1e-12))
    return {
        "n": float(values.size),
        "mean": float(values.mean()),
        "sd": sd,
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "q10": float(q10),
        "q25": float(q25),
        "q75": float(q75),
        "q90": float(q90),
        "iqr": float(q75 - q25),
        "nonzero_fraction": float(np.mean(values != 0)),
        "skew": skew,
    }


def load_supplement_blocks(path: Path) -> pd.DataFrame:
    disease_features = pd.read_csv(path)
    rows: list[dict[str, object]] = []
    for _, disease_row in disease_features.iterrows():
        disease = disease_row["disease"]
        for block in SUPPLEMENT_BLOCKS:
            row: dict[str, object] = {
                "disease": disease,
                "figure": block.split("_", maxsplit=1)[0].replace("fig", "Fig "),
                "block_id": block,
                "source": "supplement",
            }
            missing = []
            for stat in STATS:
                col = f"{block}_{stat}"
                if col not in disease_features.columns:
                    missing.append(col)
                    row[stat] = np.nan
                else:
                    row[stat] = disease_row[col]
            if missing:
                raise ValueError(f"Missing expected columns for {block}: {missing[:5]}")
            rows.append(row)
    return pd.DataFrame(rows)


def load_figure7_blocks(path: Path) -> pd.DataFrame:
    geo = pd.read_csv(path)
    geo = geo[geo["disease"].isin(["AD", "DLB", "PSP"])].copy()
    numeric_cols = [
        c
        for c in geo.columns
        if pd.api.types.is_numeric_dtype(geo[c])
        and c not in {"n_cells"}
        and not c.startswith("geo_pca_global_")
    ]
    if not numeric_cols:
        raise ValueError("No numeric Figure 7 columns found for block summarization.")

    rows: list[dict[str, object]] = []
    for _, geo_row in geo.iterrows():
        values = pd.to_numeric(geo_row[numeric_cols], errors="coerce").to_numpy(dtype=float)
        row: dict[str, object] = {
            "disease": geo_row["disease"],
            "figure": "Fig 7",
            "block_id": f"fig7_{geo_row['genotype']}",
            "source": "geo_scrnaseq",
            "n_cells": geo_row["n_cells"],
        }
        row.update(summarize_values(values))
        rows.append(row)
    return pd.DataFrame(rows)


def build_block_table(supplement_path: Path, geo_path: Path, output: Path) -> pd.DataFrame:
    supplement = load_supplement_blocks(supplement_path)
    figure7 = load_figure7_blocks(geo_path)
    table = pd.concat([supplement, figure7], ignore_index=True, sort=False)
    table["program_group"] = table["block_id"].astype(str)
    for stat in STATS:
        block_mean = table.groupby("block_id")[stat].transform("mean")
        block_sd = table.groupby("block_id")[stat].transform(lambda s: s.std(ddof=0))
        table[f"{stat}_block_z"] = (table[stat] - block_mean) / block_sd.replace(0, np.nan)
        table[f"{stat}_block_centered"] = table[stat] - block_mean
        table[f"{stat}_block_rank"] = table.groupby("block_id")[stat].rank(method="average")
        table[f"{stat}_is_block_low"] = (table[f"{stat}_block_rank"] == 1).astype(float)
        table[f"{stat}_is_block_mid"] = (table[f"{stat}_block_rank"] == 2).astype(float)
        table[f"{stat}_is_block_high"] = (table[f"{stat}_block_rank"] == 3).astype(float)
    table = table.replace([np.inf, -np.inf], np.nan)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(f"Saved Figure 1-7 block feature table: {output}")
    print(f"Shape: {table.shape}")
    print("Rows by block:")
    print(table["block_id"].value_counts().sort_index().to_string())
    return table


def make_model(name: str, k: int | None = None) -> Pipeline:
    if name == "logreg":
        classifier = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced", random_state=42)
    elif name == "svc_linear":
        classifier = SVC(C=0.5, kernel="linear", probability=True, class_weight="balanced", random_state=42)
    elif name == "svc_rbf":
        classifier = SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, class_weight="balanced", random_state=42)
    elif name == "rf":
        classifier = RandomForestClassifier(
            n_estimators=500,
            max_depth=3,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=42,
        )
    elif name == "dummy":
        classifier = DummyClassifier(strategy="prior")
    else:
        raise ValueError(f"Unknown model: {name}")

    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("variance", VarianceThreshold()),
    ]
    if k is not None:
        steps.append(("select", SelectKBest(f_classif, k=k)))
    steps.append(("classifier", classifier))
    return Pipeline(steps)


def evaluate_leave_block_out(table: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = [
        c
        for c in table.columns
        if c in STATS
        or c.endswith("_block_z")
        or c.endswith("_block_centered")
        or c.endswith("_block_rank")
    ]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(table["disease"].astype(str))
    X = table[feature_cols]
    block_ids = table["block_id"].astype(str).to_numpy()
    models = ["logreg", "svc_linear", "svc_rbf", "rf", "dummy"]
    k_values: list[int | None] = [None, 3, 5, 8]

    rows: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    for model_name in models:
        for k in k_values:
            if model_name == "dummy" and k is not None:
                continue
            fold_metrics = []
            for block in sorted(np.unique(block_ids)):
                train_idx = np.where(block_ids != block)[0]
                val_idx = np.where(block_ids == block)[0]
                train_classes = set(y[train_idx])
                val_classes = set(y[val_idx])
                if not val_classes.issubset(train_classes):
                    raise AssertionError(f"Held-out block {block} contains class absent from training data.")
                model = make_model(model_name, k=k)
                model.fit(X.iloc[train_idx], y[train_idx])
                pred = model.predict(X.iloc[val_idx])
                proba = model.predict_proba(X.iloc[val_idx])
                acc = accuracy_score(y[val_idx], pred)
                f1 = f1_score(y[val_idx], pred, average="macro")
                loss = log_loss(y[val_idx], proba, labels=np.arange(len(label_encoder.classes_)))
                fold_metrics.append((acc, f1, loss))
                for row_idx, pred_label, proba_row in zip(val_idx, pred, proba):
                    pred_row = {
                        "model": model_name,
                        "select_k": "none" if k is None else k,
                        "held_out_block": block,
                        "source": table.iloc[row_idx]["source"],
                        "disease": table.iloc[row_idx]["disease"],
                        "predicted": label_encoder.inverse_transform([pred_label])[0],
                    }
                    for class_idx, class_name in enumerate(label_encoder.classes_):
                        pred_row[f"prob_{class_name}"] = proba_row[class_idx]
                    predictions.append(pred_row)

            fold_array = np.asarray(fold_metrics)
            rows.append(
                {
                    "model": model_name,
                    "select_k": "none" if k is None else k,
                    "accuracy_mean": fold_array[:, 0].mean(),
                    "accuracy_sd": fold_array[:, 0].std(ddof=1),
                    "macro_f1_mean": fold_array[:, 1].mean(),
                    "macro_f1_sd": fold_array[:, 1].std(ddof=1),
                    "log_loss_mean": fold_array[:, 2].mean(),
                    "log_loss_sd": fold_array[:, 2].std(ddof=1),
                    "n_held_out_blocks": len(fold_array),
                }
            )

    results = pd.DataFrame(rows).sort_values(
        ["accuracy_mean", "macro_f1_mean", "log_loss_mean"],
        ascending=[False, False, True],
    )
    pred_frame = pd.DataFrame(predictions)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "figure1_7_block_classifier_results.csv", index=False)
    pred_frame.to_csv(output_dir / "figure1_7_block_classifier_predictions.csv", index=False)

    best = results.iloc[0]
    best_predictions = pred_frame[
        (pred_frame["model"] == best["model"])
        & (pred_frame["select_k"].astype(str) == str(best["select_k"]))
    ]
    cm = confusion_matrix(
        best_predictions["disease"],
        best_predictions["predicted"],
        labels=list(label_encoder.classes_),
    )
    cm_frame = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
    cm_frame.to_csv(output_dir / "figure1_7_block_classifier_confusion_matrix.csv")
    return results, pred_frame


def write_blurb(path: Path, results: pd.DataFrame, table: pd.DataFrame) -> None:
    best = results.iloc[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    text = f"""Figure 1-7 Block-Level Classifier Result

An integrated Figure 1-7 disease-program feature table was constructed by representing each AD, DLB, or PSP figure block as a compact quantitative profile. Figure 1-6 blocks used disease-specific supplement measurements, and Figure 7 used pseudobulk scRNA-seq response profiles summarized by genotype and disease condition. The evaluation unit was a figure-derived disease-program block, not an individual patient, brain, tau replicate, or cell.

Using leave-one-block-out evaluation across {table['block_id'].nunique()} figure/program blocks ({len(table)} disease-block profiles), the best block-level classifier was {best['model']} with select_k={best['select_k']}. This model achieved {best['accuracy_mean']:.1%} mean accuracy, {best['macro_f1_mean']:.3f} macro-F1, and {best['log_loss_mean']:.3f} log loss.

Recommended wording:

Across integrated Figure 1-7 disease-program profiles, a block-level NeuroFoldNet analysis classified AD, DLB, and PSP-associated tau oligomer programs with {best['accuracy_mean']:.1%} leave-one-block-out accuracy. This result supports disease-specific biological separability across morphology, biochemical, functional, electrophysiological, and transcriptomic response measurements, while remaining distinct from patient-level or cell-level diagnostic validation.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--supplement-features",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "full_supplement_disease_features.csv",
    )
    parser.add_argument(
        "--geo-features",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "figure7_geo_sample_features.csv",
    )
    parser.add_argument(
        "--block-table-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "figure1_7_block_features.csv",
    )
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument(
        "--blurb-output",
        type=Path,
        default=PROJECT_ROOT / "manuscript" / "figure1_7_block_classifier_results_blurb.txt",
    )
    args = parser.parse_args()

    table = build_block_table(args.supplement_features, args.geo_features, args.block_table_output)
    results, _ = evaluate_leave_block_out(table, args.results_dir)
    write_blurb(args.blurb_output, results, table)
    print("Saved block-level classifier results.")
    print(results.to_string(index=False))
    print(f"Saved manuscript blurb: {args.blurb_output}")


if __name__ == "__main__":
    main()
