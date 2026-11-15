from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
PUBLIC_DIR = PROJECT_ROOT / "UI design" / "public"
FEATURE_TABLE = PROJECT_ROOT / "data" / "processed" / "tau_plus_one_fig7_nonleaking_feature.csv"
PREDICTIONS = RESULTS_DIR / "cv_predictions_tau_plus_one_fig7_nonleaking.csv"
FIGURE_FEATURES = {
    "Fig 1": ["height", "area", "diameter"],
    "Fig 2": [],
    "Fig 3": ["proteo_resist_05", "proteo_resist_1"],
    "Fig 4": ["seed_density"],
    "Fig 5": ["basal_trans", "induced_trans"],
    "Fig 6": [],
    "Fig 7": ["fig7_global_tau_neuronal_response"],
}
RAW_POINT_COUNTS = {
    "Fig 1": 10871,
    "Fig 2": 58980,
    "Fig 3": 36,
    "Fig 4": 120,
    "Fig 5": 991,
    "Fig 6": 1302,
    "Fig 7": 73630,
}
CLASS_ORDER = ["AD", "DLB", "PSP"]
COLORS = {"AD": "#2f6f73", "DLB": "#9b4d3d", "PSP": "#5f5aa2"}


def save_confusion_matrix() -> Path:
    predictions = pd.read_csv(PREDICTIONS)
    cm = confusion_matrix(
        predictions["true_label"],
        predictions["predicted_label"],
        labels=CLASS_ORDER,
    )
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_ORDER)
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted disease")
    ax.set_ylabel("True disease")
    fig.tight_layout()
    output = RESULTS_DIR / "confusion_matrix.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output


def save_pca_plot() -> Path:
    data = pd.read_csv(FEATURE_TABLE)
    feature_cols = [
        feature
        for features in FIGURE_FEATURES.values()
        for feature in features
        if feature in data.columns
    ]
    variable_cols = [col for col in feature_cols if data[col].nunique(dropna=True) > 1]
    X = data[variable_cols]
    X_scaled = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    ).fit_transform(X)
    coords = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
    pca = PCA(n_components=2, random_state=42).fit(X_scaled)

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    for disease in CLASS_ORDER:
        mask = data["disease"].astype(str).eq(disease).to_numpy()
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=64,
            color=COLORS[disease],
            label=disease,
            edgecolor="white",
            linewidth=0.8,
        )
    ax.axhline(0, color="#d0d7de", linewidth=0.8)
    ax.axvline(0, color="#d0d7de", linewidth=0.8)
    ax.set_title("PCA Plot")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    ax.legend(frameon=False, title="Disease")
    fig.tight_layout()
    output = RESULTS_DIR / "pca_plot.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output


def save_feature_importance_by_figure() -> tuple[Path, Path]:
    data = pd.read_csv(FEATURE_TABLE)
    feature_cols = [
        feature
        for features in FIGURE_FEATURES.values()
        for feature in features
        if feature in data.columns
    ]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data["disease"].astype(str))
    groups = data["group"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []

    for fold, (train_idx, val_idx) in enumerate(splitter.split(data[feature_cols], y, groups), start=1):
        estimator = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        C=0.5,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )
        estimator.fit(data.iloc[train_idx][feature_cols], y[train_idx])
        importance = permutation_importance(
            estimator,
            data.iloc[val_idx][feature_cols],
            y[val_idx],
            scoring="balanced_accuracy",
            n_repeats=100,
            random_state=42 + fold,
        )
        for feature, mean, sd in zip(feature_cols, importance.importances_mean, importance.importances_std):
            figure = next(
                figure_name
                for figure_name, features in FIGURE_FEATURES.items()
                if feature in features
            )
            rows.append(
                {
                    "fold": fold,
                    "feature": feature,
                    "experimental_figure": figure,
                    "importance_mean": mean,
                    "importance_sd": sd,
                }
            )

    feature_importance = pd.DataFrame(rows)
    feature_path = RESULTS_DIR / "feature_importance_by_feature.csv"
    figure_path = RESULTS_DIR / "feature_importance_by_figure.csv"
    feature_importance.to_csv(feature_path, index=False)

    grouped = (
        feature_importance.groupby("experimental_figure", as_index=False)["importance_mean"]
        .mean()
        .rename(columns={"importance_mean": "mean_permutation_importance"})
    )
    all_figures = pd.DataFrame({"experimental_figure": list(FIGURE_FEATURES)})
    grouped = all_figures.merge(grouped, on="experimental_figure", how="left").fillna(0.0)
    grouped["included_features"] = grouped["experimental_figure"].map(
        lambda figure: ", ".join(FIGURE_FEATURES[figure]) if FIGURE_FEATURES[figure] else "none"
    )
    grouped.to_csv(figure_path, index=False)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.bar(
        grouped["experimental_figure"],
        grouped["mean_permutation_importance"],
        color=["#2f6f73", "#d0d7de", "#8a7a3d", "#6f8f46", "#9b4d3d", "#d0d7de", "#5f5aa2"],
    )
    ax.set_title("Feature Importance by Figure")
    ax.set_xlabel("Experimental figure")
    ax.set_ylabel("Mean validation permutation importance")
    ax.axhline(0, color="#6e7781", linewidth=0.8)
    fig.tight_layout()
    output = RESULTS_DIR / "feature_importance_by_figure.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output, figure_path


def save_data_points_by_figure() -> tuple[Path, Path]:
    table = pd.DataFrame(
        [
            {"experimental_figure": figure, "numeric_points_or_cells": count}
            for figure, count in RAW_POINT_COUNTS.items()
        ]
    )
    table.loc[len(table)] = {
        "experimental_figure": "Total",
        "numeric_points_or_cells": int(table["numeric_points_or_cells"].sum()),
    }
    csv_path = RESULTS_DIR / "data_points_by_figure.csv"
    table.to_csv(csv_path, index=False)

    plot_table = table[table["experimental_figure"].ne("Total")].copy()
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.bar(
        plot_table["experimental_figure"],
        plot_table["numeric_points_or_cells"],
        color=["#2f6f73", "#8a7a3d", "#6f8f46", "#b07d3c", "#9b4d3d", "#5f5aa2", "#3f6fa6"],
    )
    ax.set_title("Data Points by Figure")
    ax.set_xlabel("Experimental figure")
    ax.set_ylabel("Numeric points or cells")
    ax.text(
        0.98,
        0.94,
        f"Total = {int(table.iloc[-1]['numeric_points_or_cells']):,}",
        transform=ax.transAxes,
        ha="right",
        va="top",
    )
    fig.tight_layout()
    output = RESULTS_DIR / "data_points_by_figure.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output, csv_path


def copy_to_public(paths: list[Path]) -> None:
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    name_map = {
        "confusion_matrix.png": "confusion-matrix.png",
        "pca_plot.png": "pca-plot.png",
        "feature_importance_by_figure.png": "feature-importance-by-figure.png",
        "data_points_by_figure.png": "data-points-by-figure.png",
    }
    for path in paths:
        shutil.copy2(path, PUBLIC_DIR / name_map[path.name])


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    missing = [path for path in [FEATURE_TABLE, PREDICTIONS] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input files: " + ", ".join(map(str, missing)))

    confusion_path = save_confusion_matrix()
    pca_path = save_pca_plot()
    importance_path, importance_csv = save_feature_importance_by_figure()
    point_path, point_csv = save_data_points_by_figure()
    copy_to_public([confusion_path, pca_path, importance_path, point_path])

    print(f"Saved confusion matrix: {confusion_path}")
    print(f"Saved PCA plot: {pca_path}")
    print(f"Saved feature importance plot: {importance_path}")
    print(f"Saved feature importance table: {importance_csv}")
    print(f"Saved data point plot: {point_path}")
    print(f"Saved data point table: {point_csv}")
    print(f"Copied images to: {PUBLIC_DIR}")


if __name__ == "__main__":
    main()
