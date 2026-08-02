from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
PUBLIC_DIR = PROJECT_ROOT / "UI design" / "public"
FEATURE_TABLE = PROJECT_ROOT / "data" / "processed" / "tau_plus_one_fig7_nonleaking_feature.csv"
PREDICTIONS = RESULTS_DIR / "cv_predictions_tau_plus_one_fig7_nonleaking.csv"
FIGURE_FEATURES = {
    "FIG1": ["height", "area", "diameter"],
    "FIG2": [],
    "FIG3": ["proteo_resist_05", "proteo_resist_1"],
    "FIG4": ["seed_density"],
    "FIG5": ["basal_trans", "induced_trans"],
    "FIG6": [],
    "FIG7": ["fig7_global_tau_neuronal_response"],
}
CLASS_ORDER = ["AD", "DLB", "PSP"]
COLORS = {"AD": "#1f77b4", "DLB": "#ff7f0e", "PSP": "#2ca02c"}


def save_confusion_matrix() -> Path:
    predictions = pd.read_csv(PREDICTIONS)
    cm = confusion_matrix(
        predictions["true_label"],
        predictions["predicted_label"],
        labels=CLASS_ORDER,
    )
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_ORDER)
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    for text in display.text_.ravel():
        text.set_fontweight("bold")
        text.set_fontsize(10)
    ax.tick_params(axis="both", labelsize=9)
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
    coords = PCA(n_components=2, random_state=42).fit_transform(X_scaled) * 10.0
    pca = PCA(n_components=2, random_state=42).fit(X_scaled)

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for disease in CLASS_ORDER:
        mask = data["disease"].astype(str).eq(disease).to_numpy()
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=28,
            color=COLORS[disease],
            label=disease,
            alpha=0.85,
        )
    ax.set_title("PCA Plot")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.grid(True, color="#bcbcbc", alpha=0.55)
    ax.legend(title="disease", fontsize=8, title_fontsize=8, loc="upper right")
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
    variable_cols = [col for col in feature_cols if data[col].nunique(dropna=True) > 1]
    X = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    ).fit_transform(data[variable_cols])
    scores, _ = f_classif(X, y)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    rows = []
    for feature, score in zip(variable_cols, scores):
        figure = next(
            figure_name
            for figure_name, features in FIGURE_FEATURES.items()
            if feature in features
        )
        rows.append(
            {
                "feature": feature,
                "experimental_figure": figure,
                "importance_score": float(score),
            }
        )

    feature_importance = pd.DataFrame(rows)
    feature_path = RESULTS_DIR / "feature_importance_by_feature.csv"
    figure_path = RESULTS_DIR / "feature_importance_by_figure.csv"
    feature_importance.to_csv(feature_path, index=False)

    grouped = (
        feature_importance.groupby("experimental_figure", as_index=False)["importance_score"]
        .sum()
        .rename(columns={"importance_score": "importance_score"})
    )
    total = grouped["importance_score"].sum()
    grouped["relative_importance_percent"] = np.where(
        total > 0,
        grouped["importance_score"] / total * 100.0,
        0.0,
    )
    grouped["included_features"] = grouped["experimental_figure"].map(
        lambda figure: ", ".join(FIGURE_FEATURES[figure]) if FIGURE_FEATURES[figure] else "none"
    )
    grouped = grouped.sort_values("experimental_figure", ascending=False)
    grouped.to_csv(figure_path, index=False)

    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    ax.barh(
        grouped["experimental_figure"],
        grouped["relative_importance_percent"],
        color="#4f7fab",
    )
    for _, row in grouped.iterrows():
        value = row["relative_importance_percent"]
        ax.text(
            max(value, 0.15),
            row["experimental_figure"],
            f"{value:.2f}%",
            va="center",
            ha="left",
            fontsize=8,
        )
    ax.set_title("Feature Importance by Figure")
    ax.set_xlabel("Relative importance (%)")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#d0d0d0", alpha=0.55)
    ax.set_axisbelow(True)
    fig.tight_layout()
    output = RESULTS_DIR / "feature_importance_by_figure.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output, figure_path


def copy_to_public(paths: list[Path]) -> None:
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    name_map = {
        "confusion_matrix.png": "confusion-matrix.png",
        "pca_plot.png": "pca-plot.png",
        "feature_importance_by_figure.png": "feature-importance-by-figure.png",
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
    copy_to_public([confusion_path, pca_path, importance_path])

    print(f"Saved confusion matrix: {confusion_path}")
    print(f"Saved PCA plot: {pca_path}")
    print(f"Saved feature importance plot: {importance_path}")
    print(f"Saved feature importance table: {importance_csv}")
    print(f"Copied images to: {PUBLIC_DIR}")


if __name__ == "__main__":
    main()
