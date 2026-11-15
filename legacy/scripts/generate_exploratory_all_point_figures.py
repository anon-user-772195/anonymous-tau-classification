from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_WORKBOOK = PROJECT_ROOT / "data" / "raw" / "42003_2025_7499_MOESM3_ESM.xlsx"
GEO_METADATA = PROJECT_ROOT / "data" / "figure7_geo" / "GSE277105_metadata.csv.gz"
RESULTS_DIR = PROJECT_ROOT / "results"

SUPPLEMENT_DISEASE_COLUMNS = {
    "Fig 1": {
        "AD": [0, 1, 2, 3, 4, 5, 21, 25, 29, 37, 38],
        "DLB": [7, 8, 9, 10, 11, 12, 22, 26, 30, 35, 36],
        "PSP": [14, 15, 16, 17, 18, 19, 23, 27, 31, 33, 34],
    },
    "Fig 2": {
        "AD": list(range(1, 151)),
        "DLB": list(range(151, 301)),
        "PSP": list(range(301, 451)),
    },
    "Fig 3": {
        "AD": [0, 1, 2],
        "DLB": [4, 5, 6],
        "PSP": [8, 9, 10],
    },
    "Fig 4": {
        "AD": list(range(1, 7)),
        "DLB": list(range(7, 13)),
        "PSP": list(range(13, 19)),
    },
    "Fig 5": {
        "AD": [0, 4, 8, 12, 16, 20],
        "DLB": [1, 5, 9, 13, 17, 21],
        "PSP": [2, 6, 10, 14, 18, 22],
    },
    "Fig 6": {
        "AD": [7, 8, 17, 23, 24, 36],
        "DLB": [1, 2, 14, 26, 27, 33],
        "PSP": [4, 5, 15, 29, 30, 34],
    },
}


def load_supplement_points() -> pd.DataFrame:
    rows = []
    for figure, disease_columns in SUPPLEMENT_DISEASE_COLUMNS.items():
        df = pd.read_excel(RAW_WORKBOOK, sheet_name=figure, header=None)
        label_by_col = {
            col: disease
            for disease, cols in disease_columns.items()
            for col in cols
        }
        numeric = df.apply(pd.to_numeric, errors="coerce")
        for row_idx, col_idx in zip(*np.where(numeric.notna().to_numpy())):
            value = float(numeric.iat[row_idx, col_idx])
            disease = label_by_col.get(int(col_idx), "Unmapped")
            rows.append(
                {
                    "source": "supplement",
                    "figure": figure,
                    "label": disease,
                    "value": value,
                    "row_index": float(row_idx),
                    "col_index": float(col_idx),
                    "n_feature": np.nan,
                    "percent_mito": np.nan,
                    "percent_ribo": np.nan,
                }
            )
    return pd.DataFrame(rows)


def load_geo_points() -> pd.DataFrame:
    meta = pd.read_csv(GEO_METADATA)
    sample = meta["sample"].astype(str)
    disease = np.select(
        [
            sample.str.contains("_AD", regex=False),
            sample.str.contains("_DLB", regex=False),
            sample.str.contains("_PSP", regex=False),
            sample.str.startswith("UT_"),
        ],
        ["AD", "DLB", "PSP", "Untreated"],
        default="Unmapped",
    )
    return pd.DataFrame(
        {
            "source": "geo_scrnaseq",
            "figure": "Fig 7",
            "label": disease,
            "value": pd.to_numeric(meta["nCount_RNA"], errors="coerce"),
            "row_index": np.arange(len(meta), dtype=float),
            "col_index": pd.to_numeric(meta["seurat_clusters"], errors="coerce"),
            "n_feature": pd.to_numeric(meta["nFeature_RNA"], errors="coerce"),
            "percent_mito": pd.to_numeric(meta["percent.mito"], errors="coerce"),
            "percent_ribo": pd.to_numeric(meta["percent.ribo"], errors="coerce"),
        }
    )


def build_all_points() -> pd.DataFrame:
    all_points = pd.concat([load_supplement_points(), load_geo_points()], ignore_index=True)
    all_points["abs_value"] = all_points["value"].abs()
    all_points["log_abs_value"] = np.log1p(all_points["abs_value"])
    return all_points


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                ["value", "log_abs_value", "row_index", "col_index", "n_feature", "percent_mito", "percent_ribo"],
            ),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), ["source", "figure"]),
        ]
    )


def save_pca(points: pd.DataFrame) -> Path:
    x = make_preprocessor().fit_transform(points)
    coords = PCA(n_components=2, random_state=42).fit_transform(x.toarray() if hasattr(x, "toarray") else x)
    colors = {
        "AD": "#1f77b4",
        "DLB": "#ff7f0e",
        "PSP": "#2ca02c",
        "Untreated": "#7f7f7f",
        "Unmapped": "#9467bd",
    }
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    for label in ["AD", "DLB", "PSP", "Untreated", "Unmapped"]:
        mask = points["label"].eq(label).to_numpy()
        if mask.any():
            ax.scatter(coords[mask, 0], coords[mask, 1], s=2, alpha=0.28, color=colors[label], label=label, linewidths=0)
    ax.set_title("PCA Plot")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.grid(True, color="#bcbcbc", alpha=0.55)
    ax.legend(title="label", markerscale=5, fontsize=8, title_fontsize=8, loc="upper right")
    fig.tight_layout()
    output = RESULTS_DIR / "pca_plot_145930_points.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output


def save_confusion_matrix(points: pd.DataFrame) -> Path:
    x = points[["source", "figure", "value", "log_abs_value", "row_index", "col_index", "n_feature", "percent_mito", "percent_ribo"]]
    y = points["label"].astype(str)
    model = Pipeline(
        [
            ("preprocess", make_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=10,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )
    labels = ["AD", "DLB", "PSP", "Untreated", "Unmapped"]
    pred = cross_val_predict(model, x, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), n_jobs=1)
    cm = confusion_matrix(y, pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(RESULTS_DIR / "confusion_matrix_145930_points.csv")
    pd.DataFrame({"label": y, "predicted": pred}).to_csv(RESULTS_DIR / "predictions_145930_points.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    for text in display.text_.ravel():
        text.set_fontweight("bold")
        text.set_fontsize(8)
    ax.tick_params(axis="both", labelsize=8)
    fig.tight_layout()
    output = RESULTS_DIR / "confusion_matrix_145930_points.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output


def main() -> None:
    if not RAW_WORKBOOK.exists():
        raise FileNotFoundError(RAW_WORKBOOK)
    if not GEO_METADATA.exists():
        raise FileNotFoundError(GEO_METADATA)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    points = build_all_points()
    counts = points.groupby(["source", "figure", "label"], dropna=False).size().reset_index(name="n_points")
    counts.to_csv(RESULTS_DIR / "label_counts_145930_points.csv", index=False)
    print(f"Total exploratory rows: {len(points):,}")
    print(counts.to_string(index=False))
    if len(points) != 145930:
        raise AssertionError(f"Expected 145,930 rows, found {len(points):,}")
    pca_path = save_pca(points)
    cm_path = save_confusion_matrix(points)
    print(f"Saved PCA: {pca_path}")
    print(f"Saved confusion matrix: {cm_path}")
    print("These images are exploratory point-level visuals and are not valid diagnostic performance figures.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
