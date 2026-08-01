from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from neurofoldnet.tau_data import find_tau_workbook, load_tau_feature_table


DISEASES = ("AD", "DLB", "PSP")


def _numeric_values(df: pd.DataFrame, rows: slice, cols: list[int]) -> np.ndarray:
    block = df.iloc[rows, cols]
    values = pd.to_numeric(pd.Series(block.to_numpy().ravel()), errors="coerce").dropna().to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _summaries(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            f"{prefix}_n": 0.0,
            f"{prefix}_mean": np.nan,
            f"{prefix}_sd": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_q10": np.nan,
            f"{prefix}_q25": np.nan,
            f"{prefix}_q75": np.nan,
            f"{prefix}_q90": np.nan,
            f"{prefix}_iqr": np.nan,
            f"{prefix}_nonzero_fraction": np.nan,
            f"{prefix}_skew": np.nan,
        }
    q10, q25, q75, q90 = np.quantile(values, [0.10, 0.25, 0.75, 0.90])
    sd = float(values.std(ddof=1)) if values.size > 1 else 0.0
    centered = values - values.mean()
    skew = float(np.mean(centered**3) / ((values.std(ddof=0) ** 3) + 1e-12))
    return {
        f"{prefix}_n": float(values.size),
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_sd": sd,
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_min": float(values.min()),
        f"{prefix}_max": float(values.max()),
        f"{prefix}_q10": float(q10),
        f"{prefix}_q25": float(q25),
        f"{prefix}_q75": float(q75),
        f"{prefix}_q90": float(q90),
        f"{prefix}_iqr": float(q75 - q25),
        f"{prefix}_nonzero_fraction": float(np.mean(values != 0)),
        f"{prefix}_skew": skew,
    }


def _add_block_features(features: dict[str, dict[str, float]], df: pd.DataFrame, sheet_prefix: str, blocks: dict[str, list[int]]) -> None:
    for disease, cols in blocks.items():
        values = _numeric_values(df, slice(0, df.shape[0]), cols)
        features[disease].update(_summaries(values, f"{sheet_prefix}_all_numeric"))


def load_full_supplement_disease_features(path: str | Path | None = None) -> pd.DataFrame:
    """
    Summarize all disease-specific numeric values available in the supplement.

    These are disease-level descriptors, not independent sample rows. They are
    useful for exploratory feature augmentation but should not be framed as
    matched-sample multimodal data.
    """
    workbook = find_tau_workbook(path)
    print(f"Loading full supplement numeric features from: {workbook}")
    features: dict[str, dict[str, float | str]] = {
        disease: {"disease": disease} for disease in DISEASES
    }

    # Fig 1: AFM distributions, replicated values, and Fig 1G traces.
    fig1 = pd.read_excel(workbook, sheet_name="Fig 1", header=None)
    _add_block_features(
        features,
        fig1,
        "fig1",
        {
            "AD": [0, 1, 2, 3, 4, 5, 21, 25, 29, 37, 38],
            "DLB": [7, 8, 9, 10, 11, 12, 22, 26, 30, 35, 36],
            "PSP": [14, 15, 16, 17, 18, 19, 23, 27, 31, 33, 34],
        },
    )
    _add_named_distribution(features, fig1, "fig1_height_distribution", {"AD": [0, 1], "DLB": [7, 8], "PSP": [14, 15]})
    _add_named_distribution(features, fig1, "fig1_area_distribution", {"AD": [2, 3], "DLB": [9, 10], "PSP": [16, 17]})
    _add_named_distribution(features, fig1, "fig1_diameter_distribution", {"AD": [4, 5], "DLB": [11, 12], "PSP": [18, 19]})

    # Fig 2: ThT spectra. Column ranges are explicitly labeled in row 2.
    fig2 = pd.read_excel(workbook, sheet_name="Fig 2", header=None)
    _add_block_features(
        features,
        fig2,
        "fig2_tht_spectra",
        {
            "AD": list(range(1, 151)),
            "DLB": list(range(151, 301)),
            "PSP": list(range(301, 451)),
        },
    )

    # Fig 3: proteolytic resistance timecourse blocks.
    fig3 = pd.read_excel(workbook, sheet_name="Fig 3", header=None)
    _add_block_features(features, fig3, "fig3_proteolysis", {"AD": [0, 1, 2], "DLB": [4, 5, 6], "PSP": [8, 9, 10]})

    # Fig 4: antibody/seeding matrix. A beta oligomer controls are excluded.
    fig4 = pd.read_excel(workbook, sheet_name="Fig 4", header=None)
    _add_block_features(
        features,
        fig4,
        "fig4_antibody",
        {
            "AD": list(range(1, 7)),
            "DLB": list(range(7, 13)),
            "PSP": list(range(13, 19)),
        },
    )

    # Fig 5: toxicity/transition assays, split by disease columns across subpanels.
    fig5 = pd.read_excel(workbook, sheet_name="Fig 5", header=None)
    _add_block_features(
        features,
        fig5,
        "fig5_functional",
        {
            "AD": [0, 4, 8, 12, 16, 20],
            "DLB": [1, 5, 9, 13, 17, 21],
            "PSP": [2, 6, 10, 14, 18, 22],
        },
    )

    # Fig 6: electrophysiology traces and bar values. Vehicle/CTRL/n columns are excluded.
    fig6 = pd.read_excel(workbook, sheet_name="Fig 6", header=None)
    _add_block_features(
        features,
        fig6,
        "fig6_ephys",
        {
            "AD": [7, 8, 17, 23, 24, 36],
            "DLB": [1, 2, 14, 26, 27, 33],
            "PSP": [4, 5, 15, 29, 30, 34],
        },
    )

    out = pd.DataFrame([features[disease] for disease in DISEASES])
    print(f"Full supplement disease-level feature table shape: {out.shape}")
    return out


def _add_named_distribution(features: dict[str, dict[str, float]], df: pd.DataFrame, prefix: str, blocks: dict[str, list[int]]) -> None:
    for disease, cols in blocks.items():
        values = _numeric_values(df, slice(0, df.shape[0]), cols)
        features[disease].update(_summaries(values, prefix))


def build_tau_full_supplement_table(path: str | Path | None = None) -> pd.DataFrame:
    tau = load_tau_feature_table(path)
    full = load_full_supplement_disease_features(path)
    merged = tau.merge(full, on="disease", how="left")
    merged["supplement_auxiliary_level"] = "disease_summary"
    print(f"Tau + full supplement feature table shape: {merged.shape}")
    return merged


def save_full_supplement_outputs(
    tau_output: str | Path = "data/processed/tau_full_supplement_features.csv",
    disease_output: str | Path = "data/processed/full_supplement_disease_features.csv",
) -> tuple[Path, Path]:
    disease_path = Path(disease_output)
    tau_path = Path(tau_output)
    disease_path.parent.mkdir(parents=True, exist_ok=True)
    tau_path.parent.mkdir(parents=True, exist_ok=True)
    disease = load_full_supplement_disease_features()
    disease.to_csv(disease_path, index=False)
    tau = build_tau_full_supplement_table()
    tau.to_csv(tau_path, index=False)
    print(f"Saved disease-level full supplement features: {disease_path}")
    print(f"Saved tau + full supplement features: {tau_path}")
    return tau_path, disease_path
