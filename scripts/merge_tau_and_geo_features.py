from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neurofoldnet.tau_data import load_tau_feature_table


DEFAULT_TAU_OUTPUT = PROJECT_ROOT / "data" / "processed" / "tau_features.csv"
DEFAULT_GEO_INPUT = PROJECT_ROOT / "data" / "processed" / "figure7_geo_sample_features.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "tau_plus_figure7_features.csv"


def numeric_geo_columns(frame: pd.DataFrame) -> list[str]:
    blocked = {
        "n_cells",
        "is_untreated_control",
    }
    return [
        col
        for col in frame.columns
        if col not in blocked
        and col.startswith(("geo_expr_", "geo_module_", "geo_pca_global_"))
        and pd.api.types.is_numeric_dtype(frame[col])
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge tau polymorph features with Figure 7 GEO features.")
    parser.add_argument("--tau-input", type=Path, default=None, help="Optional precomputed tau CSV.")
    parser.add_argument("--tau-workbook", type=Path, default=None, help="Optional Nature supplement workbook.")
    parser.add_argument("--geo-input", type=Path, default=DEFAULT_GEO_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.tau_input:
        tau = pd.read_csv(args.tau_input)
        print(f"Loaded tau features from CSV: {args.tau_input}")
    else:
        tau = load_tau_feature_table(args.tau_workbook)
        DEFAULT_TAU_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        tau.to_csv(DEFAULT_TAU_OUTPUT, index=False)
        print(f"Saved reconstructed tau features: {DEFAULT_TAU_OUTPUT}")

    if not args.geo_input.exists():
        raise FileNotFoundError(
            f"Missing Figure 7 GEO feature table: {args.geo_input}\n"
            "Run: python scripts/prepare_figure7_geo_features.py"
        )
    geo = pd.read_csv(args.geo_input)
    print(f"Loaded GEO features: {args.geo_input}, shape={geo.shape}")

    required_tau = {"sample_id", "group", "disease"}
    required_geo = {"sample_id", "disease"}
    missing_tau = required_tau.difference(tau.columns)
    missing_geo = required_geo.difference(geo.columns)
    if missing_tau:
        raise ValueError(f"Tau table is missing required columns: {sorted(missing_tau)}")
    if missing_geo:
        raise ValueError(f"GEO table is missing required columns: {sorted(missing_geo)}")

    geo_cols = numeric_geo_columns(geo)
    if not geo_cols:
        raise ValueError("No numeric GEO feature columns were found.")

    shared_sample_ids = set(tau["sample_id"].astype(str)).intersection(set(geo["sample_id"].astype(str)))
    if shared_sample_ids:
        print(f"Found {len(shared_sample_ids)} exact sample ID matches; merging by sample_id.")
        geo_subset = geo[["sample_id", "disease", *geo_cols]].copy()
        merged = tau.merge(geo_subset, on=["sample_id", "disease"], how="left")
        merged["geo_auxiliary_level"] = "sample_id"
    else:
        print("No exact tau/GEO sample ID matches found.")
        print("Using disease-level GEO summary features. This is auxiliary disease-level information, not a one-to-one sample merge.")
        disease_summary = geo.groupby("disease", as_index=False)[geo_cols].mean()
        disease_summary = disease_summary.rename(columns={col: f"{col}_disease_mean" for col in geo_cols})
        merged = tau.merge(disease_summary, on="disease", how="left")
        merged["geo_auxiliary_level"] = "disease_mean"

    if merged.filter(regex=r"^geo_").isna().all(axis=None):
        raise ValueError("Merge produced only missing GEO values; check disease labels and GEO preprocessing.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print("=" * 80)
    print(f"Saved merged feature table: {args.output}")
    print(f"Output shape: {merged.shape}")
    print("Disease distribution:")
    print(merged["disease"].value_counts().to_string())
    print("GEO auxiliary levels:")
    print(merged["geo_auxiliary_level"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
