"""Create a tau feature table with one non-leaking Figure 7 context feature.

Unlike disease-level GEO joins, this feature is computed once across all
treated Figure 7 profiles and assigned identically to every tau feature vector.
It is therefore available without knowing a sample's disease label. It is a
Figure 7 context feature, not a matched transcriptomic measurement.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_FEATURE = "geo_module_tau_neuronal_vs_untreated"
OUTPUT_COLUMN = "fig7_global_tau_neuronal_response"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau-input", type=Path, default=Path("data/processed/tau_features.csv"))
    parser.add_argument(
        "--geo-input",
        type=Path,
        default=Path("data/processed/figure7_geo_sample_features.csv"),
    )
    parser.add_argument("--geo-feature", default=DEFAULT_FEATURE)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/tau_plus_one_fig7_nonleaking_feature.csv"),
    )
    args = parser.parse_args()

    tau = pd.read_csv(args.tau_input)
    geo = pd.read_csv(args.geo_input)
    if args.geo_feature not in geo.columns:
        raise ValueError(f"Missing Figure 7 feature: {args.geo_feature}")

    treated = geo[geo["disease"].isin(["AD", "DLB", "PSP"])].copy()
    global_value = float(pd.to_numeric(treated[args.geo_feature], errors="coerce").mean())
    merged = tau.copy()
    merged[OUTPUT_COLUMN] = global_value
    merged["fig7_context_feature_name"] = args.geo_feature
    merged["fig7_context_level"] = "global_treated_profile_mean"
    merged["fig7_context_note"] = "non-leaking disease-blind Figure 7 context feature"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"Saved tau + non-leaking one Fig 7 feature table: {args.output}")
    print(f"Shape: {merged.shape}")
    print(f"{OUTPUT_COLUMN} = {global_value:.12f}")
    print("This value is identical for all rows and was computed without using disease labels.")


if __name__ == "__main__":
    main()
