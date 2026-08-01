"""Create a tau feature table with one auxiliary Figure 7 feature.

The added feature is disease-level because GSE277105 does not provide tau
replicate-matched sample IDs. This is intended as a sensitivity analysis, not a
matched multimodal classifier.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_FEATURE = "geo_module_tau_neuronal_vs_untreated"


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
        default=Path("data/processed/tau_plus_one_fig7_feature.csv"),
    )
    args = parser.parse_args()

    tau = pd.read_csv(args.tau_input)
    geo = pd.read_csv(args.geo_input)
    if args.geo_feature not in geo.columns:
        raise ValueError(
            f"Missing Fig 7 feature '{args.geo_feature}'. Available geo_module columns: "
            f"{[c for c in geo.columns if c.startswith('geo_module_')]}"
        )

    geo = geo[geo["disease"].isin(["AD", "DLB", "PSP"])].copy()
    one_feature = (
        geo.groupby("disease", as_index=False)[args.geo_feature]
        .mean()
        .rename(columns={args.geo_feature: f"{args.geo_feature}_one_fig7_aux"})
    )
    merged = tau.merge(one_feature, on="disease", how="left")
    merged["one_fig7_feature_name"] = args.geo_feature
    merged["one_fig7_auxiliary_level"] = "disease_mean_across_genotypes"
    merged["one_fig7_note"] = "single unmatched Figure 7 auxiliary feature"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"Saved tau + one Fig 7 feature table: {args.output}")
    print(f"Shape: {merged.shape}")
    print("Added feature values by disease:")
    print(one_feature.to_string(index=False))


if __name__ == "__main__":
    main()
