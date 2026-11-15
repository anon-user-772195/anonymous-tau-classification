"""Supporting statistics for assays that cannot support a per-sample classifier.

  Fig 1D morphology: histograms only (no per-particle IDs). Particle-level values are reconstructed
         at bin resolution from (bin centre, count); pairwise two-sample KS tests and medians.
  Fig 5E-G electrophysiology: unpaired per-recording values; Kruskal-Wallis + pairwise Mann-Whitney,
         Holm-corrected, with epsilon-squared / rank-biserial effect sizes.
  Fig 3B proteolysis: one curve per disease (no replicates) -> descriptive only (area under curve).

Usage:  python scripts/run_assay_statistics.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurofoldnet import assays as A  # noqa: E402
from scripts.run_assay_models import holm  # noqa: E402

OUT = ROOT / "results" / "v2"
CLASSES = list(A.DISEASES)


def morphology() -> pd.DataFrame:
    h = A.load_morphology_histograms()
    rows = []
    for measure, sub in h.groupby("measure"):
        vals = {c: np.repeat(s["bin_centre"].to_numpy(), s["count"].to_numpy()) for c, s in sub.groupby("label")}
        for a, b in itertools.combinations(CLASSES, 2):
            ks = stats.ks_2samp(vals[a], vals[b])
            rows.append({"measure": measure, "comparison": f"{a} vs {b}", "n_a": len(vals[a]), "n_b": len(vals[b]),
                         "median_a": np.median(vals[a]), "median_b": np.median(vals[b]),
                         "ks_D": ks.statistic, "p_value": ks.pvalue})
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df["p_value"].to_numpy())
    df["note"] = "particle values reconstructed at histogram-bin resolution; particles are not independent patients"
    return df


def electrophysiology() -> pd.DataFrame:
    e = A.load_electrophysiology()
    rows = []
    for (panel, tp), sub in e.groupby(["panel", "timepoint"]):
        g = {c: sub.loc[sub["label"] == c, "value"].to_numpy() for c in CLASSES}
        hstat, p = stats.kruskal(*g.values())
        n = sum(len(v) for v in g.values())
        rows.append({"panel": panel, "timepoint": tp, "test": "Kruskal-Wallis", "comparison": "AD/DLB/PSP",
                     "statistic": hstat, "effect_size": (hstat - 2) / (n - 3), "effect": "epsilon_squared",
                     "p_value": p, **{f"n_{c}": len(v) for c, v in g.items()},
                     **{f"median_{c}": float(np.median(v)) for c, v in g.items()}})
        for a, b in itertools.combinations(CLASSES, 2):
            u = stats.mannwhitneyu(g[a], g[b], alternative="two-sided")
            rbc = 1 - 2 * u.statistic / (len(g[a]) * len(g[b]))
            rows.append({"panel": panel, "timepoint": tp, "test": "Mann-Whitney U", "comparison": f"{a} vs {b}",
                         "statistic": u.statistic, "effect_size": rbc, "effect": "rank_biserial", "p_value": u.pvalue})
    df = pd.DataFrame(rows)
    # Holm across the six omnibus tests; post-hoc pairs Holm-corrected within each panel/timepoint
    omni = df["test"] == "Kruskal-Wallis"
    df.loc[omni, "p_holm"] = holm(df.loc[omni, "p_value"].to_numpy())
    for _, idx in df[~omni].groupby(["panel", "timepoint"]).groups.items():
        df.loc[idx, "p_holm"] = holm(df.loc[idx, "p_value"].to_numpy())
    df["correction"] = np.where(omni, "Holm across 6 omnibus tests", "Holm within panel/timepoint (post hoc)")
    return df


def proteolysis() -> pd.DataFrame:
    p = A.load_proteolysis()
    rows = []
    for (label, dose), sub in p.groupby(["label", "dose"]):
        sub = sub.sort_values("minutes")
        rows.append({"label": label, "dose": dose,
                     "auc_0_15min": np.trapezoid(sub["value"], sub["minutes"]),
                     "peak_value": sub["value"].max(), "peak_minutes": sub.loc[sub["value"].idxmax(), "minutes"],
                     "note": "single curve per disease; descriptive only, no inferential test possible"})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    morphology().to_csv(OUT / "stats_fig1_morphology.csv", index=False)
    electrophysiology().to_csv(OUT / "stats_fig5_electrophysiology.csv", index=False)
    proteolysis().to_csv(OUT / "stats_fig3_proteolysis.csv", index=False)
    print("Saved supporting statistics to results/v2/")


if __name__ == "__main__":
    main()
