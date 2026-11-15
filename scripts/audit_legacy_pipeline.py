"""Reproducible audit of the legacy 9-feature table (legacy/data_processed/tau_plus_one_fig7_nonleaking_feature.csv).

Checks, each printed with evidence:
  1. Fig 1 'height/area/diameter' values are histogram bin centres / modal values, not measurements.
  2. Fig 3 'replicates' are proteolysis time points (0, 1, 5, 15 min).
  3. Fig 4 'replicates' are different antibodies of replicate column 1.
  4. Rows are joined across figures by row position, not by any shared sample ID.
  5. The Figure 7 feature is one constant value in every row.
  6. A single class-constant column (diameter) identifies the disease perfectly, explaining the 95%.

Usage:  python scripts/audit_legacy_pipeline.py   -> results/v2/legacy_audit.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurofoldnet import assays as A  # noqa: E402

LEGACY = ROOT / "legacy" / "data_processed" / "tau_plus_one_fig7_nonleaking_feature.csv"
OUT = ROOT / "results" / "v2" / "legacy_audit.txt"


def main() -> None:
    t = pd.read_csv(LEGACY)
    wb = pd.read_excel(A.WORKBOOK, sheet_name=None, header=None)
    f1, f3, f4 = wb["Fig 1"], wb["Fig 3"], wb["Fig 4"]
    lines = [f"Legacy table: {LEGACY.name}  shape={t.shape}", ""]

    ad_bins = pd.to_numeric(f1.iloc[3:8, 0]).round(3).tolist()
    lines += ["1. Fig 1 morphology",
              f"   Workbook Fig 1D col 0 header='{f1.iat[2, 0]}', next col header='{f1.iat[2, 1]}' (histogram).",
              f"   AD height bin centres rows 3-7: {ad_bins}",
              f"   Legacy AD height values:        {t.loc[t.disease == 'AD', 'height'].round(3).tolist()}",
              f"   Fig 1E (modal values) repeats one number per disease: AD diameter rows 3-6 = "
              f"{pd.to_numeric(f1.iloc[3:7, 29]).tolist()}",
              f"   Legacy diameter unique values by class: "
              f"{t.groupby('disease')['diameter'].unique().apply(list).to_dict()}", ""]
    lines += ["2. Fig 3 proteolysis",
              f"   Workbook Fig 3 col 0 rows 3-6 (time, min): {pd.to_numeric(f3.iloc[3:7, 0]).tolist()}",
              f"   Legacy AD proteo_resist_05 rep0-3: {t.loc[t.disease == 'AD', 'proteo_resist_05'].tolist()[:4]}"
              f"  == the AD time course at 0/1/5/15 min", ""]
    lines += ["3. Fig 4 seeding",
              f"   Workbook Fig 4 rows 2-6 col 0 (antibodies): {['' .join(str(v).split()) for v in f4.iloc[2:7, 0]]}",
              f"   Workbook AD replicate column 1, rows 3-6: {pd.to_numeric(f4.iloc[3:7, 1]).round(6).tolist()}",
              f"   Legacy AD seed_density rep0-3:            {t.loc[t.disease == 'AD', 'seed_density'].round(6).tolist()[:4]}",
              "   -> TOMA2, TOMA3, TOMA4, T22 of one replicate were treated as four replicates.", ""]
    lines += ["4. Cross-figure join",
              "   legacy/neurofoldnet/tau_data.py assigns replicate = groupby('disease').cumcount() in each figure",
              "   and outer-merges on (disease, replicate): row k of Fig 1 is glued to row k of Figs 3/4/5.",
              f"   Values imputed with a pooled median appear in all classes: "
              f"{sorted(t.loc[t.proteo_resist_05 == 22.7945, 'sample_id'].tolist())}", ""]
    f7 = t["fig7_global_tau_neuronal_response"]
    lines += ["5. Figure 7 feature",
              f"   unique values: {f7.unique().tolist()}  (n_unique={f7.nunique()}) -> constant, removed by VarianceThreshold;",
              "   it cannot carry information and cannot have 14% feature importance.", ""]
    acc = (t.groupby("diameter")["disease"].agg(lambda s: s.mode()[0]).reindex(t["diameter"]).to_numpy()
           == t["disease"].to_numpy()).mean()
    lines += ["6. Label leakage",
              f"   Lookup rule 'diameter -> disease' classifies {acc:.0%} of legacy rows correctly with no learning.",
              "   The legacy 95% therefore reflects class-constant columns, not tau-polymorph signal.", ""]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
