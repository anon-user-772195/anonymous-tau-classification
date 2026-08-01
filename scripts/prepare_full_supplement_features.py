from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neurofoldnet.full_supplement import build_tau_full_supplement_table, load_full_supplement_disease_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare disease-level features from all numeric supplement points.")
    parser.add_argument("--tau-workbook", type=Path, default=None)
    parser.add_argument("--disease-output", type=Path, default=PROJECT_ROOT / "data" / "processed" / "full_supplement_disease_features.csv")
    parser.add_argument("--tau-output", type=Path, default=PROJECT_ROOT / "data" / "processed" / "tau_full_supplement_features.csv")
    args = parser.parse_args()

    disease = load_full_supplement_disease_features(args.tau_workbook)
    tau = build_tau_full_supplement_table(args.tau_workbook)

    args.disease_output.parent.mkdir(parents=True, exist_ok=True)
    args.tau_output.parent.mkdir(parents=True, exist_ok=True)
    disease.to_csv(args.disease_output, index=False)
    tau.to_csv(args.tau_output, index=False)

    print("=" * 80)
    print(f"Saved full supplement disease summaries: {args.disease_output}")
    print(f"Shape: {disease.shape}")
    print(f"Saved tau + full supplement features: {args.tau_output}")
    print(f"Shape: {tau.shape}")
    print("NOTE: full supplement features are disease-level summaries, not matched sample-level measurements.")


if __name__ == "__main__":
    main()
