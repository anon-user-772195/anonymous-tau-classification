"""LaTeX macros for the audit/robustness results (results/v2/robustness_*.csv) used by main_aim_rev1.tex.

Usage: python scripts/export_robustness_numbers.py   -> results/v2/robustness_numbers.tex
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

R = Path(__file__).resolve().parents[1] / "results" / "v2"
out = []


def mac(name, val):
    out.append(f"\\newcommand{{\\{name}}}{{{val}}}")


def p3(p):
    return f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"


# nested seeding selection
s = pd.read_csv(R / "robustness_seeding_nested.csv").set_index(["analysis", "metric"])["mean"]
mac("rbSeedNestBA", f"{s[('family_selected_in_inner_cv', 'balanced_accuracy')]:.2f}")
mac("rbSeedNestAUC", f"{s[('family_selected_in_inner_cv', 'macro_auc_ovr')]:.2f}")
mac("rbSeedNestP", p3(s[("permutation_p", "balanced_accuracy")]))
mac("rbSeedNestLRShare", f"{100 * s[('inner_choice_frequency', 'LogisticRegression')]:.0f}\\%")
perm = pd.read_csv(R / "permutation_tests_seeding.csv").set_index("model")["p_value"]
mac("rbSeedLRP", p3(perm["logistic_regression"]))
mac("rbSeedLDAP", p3(perm["shrinkage_lda"]))

# seeding expert -> fusion sensitivity
f = pd.read_csv(R / "robustness_seeding_fusion.csv")
for exp_key, tag in [("primary", "Prim"), ("inner", "Nest")]:
    sub = f[f["seeding_expert"].str.contains(exp_key)]
    for panel, ptag in [("AFM + dyes + seeding", "All"), ("seeding only", "Seed")]:
        for n, ntag in [(1, "One"), (3, "Three")]:
            v = sub[(sub["panel"] == panel) & (sub["measurements_per_source"] == n)]["accuracy_mean"].iloc[0]
            mac(f"rbFus{tag}{ptag}{ntag}", f"{100 * v:.1f}\\%")

# drift
d = pd.read_csv(R / "robustness_drift.csv")
mac("rbDriftAUCmin", f"{d['auc_first_vs_second_half'].min():.2f}")
mac("rbDriftAUCmax", f"{d['auc_first_vs_second_half'].max():.2f}")
mac("rbDriftAUCmean", f"{d['auc_first_vs_second_half'].mean():.2f}")
mac("rbDriftNsig", str(int((d["p_value"] <= 0.05).sum())))
mac("rbDriftMinHolm", f"{d['p_holm'].min():.2f}")
mac("rbDriftFisherP", p3(stats.combine_pvalues(d["p_value"], method="fisher").pvalue))

# block size
b = pd.read_csv(R / "robustness_block_size.csv").pivot(index="block_size", columns="dye",
                                                        values="balanced_accuracy_mean")
for dye, tag in [("ThT", "ThT"), ("Curcumin", "Curcumin"), ("Bis-ANS", "BisANS"), ("FSB", "FSB")]:
    for blk, btag in [(1, "One"), (10, "Ten"), (20, "Twenty"), (50, "Fifty")]:
        mac(f"rbBlk{tag}{btag}", f"{b.loc[blk, dye]:.2f}")

(R / "robustness_numbers.tex").write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
