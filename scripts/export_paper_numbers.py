"""Collect every number quoted in the manuscript from results/v2 into one file.

Writes results/v2/paper_numbers.md (human-readable) and results/v2/paper_numbers.tex
(\\newcommand macros), so manuscript values can always be traced to a result file.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results" / "v2"


def f(x, d=2):
    return f"{x:.{d}f}"


def pct(x):
    return f"{100 * x:.1f}\\%"


def main() -> None:
    md, tex = [], []

    def put(key, value, label):
        md.append(f"| {label} | {value} |")
        tex.append(f"\\newcommand{{\\{key}}}{{{value}}}")

    md += ["| Quantity | Value |", "|---|---|"]
    comp = pd.read_csv(R / "spectra_model_comparison.csv")
    ci = pd.read_csv(R / "spectra_neurofoldnet_bootstrap_ci.csv")
    perm = pd.read_csv(R / "permutation_tests_spectra.csv") if (R / "permutation_tests_spectra.csv").exists() else None
    for dye in comp["dye"].unique():
        k = dye.replace("-", "")
        row = comp[(comp.dye == dye) & (comp.model == "NeuroFoldNet")].iloc[0]
        c = ci[ci.dye == dye].set_index("metric")
        for m, short in [("balanced_accuracy", "BA"), ("macro_auc_ovr", "AUC"), ("macro_f1", "Fone")]:
            put(f"nfn{k}{short}", f"{f(row[m + '_mean'])} [{f(c.loc[m, 'ci_low'])}, {f(c.loc[m, 'ci_high'])}]",
                f"NeuroFoldNet {dye} {m} (mean, 95% CI)")
        best = comp[comp.dye == dye].sort_values("balanced_accuracy_mean", ascending=False).iloc[0]
        put(f"best{k}", f"{best['model']} ({f(best['balanced_accuracy_mean'])})", f"Best model {dye} (balanced acc.)")
        if perm is not None:
            p = perm[perm.assay == f"spectra_{dye}"].iloc[0]
            put(f"perm{k}", f"{p['p_value']:.3f}", f"Permutation p {dye} (n={int(p['n_perm'])})")
    bag = pd.read_csv(R / "spectra_preparation_level.csv")
    for _, b in bag[bag.assay == "4-dye panel"].iterrows():
        n = int(b.spectra_per_bag)
        put(f"panel{['', 'One', '', '', '', 'Five', '', '', '', '', 'Ten'][n] if n <= 10 else ('Twenty' if n == 20 else 'Thirty')}",
            f"{f(b.accuracy)} [{f(b.ci_low)}, {f(b.ci_high)}] (n={int(b.n_bags)})", f"4-dye panel accuracy, {n} spectra/dye")
    seed = pd.read_csv(R / "seeding_model_comparison.csv")
    sci = pd.read_csv(R / "seeding_bootstrap_ci.csv")
    sb = seed.sort_values("balanced_accuracy_mean", ascending=False).iloc[0]
    c = sci[sci.model == sb.model].set_index("metric")
    put("seedModel", {"shrinkage_lda": "shrinkage LDA", "logistic_regression": "logistic regression"}
        .get(sb.model, sb.model.replace("_", " ")), "Seeding best model")
    put("seedBA", f"{f(sb.balanced_accuracy_mean)} [{f(c.loc['balanced_accuracy', 'ci_low'])}, "
                  f"{f(c.loc['balanced_accuracy', 'ci_high'])}]", "Seeding balanced accuracy")
    put("seedAUC", f"{f(sb.macro_auc_ovr_mean)} [{f(c.loc['macro_auc_ovr', 'ci_low'])}, "
                   f"{f(c.loc['macro_auc_ovr', 'ci_high'])}]", "Seeding macro AUC")
    if (R / "permutation_tests_seeding.csv").exists():
        sp = pd.read_csv(R / "permutation_tests_seeding.csv").set_index("model").loc[sb.model]
        put("seedPerm", f"{sp['p_value']:.3f}", f"Seeding permutation p (n={int(sp['n_perm'])})")
    f7 = json.loads((R / "fig7_summary.json").read_text())
    cell = pd.read_csv(R / "fig7_cross_genotype_cell_level.csv")
    put("figSevenBA", f"{f(cell.balanced_accuracy.min())}--{f(cell.balanced_accuracy.max())}",
        "Fig 7 cross-genotype balanced accuracy range")
    put("figSevenAUC", f"{f(cell.macro_auc_ovr.min())}--{f(cell.macro_auc_ovr.max())}", "Fig 7 cross-genotype AUC range")
    put("figSevenP", f"{f7['exact_p_value_36_assignments']:.2f}", "Fig 7 exact p (36 assignments)")
    pc = pd.DataFrame(f7["positive_control"])
    put("figSevenCtrlAUC", f"{f(pc.auc.min())}--{f(pc.auc.max())}", "Fig 7 positive control AUC range")
    # Fig 1 morphology
    ms = pd.read_csv(R / "morphology_particle_level_summary.csv", header=[0, 1], index_col=0)
    for meas in ["height", "area", "diameter"]:
        put(f"morph{meas.capitalize()}BA", f(ms.loc[meas, ("balanced_accuracy", "mean")]),
            f"Morphology {meas} single-particle balanced accuracy")
        put(f"morph{meas.capitalize()}AUC", f(ms.loc[meas, ("macro_auc_ovr", "mean")]),
            f"Morphology {meas} single-particle macro AUC")
    mp = pd.read_csv(R / "morphology_preparation_level.csv")
    allm = mp[mp.measure == "all three"].set_index("particles_per_sample")
    for n, word in [(1, "One"), (5, "Five"), (10, "Ten")]:
        put(f"morphAll{word}", pct(allm.loc[n, "accuracy_mean"]), f"Morphology 3 measures, {n} particle(s) each")
    # multi-assay panel
    if (R / "multiassay_panel.csv").exists():
        mpan = pd.read_csv(R / "multiassay_panel.csv")
        for panel, key in [("AFM + 4 dyes", "panelAFMDyes"), ("AFM + 4 dyes + seeding", "panelAll"),
                           ("4-dye spectra + seeding", "panelDyesSeed"),
                           ("AFM morphology (3 measures)", "panelAFM"), ("4-dye spectra", "panelDyes")]:
            for rule, rkey in [("evidence_sum", ""), ("majority_vote", "Vote")]:
                sub = mpan[(mpan.panel == panel) & (mpan.fusion_rule == rule)].set_index("per_source_N")
                for n, word in [(1, "One"), (2, "Two"), (3, "Three"), (5, "Five")]:
                    if n in sub.index:
                        put(f"{key}{rkey}{word}", pct(sub.loc[n, "accuracy_mean"]),
                            f"{panel} [{rule}], {n} measurement(s) per source")
                        if rule == "evidence_sum" and key == "panelAll":
                            put(f"{key}{word}Range", f"{pct(sub.loc[n, 'accuracy_p2.5'])}--"
                                f"{pct(sub.loc[n, 'accuracy_p97.5'])}", f"{panel}, {n}: 2.5-97.5% over draws")
        wide = mpan.pivot_table(index=["panel", "per_source_N"], columns="fusion_rule", values="accuracy_mean")
        diff = wide["evidence_sum"] - wide["majority_vote"]
        put("fusionWins", f"{int((diff > 0).sum())} of {len(diff)}",
            "Configurations where evidence fusion beat majority voting")
        put("fusionWorstGap", f"{abs(100 * diff.min()):.1f}",
            "Largest shortfall of fusion vs voting (percentage points, absolute)")
    if (R / "multiassay_panel_metrics.csv").exists():
        mt = pd.read_csv(R / "multiassay_panel_metrics.csv")
        mt = mt[mt.panel == "AFM + 4 dyes + seeding"]
        for n, word in [(1, "One"), (3, "Three")]:
            s = mt[mt.per_source_N == n].set_index("metric")
            for metric, short in [("macro_auc_ovr", "AUC"), ("balanced_accuracy", "BA"), ("macro_f1", "Fone"),
                                  ("macro_sensitivity", "Sens"), ("macro_specificity", "Spec"),
                                  ("macro_precision", "Prec"), ("log_loss", "LogLoss"), ("brier", "Brier")]:
                put(f"fused{word}{short}", f"{f(s.loc[metric, 'mean'])} [{f(s.loc[metric, 'p2.5'])}, "
                    f"{f(s.loc[metric, 'p97.5'])}]", f"AFM + 4 dyes + seeding, N={n}: {metric} (mean, 2.5-97.5%)")
            for c in ["AD", "DLB", "PSP"]:
                for stat, short in [("sensitivity", "Sens"), ("specificity", "Spec"),
                                    ("precision", "Prec"), ("f1", "Fone")]:
                    put(f"fused{word}{c}{short}", f(s.loc[f"{c}_{stat}", "mean"]),
                        f"AFM + 4 dyes + seeding, N={n}: {c} {stat}")
    if (R / "spectra_preprocessing_sensitivity.csv").exists():
        ps = pd.read_csv(R / "spectra_preprocessing_sensitivity.csv")
        put("prepMax", f(ps.balanced_accuracy.max()), "Preprocessing grid: best single-spectrum balanced accuracy")
    if (R / "fig7_per_cluster_transfer.csv").exists():
        pcl = pd.read_csv(R / "fig7_per_cluster_transfer.csv")
        put("clusterMax", f(pcl.balanced_accuracy.max()), "Fig 7 per-cluster: max balanced accuracy")
        put("clusterN", str(len(pcl)), "Fig 7 per-cluster: number of tests")
    (R / "paper_numbers.md").write_text("\n".join(md), encoding="utf-8")
    (R / "paper_numbers.tex").write_text("\n".join(tex), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
