"""Write every Supplementary Table of the manuscript from the archived result files.

Output: results/v2/supplementary_tables.tex (input by manuscript/supplementary.tex).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results" / "v2"
DYES = ["ThT", "Curcumin", "Bis-ANS", "FSB"]
MODEL_NAMES = {
    "NeuroFoldNet": "NeuroFoldNet expert", "logistic_regression": "Logistic regression",
    "shrinkage_lda": "Shrinkage LDA", "svm_rbf": "RBF-SVM", "random_forest": "Random forest",
    "descriptors_only_lr": "Descriptor-only LR",
}


def esc(s) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def f2(x) -> str:
    return "--" if pd.isna(x) else f"{x:.2f}"


def f3(x) -> str:
    return "--" if pd.isna(x) else f"{x:.3f}"


def pfmt(p) -> str:
    if pd.isna(p):
        return "--"
    return f"{p:.3f}" if p >= 0.001 else f"{p:.1e}".replace("e-0", r"$\times10^{-").replace("e-", r"$\times10^{-") + "}$"


def table(label: str, caption: str, header: list[str], rows: list[list[str]], spec: str | None = None) -> str:
    spec = spec or "l" + "c" * (len(header) - 1)
    out = [r"\begin{table}[!htbp]", r"\centering\scriptsize", rf"\caption{{{caption}}}\label{{{label}}}",
           rf"\begin{{tabular}}{{{spec}}}", r"\toprule", " & ".join(rf"\textbf{{{h}}}" for h in header) + r" \\",
           r"\midrule"]
    out += [" & ".join(r) + r" \\" for r in rows]
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(out)


def main() -> None:
    parts: list[str] = []

    # S1 assays
    parts.append(table(
        "tab:s_assays", "Assays in the source data, units of observation and their role in this study.",
        ["Source", "Content as released", "$n$ per disease", "Role"],
        [["Fig.~1D", "AFM histograms of oligomer height, area, diameter", "490--966 particles", "Expert + fusion"],
         ["Fig.~2A", "Emission spectra (416--706\\,nm), ThT, Curcumin, Bis-ANS, FSB; rTauO", "$\\approx$150 per dye", "Expert + fusion"],
         ["Fig.~4D", "Seeding with TOMA1--4 and T22 antibodies", "6 replicates", "Expert + fusion"],
         ["GSE277105", "Neurons (C57BL/6, MAPT) treated with AD/DLB/PSP aBDTOs, plus untreated", "1 library per genotype", "Held-out genotype"],
         ["Fig.~5E--G", "Per-recording electrophysiology (12\\,h, 24\\,h)", "5--192 recordings", "Rank tests"],
         ["Fig.~3B, 6", "Proteolysis curves; LTP summaries", "1 curve or $n\\le6$", "Descriptive"]],
        "p{1.6cm}p{6.2cm}p{2.4cm}p{2.4cm}"))

    # S2 external search
    ext = pd.read_csv(R / "external_dataset_search.csv")
    parts.append(table(
        "tab:s_external", "Feature-by-feature search for independent external datasets (searched November 2026).",
        ["Feature", "Source", "Diseases", "Why not usable"],
        [[esc(r.feature_type), esc(r.source), esc(r.diseases), esc(r.reason)] for r in ext.itertuples()],
        "p{2.0cm}p{4.6cm}p{2.0cm}p{4.6cm}"))

    # S3 spectral model comparison
    comp = pd.read_csv(R / "spectra_model_comparison.csv")
    rows = []
    for dye in DYES:
        for m in MODEL_NAMES:
            r = comp[(comp.dye == dye) & (comp.model == m)].iloc[0]
            rows.append([dye if m == "NeuroFoldNet" else "", MODEL_NAMES[m],
                         f"{f2(r.balanced_accuracy_mean)} $\\pm$ {f2(r.balanced_accuracy_sd)}",
                         f"{f2(r.macro_auc_ovr_mean)} $\\pm$ {f2(r.macro_auc_ovr_sd)}",
                         f"{f2(r.macro_f1_mean)} $\\pm$ {f2(r.macro_f1_sd)}", f3(r.log_loss_mean), f3(r.brier_mean)])
    parts.append(table(
        "tab:s_spectra_models",
        "Single-spectrum performance of the NeuroFoldNet spectral expert and five tuned baselines "
        "(mean $\\pm$ SD over 5 repeats of nested group-aware cross-validation).",
        ["Dye", "Model", "Balanced acc.", "Macro-AUC", "Macro-F1", "Log loss", "Brier"], rows, "llccccc"))

    # S4 per-class metrics of the spectral expert
    rows = []
    for dye in DYES:
        r = comp[(comp.dye == dye) & (comp.model == "NeuroFoldNet")].iloc[0]
        for c in ["AD", "DLB", "PSP"]:
            rows.append([dye if c == "AD" else "", c, f2(r[f"{c}_sensitivity_mean"]), f2(r[f"{c}_specificity_mean"]),
                         f2(r[f"{c}_precision_mean"]), f2(r[f"{c}_f1_mean"])])
    parts.append(table(
        "tab:s_spectra_perclass", "Per-class sensitivity (recall), specificity, precision and F1 of the spectral expert.",
        ["Dye", "Class", "Sensitivity", "Specificity", "Precision", "F1"], rows))

    # S5 corrected t-tests
    tt = pd.read_csv(R / "spectra_corrected_ttests.csv")
    parts.append(table(
        "tab:s_ttests", "Spectral expert versus each baseline: Nadeau--Bengio corrected resampled $t$-tests on balanced accuracy.",
        ["Dye", "Comparison", "Mean difference", "$t$", "$p$"],
        [[r.dye, "Expert $-$ " + MODEL_NAMES[r.comparison.replace("NeuroFoldNet - ", "")],
          f"{r.mean_diff:+.3f}", f"{r.t:.2f}", f3(r.p_value)] for r in tt.itertuples()], "llccc"))

    # S6 preprocessing grid
    pg = pd.read_csv(R / "spectra_preprocessing_sensitivity.csv")
    piv = pg.pivot_table(index=["dye", "preprocessing"], columns="classifier", values="balanced_accuracy")
    cols = list(piv.columns)
    rows = [[d if i == 0 else "", esc(p)] + [f2(piv.loc[(d, p), c]) for c in cols]
            for d in DYES for i, p in enumerate(piv.loc[d].index)]
    parts.append(table(
        "tab:s_preproc", "Preprocessing $\\times$ classifier sensitivity grid: single-spectrum balanced accuracy.",
        ["Dye", "Preprocessing"] + [esc(c).replace("\\_", " ") for c in cols], rows))

    # S7 descriptor statistics (all)
    ds = pd.read_csv(R / "spectra_descriptor_statistics.csv")
    parts.append(table(
        "tab:s_descriptors", "Spectral shape descriptors: medians per disease and Kruskal--Wallis tests (Holm-corrected within dye).",
        ["Dye", "Descriptor", "AD", "DLB", "PSP", "$\\varepsilon^2$", "Holm $p$"],
        [[r.dye, esc(r.descriptor), f"{r.median_AD:.3g}", f"{r.median_DLB:.3g}", f"{r.median_PSP:.3g}",
          f3(r.epsilon_squared), pfmt(r.p_holm)] for r in ds.itertuples()], "llccccc"))

    # S8 seeding
    sm = pd.read_csv(R / "seeding_model_comparison.csv")
    sa = pd.read_csv(R / "seeding_antibody_statistics.csv")
    rows = [[esc(MODEL_NAMES.get(r.model, r.model)), f"{f2(r.balanced_accuracy_mean)} $\\pm$ {f2(r.balanced_accuracy_sd)}",
             f"{f2(r.macro_auc_ovr_mean)} $\\pm$ {f2(r.macro_auc_ovr_sd)}", f"{f2(r.macro_f1_mean)} $\\pm$ {f2(r.macro_f1_sd)}"]
            for r in sm.itertuples()]
    parts.append(table("tab:s_seeding_models", "Seeding classifiers (mean $\\pm$ SD over 20 repeats of 3-fold cross-validation).",
                       ["Model", "Balanced acc.", "Macro-AUC", "Macro-F1"], rows))
    parts.append(table(
        "tab:s_seeding_ab", "Seeding readout per antibody: mean (SD) per disease and Kruskal--Wallis tests (Holm-corrected).",
        ["Antibody", "AD", "DLB", "PSP", "$H$", "Holm $p$"],
        [[esc(r.antibody), f"{r.mean_AD:.3f} ({r.sd_AD:.3f})", f"{r.mean_DLB:.3f} ({r.sd_DLB:.3f})",
          f"{r.mean_PSP:.3f} ({r.sd_PSP:.3f})", f"{r.kruskal_H:.2f}", pfmt(r.p_holm)] for r in sa.itertuples()]))

    # S10 out of distribution
    od = pd.read_csv(R / "spectra_rTauO_out_of_distribution.csv")
    parts.append(table(
        "tab:s_ood", "Recombinant tau oligomers (rTauO, out of distribution): mean maximum class probability and call distribution.",
        ["Dye", "$n$", "Max prob. rTauO", "Max prob. disease (OOF)", "Called AD", "Called DLB", "Called PSP"],
        [[r.dye, str(r.n_rTauO), f2(r.mean_max_prob_rTauO), f2(r.mean_max_prob_disease_oof), f2(r.called_AD),
          f2(r.called_DLB), f2(r.called_PSP)] for r in od.itertuples()]))

    # S11 fusion, all panels and sizes
    mp = pd.read_csv(R / "multiassay_panel.csv")
    wide = mp.pivot_table(index=["panel", "per_source_N"], columns="fusion_rule", values="accuracy_mean").reset_index()
    order = ["AFM morphology (3 measures)", "4-dye spectra", "AFM morphology (3 measures) + seeding",
             "4-dye spectra + seeding", "AFM + 4 dyes", "AFM + 4 dyes + seeding"]
    rows = []
    for p in order:
        sub = wide[wide.panel == p].sort_values("per_source_N")
        for i, r in enumerate(sub.itertuples()):
            rows.append([esc(p) if i == 0 else "", str(r.per_source_N), f"{100 * r.evidence_sum:.1f}",
                         f"{100 * r.majority_vote:.1f}", f"{100 * (r.evidence_sum - r.majority_vote):+.1f}"])
    parts.append(table(
        "tab:s_fusion_all", "Preparation-level accuracy (\\%) of NeuroFoldNet evidence fusion versus majority voting for every "
        "assay panel and number of held-out measurements per source (500 draws).",
        ["Panel", "$N$", "Fusion", "Majority vote", "Difference"], rows, "lcccc"))

    # S12 Fig 7 sensitivity + control + clusters
    sg = pd.read_csv(R / "fig7_sensitivity_grid.csv")
    pc = pd.read_csv(R / "fig7_positive_control_treated_vs_untreated.csv")
    rows = [["UT-centred" if r.ut_centring else "Not centred", str(r.n_hvg), f"{r.train_genotype}$\\to${r.test_genotype}",
             f2(r.balanced_accuracy), f2(r.macro_auc_ovr), f"{r.library_calls_correct}/3"] for r in sg.itertuples()]
    rows += [["Positive control", "--", f"{r.train_genotype}$\\to${r.test_genotype}", f2(r.balanced_accuracy), f2(r.auc), "--"]
             for r in pc.itertuples()]
    parts.append(table(
        "tab:s_fig7", "GSE277105 cross-genotype transfer: pre-declared sensitivity grid (disease identity, chance balanced "
        "accuracy 0.33) and treated-versus-untreated positive control.",
        ["Setting", "HVGs", "Direction", "Balanced acc.", "AUC", "Libraries correct"], rows))

    # S13 electrophysiology
    ep = pd.read_csv(R / "stats_fig5_electrophysiology.csv")
    om = ep[ep.test == "Kruskal-Wallis"]
    parts.append(table(
        "tab:s_ephys", "Fig.~5 electrophysiology: Kruskal--Wallis omnibus tests (Holm-corrected across six tests).",
        ["Panel", "Time", "$H$", "$\\varepsilon^2$", "$n$ (AD/DLB/PSP)", "Holm $p$"],
        [[r.panel, esc(r.timepoint), f"{r.statistic:.2f}", f3(r.effect_size),
          f"{int(r.n_AD)}/{int(r.n_DLB)}/{int(r.n_PSP)}", pfmt(r.p_holm)] for r in om.itertuples()]))

    # S14 fusion rules (split-half)
    if (R / "fusion_rules_comparison.csv").exists():
        fr = pd.read_csv(R / "fusion_rules_comparison.csv")
        rows = []
        for (panel, n), sub in fr.groupby(["panel", "per_source_N"], sort=False):
            for i, r in enumerate(sub.to_dict("records")):
                rows.append([esc(panel) if i == 0 else "", str(n) if i == 0 else "", esc(r["rule"]),
                             f"{100 * r['accuracy_mean']:.1f} [{100 * r['accuracy_p2.5']:.1f}, {100 * r['accuracy_p97.5']:.1f}]",
                             f2(r["macro_auc_mean"]), f3(r["log_loss_mean"]), f3(r["ece_mean"])])
        parts.append(table(
            "tab:s_rules", "Fusion rules compared on identical held-out preparations (split-half protocol, 100 random splits; "
            "accuracy in \\%, mean with 2.5--97.5\\% range). The learned fusion is fitted on preparations from one half of the "
            "held-out measurements and evaluated on the other half.",
            ["Panel", "$N$", "Rule", "Accuracy", "Macro-AUC", "Log loss", "ECE"], rows, "llp{4.2cm}cccc"))
    # S15 ablation
    if (R / "fusion_ablation.csv").exists():
        ab = pd.read_csv(R / "fusion_ablation.csv")
        rows = []
        for n, sub in ab.groupby("per_source_N", sort=False):
            for i, r in enumerate(sub.itertuples()):
                rows.append([str(n) if i == 0 else "", esc(r.removed), f"{100 * r.accuracy_mean:.1f}",
                             f"{100 * r.accuracy_drop:+.1f}", f2(r.macro_auc_mean), f3(r.log_loss_mean)])
        parts.append(table(
            "tab:s_ablation", "Ablation of NeuroFoldNet fusion (AFM + 4 dyes + seeding): accuracy (\\%) and accuracy drop "
            "(percentage points) when one source or one assay family is removed (100 draws).",
            ["$N$", "Removed", "Accuracy", "Drop", "Macro-AUC", "Log loss"], rows, "llcccc"))
    # S16 calibration
    if (R / "fusion_calibration.csv").exists():
        cal = pd.read_csv(R / "fusion_calibration.csv")
        parts.append(table(
            "tab:s_calibration", "Calibration of the fused posterior and of each single expert (one measurement): accuracy, "
            "mean top-label confidence, expected calibration error (ECE, 10 bins) and Brier score.",
            ["Setting", "$N$", "Accuracy", "Mean confidence", "ECE", "Brier"],
            [[esc(r.setting), str(r.per_source_N), f2(r.accuracy), f2(r.mean_confidence), f3(r.ece), f3(r.brier)]
             for r in cal.itertuples()], "lccccc"))

    (R / "supplementary_tables.tex").write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {len(parts)} tables to {R / 'supplementary_tables.tex'}")


if __name__ == "__main__":
    main()
