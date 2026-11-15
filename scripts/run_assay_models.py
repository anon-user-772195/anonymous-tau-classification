"""Primary analysis: assay-specific, leakage-safe classification of AD / DLB / PSP tau oligomers.

  A. Fig 2 dye-emission spectra (ThT, Curcumin, Bis-ANS, FSB): NeuroFoldNet vs five tuned baselines,
     repeated nested group-aware CV (contiguous acquisition blocks as groups), cluster-bootstrap CIs,
     group-level permutation tests, corrected resampled t-tests, preparation-level (bagged) calls,
     a fixed-rule 4-dye panel, and out-of-distribution behaviour on recombinant tau (rTauO).
  B. Fig 4 seeding (6 replicates x 5 antibodies per disease): nested CV + permutation test.

Usage:  python scripts/run_assay_models.py [--repeats 5] [--n-perm 200] [--quick]
Outputs: results/v2/*.csv and results/v2/figures/*.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurofoldnet import assays as A  # noqa: E402
from neurofoldnet import evaluation as E  # noqa: E402
from neurofoldnet import models as M  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
OUT = ROOT / "results" / "v2"
FIG = OUT / "figures"
CLASSES = list(A.DISEASES)
BLOCK = 10
BAG_SIZES = [1, 5, 10, 20, 30]


def summarise(name, res: E.CVResult, y) -> dict:
    rm = res.repeat_metrics(y)
    row = {"model": name}
    for c in rm.columns:
        row[f"{c}_mean"] = rm[c].mean()
        row[f"{c}_sd"] = rm[c].std(ddof=1) if len(rm) > 1 else 0.0
    return row


def bag_scores(y, proba, index, size):
    """Preparation-level calls: mean log-probability over `size` consecutive held-out spectra of one class."""
    bags = []
    for c in range(len(CLASSES)):
        m = np.flatnonzero(y == c)
        order = m[np.argsort(index[m])]
        for b in range(len(order) // size):
            sel = order[b * size:(b + 1) * size]
            bags.append((c, np.log(np.clip(proba[sel], 1e-6, 1)).mean(axis=0)))
    return bags


def bootstrap_acc(correct: np.ndarray, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    draws = rng.choice(correct, size=(n_boot, len(correct)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def run_spectra(args) -> None:
    comparison, ci_tables, perm_rows, ttests, ood_rows, bag_rows = [], [], [], [], [], []
    oof_store = {}
    fig_cm, axes_cm = plt.subplots(1, 4, figsize=(16, 4))
    fig_roc, axes_roc = plt.subplots(1, 4, figsize=(16, 4))
    for d_i, dye in enumerate(A.DYE_BLOCKS):
        t0 = time.time()
        s = A.load_spectra(dye)
        w = A.wavelength_columns(s)
        wl = [int(c[3:]) for c in w]
        X = s[w].to_numpy()
        y = s["label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
        idx = s["spectrum_index"].to_numpy()
        groups = E.contiguous_blocks(s["label"].to_numpy(), idx, BLOCK)

        results = {"NeuroFoldNet": E.repeated_nested_cv("NeuroFoldNet", M.neurofoldnet(wl), None, X, y, groups,
                                                         CLASSES, repeats=args.repeats, seed=1)}
        for name, (est, grid) in M.baseline_grids(wl).items():
            results[name] = E.repeated_nested_cv(name, est, grid, X, y, groups, CLASSES,
                                                 repeats=args.repeats, seed=1)
        for name, res in results.items():
            comparison.append({"dye": dye, **summarise(name, res, y)})
            res.fold_scores.assign(dye=dye, model=name).to_csv(OUT / f"folds_spectra_{dye}_{name}.csv", index=False)
        nfn = results["NeuroFoldNet"]
        n_test = int(nfn.fold_scores["n_test"].mean())
        for name, res in results.items():
            if name == "NeuroFoldNet":
                continue
            t = E.corrected_resampled_ttest(nfn.fold_scores["balanced_accuracy"], res.fold_scores["balanced_accuracy"],
                                            n_train=len(y) - n_test, n_test=n_test)
            ttests.append({"dye": dye, "comparison": f"NeuroFoldNet - {name}", **t})

        ci = E.cluster_bootstrap_ci(y, nfn.mean_proba, groups, CLASSES, n_boot=args.n_boot, seed=2)
        ci_tables.append(ci.assign(dye=dye).reset_index(names="metric"))
        oof_store[dye] = (y, nfn.mean_proba, idx)
        pd.DataFrame({"label": s["label"], "spectrum_index": idx,
                      **{f"p_{c}": nfn.mean_proba[:, i] for i, c in enumerate(CLASSES)}}).to_csv(
            OUT / f"oof_spectra_{dye}_NeuroFoldNet.csv", index=False)

        if args.n_perm > 0:
            p = E.group_permutation_test(M.neurofoldnet(wl), None, X, y, groups, CLASSES, n_perm=args.n_perm, seed=3)
            perm_rows.append({"assay": f"spectra_{dye}", "model": "NeuroFoldNet", **p})

        # out-of-distribution: recombinant tau oligomers, model trained on all disease spectra
        ctrl = A.load_spectra(dye, include_control=True)
        ctrl = ctrl[ctrl["label"] == "rTauO"]
        final = M.neurofoldnet(wl).fit(X, y)
        pc = final.predict_proba(ctrl[w].to_numpy())
        ood_rows.append({"dye": dye, "n_rTauO": len(ctrl), "mean_max_prob_rTauO": pc.max(1).mean(),
                         "mean_max_prob_disease_oof": nfn.mean_proba.max(1).mean(),
                         **{f"called_{c}": float((pc.argmax(1) == i).mean()) for i, c in enumerate(CLASSES)}})

        ConfusionMatrixDisplay(confusion_matrix(y, nfn.mean_proba.argmax(1), normalize="true"),
                               display_labels=CLASSES).plot(ax=axes_cm[d_i], cmap="Blues", colorbar=False,
                                                            values_format=".2f")
        axes_cm[d_i].set_title(f"{dye} (single spectrum)")
        for i, c in enumerate(CLASSES):
            fpr, tpr, _ = roc_curve(y == i, nfn.mean_proba[:, i])
            axes_roc[d_i].plot(fpr, tpr, label=c)
        axes_roc[d_i].plot([0, 1], [0, 1], "k--", lw=0.8)
        axes_roc[d_i].set_title(f"{dye}: one-vs-rest ROC")
        axes_roc[d_i].set_xlabel("1 - specificity")
        axes_roc[d_i].set_ylabel("Sensitivity")
        axes_roc[d_i].legend(fontsize=8)
        print(f"[spectra] {dye} done in {time.time() - t0:.0f}s", flush=True)

    # preparation-level and 4-dye panel
    for size in BAG_SIZES:
        per = {}
        for dye, (y, p, idx) in oof_store.items():
            bags = bag_scores(y, p, idx, size)
            per[dye] = bags
            correct = np.array([np.argmax(sc) == c for c, sc in bags], float)
            lo, hi = bootstrap_acc(correct)
            bag_rows.append({"assay": dye, "spectra_per_bag": size, "n_bags": len(bags),
                             "accuracy": correct.mean(), "ci_low": lo, "ci_high": hi})
        correct = []
        for c in range(len(CLASSES)):
            lists = [[sc for cc, sc in per[d] if cc == c] for d in per]
            n = min(map(len, lists))
            correct += [np.argmax(sum(lst[j] for lst in lists)) == c for j in range(n)]
        correct = np.array(correct, float)
        lo, hi = bootstrap_acc(correct)
        bag_rows.append({"assay": "4-dye panel", "spectra_per_bag": size, "n_bags": len(correct),
                         "accuracy": correct.mean(), "ci_low": lo, "ci_high": hi})

    pd.DataFrame(comparison).to_csv(OUT / "spectra_model_comparison.csv", index=False)
    pd.concat(ci_tables).to_csv(OUT / "spectra_neurofoldnet_bootstrap_ci.csv", index=False)
    pd.DataFrame(ttests).to_csv(OUT / "spectra_corrected_ttests.csv", index=False)
    pd.DataFrame(ood_rows).to_csv(OUT / "spectra_rTauO_out_of_distribution.csv", index=False)
    bag_df = pd.DataFrame(bag_rows)
    bag_df.to_csv(OUT / "spectra_preparation_level.csv", index=False)
    if perm_rows:
        pd.DataFrame(perm_rows).to_csv(OUT / "permutation_tests_spectra.csv", index=False)

    fig_cm.tight_layout(); fig_cm.savefig(FIG / "spectra_confusion_matrices.png", dpi=300); plt.close(fig_cm)
    fig_roc.tight_layout(); fig_roc.savefig(FIG / "spectra_roc_curves.png", dpi=300); plt.close(fig_roc)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for assay, sub in bag_df.groupby("assay", sort=False):
        ax.errorbar(sub["spectra_per_bag"], sub["accuracy"],
                    yerr=[sub["accuracy"] - sub["ci_low"], sub["ci_high"] - sub["accuracy"]],
                    marker="o", capsize=3, lw=2.2 if assay == "4-dye panel" else 1.2, label=assay)
    ax.axhline(1 / 3, color="grey", ls="--", lw=0.8, label="chance")
    ax.set_xlabel("Held-out spectra pooled per call (per dye)")
    ax.set_ylabel("Accuracy (95% bootstrap CI)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "spectra_preparation_level_accuracy.png", dpi=300); plt.close(fig)

    # interpretable descriptors: univariate effect sizes per dye
    rows = []
    for dye in A.DYE_BLOCKS:
        s = A.load_spectra(dye)
        w = A.wavelength_columns(s)
        D = M.SpectralDescriptors([int(c[3:]) for c in w]).transform(s[w].to_numpy())
        for j, name in enumerate(M.SpectralDescriptors.names):
            groups_ = [D[s["label"].to_numpy() == c, j] for c in CLASSES]
            h, p = stats.kruskal(*groups_)
            eps2 = (h - len(CLASSES) + 1) / (len(D) - len(CLASSES))
            rows.append({"dye": dye, "descriptor": name, "kruskal_H": h, "p_value": p, "epsilon_squared": eps2,
                         **{f"median_{c}": float(np.median(g)) for c, g in zip(CLASSES, groups_)}})
    desc = pd.DataFrame(rows)
    desc["p_holm"] = holm(desc["p_value"].to_numpy())
    desc.to_csv(OUT / "spectra_descriptor_statistics.csv", index=False)


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adj = np.empty_like(p, dtype=float)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def run_seeding(args) -> None:
    s = A.load_seeding()
    feats = ["TOMA1", "TOMA2", "TOMA3", "TOMA4", "T22"]
    X = s[feats].to_numpy()
    y = s["label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    groups = np.array([f"{l}_{r}" for l, r in zip(s["label"], s["replicate"])])
    rows, ci_tables, perm_rows = [], [], []
    best = None
    for name, (est, grid) in M.small_n_grids().items():
        res = E.repeated_nested_cv(name, est, grid, X, y, groups, CLASSES, outer_splits=3, inner_splits=2,
                                   repeats=max(args.repeats, 20), seed=11)
        rows.append(summarise(name, res, y))
        ci_tables.append(E.cluster_bootstrap_ci(y, res.mean_proba, groups, CLASSES, n_boot=args.n_boot, seed=12)
                         .assign(model=name).reset_index(names="metric"))
        if args.n_perm > 0:
            p = E.group_permutation_test(est, grid, X, y, groups, CLASSES, n_perm=max(args.n_perm, 1000) if not
                                         args.quick else args.n_perm, outer_splits=3, inner_splits=2, seed=13)
            perm_rows.append({"assay": "seeding_fig4", "model": name, **p})
        if best is None or rows[-1]["balanced_accuracy_mean"] > best[1]:
            best = (res, rows[-1]["balanced_accuracy_mean"], name)
    pd.DataFrame(rows).to_csv(OUT / "seeding_model_comparison.csv", index=False)
    pd.concat(ci_tables).to_csv(OUT / "seeding_bootstrap_ci.csv", index=False)
    if perm_rows:
        pd.DataFrame(perm_rows).to_csv(OUT / "permutation_tests_seeding.csv", index=False)
    stats_rows = []
    for f in feats:
        g = [s.loc[s["label"] == c, f] for c in CLASSES]
        h, p = stats.kruskal(*g)
        stats_rows.append({"antibody": f, "kruskal_H": h, "p_value": p,
                           **{f"mean_{c}": x.mean() for c, x in zip(CLASSES, g)},
                           **{f"sd_{c}": x.std(ddof=1) for c, x in zip(CLASSES, g)}})
    st = pd.DataFrame(stats_rows)
    st["p_holm"] = holm(st["p_value"].to_numpy())
    st.to_csv(OUT / "seeding_antibody_statistics.csv", index=False)
    res, _, name = best
    pd.DataFrame({"label": s["label"], "replicate": s["replicate"],
                  **{f"p_{c}": res.mean_proba[:, i] for i, c in enumerate(CLASSES)}}).to_csv(
        OUT / f"oof_seeding_{name}.csv", index=False)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ConfusionMatrixDisplay(confusion_matrix(y, res.mean_proba.argmax(1)), display_labels=CLASSES).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Fig 4 seeding ({name}), n=18")
    fig.tight_layout(); fig.savefig(FIG / "seeding_confusion_matrix.png", dpi=300); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--quick", action="store_true", help="1 repeat, 5 permutations (smoke test)")
    ap.add_argument("--skip-seeding", action="store_true", help="reuse existing seeding results")
    args = ap.parse_args()
    if args.quick:
        args.repeats, args.n_perm, args.n_boot = 1, 5, 200
    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if not args.skip_seeding:
        run_seeding(args)
        print(f"[seeding] done ({time.time() - t0:.0f}s)", flush=True)
    run_spectra(args)
    (OUT / "run_config_assays.json").write_text(json.dumps(vars(args), indent=2))
    print(f"All assay models done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
