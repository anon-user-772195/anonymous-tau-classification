"""Fig 1D AFM morphology: particle-level and preparation-level classification.

The workbook releases per-disease histograms (bin centre, count) for oligomer height, area and diameter.
Each disease uses its own bin grid, so raw bin centres would let a classifier recognise the grid (leakage).
We therefore reconstruct particles by drawing each value uniformly within its bin (seeded, repeated).
Height, area and diameter are separate histograms (marginals), so they cannot be paired within a particle.

  Particle level: one measure per particle, class-conditional KDE on log values, stratified 5-fold CV.
  Preparation level: a held-out sample of N particles per measure; per-measure mean log-likelihoods
  are summed over the three measures (valid at sample level because particles are independent draws).

All particles of a disease come from that disease's oligomer preparation(s); results describe separability
of preparations, not of patients.

Usage:  python scripts/run_morphology_models.py [--repeats 5]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurofoldnet import assays as A  # noqa: E402

OUT = ROOT / "results" / "v2"
FIG = OUT / "figures"
CLASSES = list(A.DISEASES)
MEASURES = ["height", "area", "diameter"]
BAG_SIZES = [1, 5, 10, 20, 30]


def reconstruct(h: pd.DataFrame, rng) -> dict[tuple[str, str], np.ndarray]:
    out = {}
    for (label, measure), s in h.groupby(["label", "measure"]):
        out[(label, measure)] = A.reconstruct_log_particles(s["bin_centre"].to_numpy(), s["count"].to_numpy(), rng)
    return out


def fold_densities(train: dict[str, np.ndarray]):
    return {c: gaussian_kde(v) for c, v in train.items()}


def run(repeats: int):
    h = A.load_morphology_histograms()
    part_rows, bag_rows = [], []
    for r in range(repeats):
        rng = np.random.default_rng(100 + r)
        data = reconstruct(h, rng)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=r)
        # per-measure particle-level CV, store test log-likelihoods per fold for bagging
        test_ll = {m: [] for m in MEASURES}  # list over folds of dict class -> array (n_test x 3)
        for m in MEASURES:
            x = np.concatenate([data[(c, m)] for c in CLASSES])
            y = np.concatenate([np.full(len(data[(c, m)]), i) for i, c in enumerate(CLASSES)])
            ll_all = np.zeros((len(y), 3))
            folds = list(skf.split(x, y))
            for f, (tr, te) in enumerate(folds):
                kdes = fold_densities({c: x[tr][y[tr] == i] for i, c in enumerate(CLASSES)})
                ll = np.column_stack([np.log(kdes[c](x[te]) + 1e-12) for c in CLASSES])
                ll_all[te] = ll
                test_ll[m].append({i: ll[y[te] == i] for i in range(3)})
            proba = np.exp(ll_all - ll_all.max(1, keepdims=True))
            proba /= proba.sum(1, keepdims=True)
            if r == 0:
                pd.DataFrame({"label": [CLASSES[i] for i in y],
                              **{f"p_{c}": proba[:, i] for i, c in enumerate(CLASSES)}}).to_csv(
                    OUT / f"oof_morphology_{m}.csv", index=False)
            part_rows.append({"repeat": r, "measure": m, "n_particles": len(y),
                              "balanced_accuracy": balanced_accuracy_score(y, ll_all.argmax(1)),
                              "macro_auc_ovr": roc_auc_score(y, proba, multi_class="ovr")})
        # preparation-level: within each fold, draw disjoint held-out samples of N particles per class
        for size in BAG_SIZES:
            correct = {m: [] for m in MEASURES + ["all three"]}
            for f in range(5):
                for i in range(3):
                    lls = {m: rng.permutation(test_ll[m][f][i]) for m in MEASURES}
                    n_bags = min(len(v) for v in lls.values()) // size
                    for b in range(n_bags):
                        scores = {m: lls[m][b * size:(b + 1) * size].mean(0) for m in MEASURES}
                        for m in MEASURES:
                            correct[m].append(scores[m].argmax() == i)
                        correct["all three"].append(sum(scores.values()).argmax() == i)
            for m, v in correct.items():
                bag_rows.append({"repeat": r, "measure": m, "particles_per_sample": size, "n_samples": len(v),
                                 "accuracy": float(np.mean(v))})
    part = pd.DataFrame(part_rows)
    bags = pd.DataFrame(bag_rows)
    part_s = part.groupby("measure")[["balanced_accuracy", "macro_auc_ovr"]].agg(["mean", "std"])
    bag_s = bags.groupby(["measure", "particles_per_sample"]).agg(
        accuracy_mean=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
        n_samples=("n_samples", "mean")).reset_index()
    part.to_csv(OUT / "morphology_particle_level_repeats.csv", index=False)
    part_s.to_csv(OUT / "morphology_particle_level_summary.csv")
    bag_s.to_csv(OUT / "morphology_preparation_level.csv", index=False)
    print(part_s.round(3))
    print(bag_s.round(3).to_string(index=False))

    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for m, sub in bag_s.groupby("measure", sort=False):
        ax.errorbar(sub["particles_per_sample"], sub["accuracy_mean"], yerr=sub["accuracy_sd"], marker="o",
                    capsize=3, lw=2.2 if m == "all three" else 1.2, label=m)
    ax.axhline(1 / 3, color="grey", ls="--", lw=0.8, label="chance")
    ax.set_xlabel("Held-out particles per sample (per measure)")
    ax.set_ylabel("Accuracy (mean ± SD over repeats)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "morphology_preparation_level_accuracy.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    run(ap.parse_args().repeats)
