"""Figure 7 (GSE277105) transcriptomic analysis with a genuinely held-out test set.

Design. GSE277105 has eight 10x libraries: wild-type C57BL/6 and humanised-tau (MAPT) neurons, each
treated with AD-, DLB- or PSP-derived tau oligomers, plus one untreated (UT) library per genotype.
Each disease/genotype is a single library, so any within-library cell split would let a classifier
recognise the library (batch) rather than the disease. We therefore train on one genotype's three
disease libraries and test on the other genotype's three libraries, which the model never sees:

  1. CP10k + log1p normalisation per cell.
  2. Per-genotype centring on that genotype's untreated library (label-free), so features express the
     transcriptional response to tau oligomers relative to the matched untreated neurons.
  3. Genes: highly variable genes ranked on the training genotype only.
  4. PCA (fit on training cells) -> class-balanced logistic regression (C fixed a priori; no tuning,
     because cell-level inner CV would be batch-confounded).
  5. Evaluation on the held-out genotype: cell-level metric panel, and library-level calls
     (mean log-probability over all cells of a library).
  6. Exact null: all 3! assignments of disease labels to the training libraries (36 joint assignments
     across both transfer directions).
  7. Response-profile concordance: correlation of pseudobulk disease-vs-UT log fold changes between
     genotypes (is the AD response in C57 most similar to the AD response in MAPT?).

Usage:  python scripts/run_fig7_cross_genotype.py [--n-hvg 2000] [--n-pcs 50]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurofoldnet.evaluation import metric_panel  # noqa: E402

GEO = ROOT / "data" / "figure7_geo"
OUT = ROOT / "results" / "v2"
FIG = OUT / "figures"
CLASSES = ["AD", "DLB", "PSP"]
GENOTYPES = ["C57", "MAPT"]
SEED = 20260801


def load() -> tuple[sp.csr_matrix, np.ndarray, pd.DataFrame]:
    npz = GEO / "GSE277105_counts_cells_x_genes.npz"
    if not npz.exists():
        raise FileNotFoundError("Run scripts/build_fig7_sparse.py first")
    X = sp.load_npz(npz).tocsr()
    genes = np.array((GEO / "GSE277105_genes.txt").read_text(encoding="utf-8").split("\n"))
    cells = (GEO / "GSE277105_cells.txt").read_text(encoding="utf-8").split("\n")
    meta = pd.read_csv(GEO / "GSE277105_metadata.csv.gz", index_col=0).loc[cells]
    meta["genotype"] = np.where(meta["sample"].str.contains("MAPT"), "MAPT", "C57")
    meta["condition"] = meta["sample"].str.replace("C57_", "").str.replace("MAPT_", "").str.replace("_C57", "") \
        .str.replace("_MAPT", "")
    return X, genes, meta


def lognorm(X: sp.csr_matrix) -> sp.csr_matrix:
    lib = np.asarray(X.sum(axis=1)).ravel()
    Xn = sp.diags(1e4 / np.maximum(lib, 1)) @ X
    Xn.data = np.log1p(Xn.data)
    return Xn.tocsr()


def hvg(Xn: sp.csr_matrix, n: int) -> np.ndarray:
    mean = np.asarray(Xn.mean(axis=0)).ravel()
    sq = np.asarray(Xn.multiply(Xn).mean(axis=0)).ravel()
    var = sq - mean ** 2
    disp = var / np.maximum(mean, 1e-6)
    ok = mean > 0.01
    score = np.where(ok, disp, -np.inf)
    return np.argsort(score)[::-1][:n]


def treated_vs_untreated_control(Xn, meta, train_g, test_g, n_hvg, n_pcs, max_cells=20000):
    """Positive control: does a generic 'tau-oligomer treated vs untreated' signal transfer across genotypes?

    No UT-centring here (centring the test genotype on its UT library would use test labels).
    UT_C57 is downsampled (seeded) to limit runtime; class weights handle imbalance.
    """
    rng = np.random.default_rng(SEED)
    tr = np.flatnonzero((meta["genotype"] == train_g).to_numpy())
    if len(tr) > max_cells:
        tr = np.sort(rng.choice(tr, max_cells, replace=False))
    te = np.flatnonzero((meta["genotype"] == test_g).to_numpy())
    if len(te) > max_cells:
        te = np.sort(rng.choice(te, max_cells, replace=False))
    genes = hvg(Xn[tr], n_hvg)
    ytr = (meta["condition"].to_numpy()[tr] != "UT").astype(int)
    yte = (meta["condition"].to_numpy()[te] != "UT").astype(int)
    Xtr, Xte = Xn[tr][:, genes].toarray(), Xn[te][:, genes].toarray()
    pca = PCA(n_components=n_pcs, random_state=SEED).fit(Xtr)
    clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000).fit(pca.transform(Xtr), ytr)
    p = clf.predict_proba(pca.transform(Xte))[:, 1]
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    return {"task": "treated_vs_untreated", "train_genotype": train_g, "test_genotype": test_g,
            "n_test_cells": len(te), "balanced_accuracy": balanced_accuracy_score(yte, p > 0.5),
            "auc": roc_auc_score(yte, p)}


def transfer(Xn, meta, train_g, test_g, n_hvg, n_pcs, label_map=None, centre=True):
    """Train on train_g disease libraries, predict test_g disease libraries."""
    centred = {}
    for g in (train_g, test_g):
        ut = (meta["genotype"] == g) & (meta["condition"] == "UT")
        centred[g] = np.asarray(Xn[ut.to_numpy()].mean(axis=0)).ravel() if centre else 0.0
    tr = ((meta["genotype"] == train_g) & meta["condition"].isin(CLASSES)).to_numpy()
    te = ((meta["genotype"] == test_g) & meta["condition"].isin(CLASSES)).to_numpy()
    genes = hvg(Xn[tr], n_hvg)
    Xtr = Xn[tr][:, genes].toarray() - (centred[train_g][genes] if centre else 0.0)
    Xte = Xn[te][:, genes].toarray() - (centred[test_g][genes] if centre else 0.0)
    ytr_names = meta.loc[tr, "condition"].to_numpy()
    if label_map is not None:
        ytr_names = np.array([label_map[v] for v in ytr_names])
    ytr = np.array([CLASSES.index(v) for v in ytr_names])
    yte = np.array([CLASSES.index(v) for v in meta.loc[te, "condition"]])
    pca = PCA(n_components=n_pcs, random_state=SEED).fit(Xtr)
    clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000).fit(pca.transform(Xtr), ytr)
    proba = clf.predict_proba(pca.transform(Xte))
    lib_calls = []
    for c in range(len(CLASSES)):
        m = yte == c
        score = np.log(np.clip(proba[m], 1e-9, 1)).mean(axis=0)
        lib_calls.append({"test_library": f"{test_g}_{CLASSES[c]}", "n_cells": int(m.sum()),
                          "called": CLASSES[int(score.argmax())], "correct": bool(score.argmax() == c),
                          **{f"mean_logp_{k}": score[i] for i, k in enumerate(CLASSES)}})
    return yte, proba, lib_calls


def response_concordance(Xn, meta) -> pd.DataFrame:
    """Pseudobulk log fold change (disease - UT) per genotype; correlate across genotypes."""
    pb = {}
    for g in GENOTYPES:
        for cond in CLASSES + ["UT"]:
            m = ((meta["genotype"] == g) & (meta["condition"] == cond)).to_numpy()
            pb[(g, cond)] = np.asarray(Xn[m].mean(axis=0)).ravel()
    expressed = np.minimum.reduce([v for v in pb.values()]) > 0.05
    lfc = {(g, c): (pb[(g, c)] - pb[(g, "UT")])[expressed] for g in GENOTYPES for c in CLASSES}
    # remove the shared (disease-agnostic) tau response so the comparison isolates disease specificity
    spec = {}
    for g in GENOTYPES:
        common = np.mean([lfc[(g, c)] for c in CLASSES], axis=0)
        for c in CLASSES:
            spec[(g, c)] = lfc[(g, c)] - common
    mat = pd.DataFrame(index=[f"C57_{c}" for c in CLASSES], columns=[f"MAPT_{c}" for c in CLASSES], dtype=float)
    for a in CLASSES:
        for b in CLASSES:
            mat.loc[f"C57_{a}", f"MAPT_{b}"] = np.corrcoef(spec[("C57", a)], spec[("MAPT", b)])[0, 1]
    mat.attrs["n_genes"] = int(expressed.sum())
    return mat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-hvg", type=int, default=2000)
    ap.add_argument("--n-pcs", type=int, default=50)
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    X, genes, meta = load()
    Xn = lognorm(X)
    print(f"Loaded {X.shape[0]} cells x {X.shape[1]} genes ({time.time() - t0:.0f}s)", flush=True)

    rows, lib_rows, preds = [], [], {}
    for train_g, test_g in [("C57", "MAPT"), ("MAPT", "C57")]:
        yte, proba, libs = transfer(Xn, meta, train_g, test_g, args.n_hvg, args.n_pcs)
        preds[(train_g, test_g)] = (yte, proba)
        rows.append({"train_genotype": train_g, "test_genotype": test_g, "n_test_cells": len(yte),
                     **metric_panel(yte, proba, CLASSES)})
        lib_rows += [{"train_genotype": train_g, **r} for r in libs]
        print(f"{train_g}->{test_g}: balanced acc={rows[-1]['balanced_accuracy']:.3f}, "
              f"AUC={rows[-1]['macro_auc_ovr']:.3f}, library calls correct="
              f"{sum(r['correct'] for r in libs)}/3 ({time.time() - t0:.0f}s)", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "fig7_cross_genotype_cell_level.csv", index=False)
    pd.DataFrame(lib_rows).to_csv(OUT / "fig7_cross_genotype_library_calls.csv", index=False)

    # exact label-assignment null over training libraries
    null_rows = []
    perms = list(itertools.permutations(CLASSES))
    per_dir = {}
    for train_g, test_g in [("C57", "MAPT"), ("MAPT", "C57")]:
        per_dir[train_g] = []
        for perm in perms:
            mapping = dict(zip(CLASSES, perm))
            yte, proba, _ = transfer(Xn, meta, train_g, test_g, args.n_hvg, args.n_pcs, label_map=mapping)
            m = metric_panel(yte, proba, CLASSES)
            per_dir[train_g].append(m["balanced_accuracy"])
            null_rows.append({"train_genotype": train_g, "assignment": "->".join(f"{k}:{v}" for k, v in mapping.items()),
                              "identity": perm == tuple(CLASSES), "balanced_accuracy": m["balanced_accuracy"],
                              "macro_auc_ovr": m["macro_auc_ovr"]})
    pd.DataFrame(null_rows).to_csv(OUT / "fig7_label_assignment_null.csv", index=False)
    joint = np.add.outer(per_dir["C57"], per_dir["MAPT"]) / 2
    observed = (per_dir["C57"][0] + per_dir["MAPT"][0]) / 2  # identity permutation is first
    p_joint = float((joint >= observed - 1e-12).mean())
    summary = {"observed_mean_balanced_accuracy": observed, "exact_p_value_36_assignments": p_joint,
               "per_direction_p": {g: float((np.array(v) >= v[0] - 1e-12).mean()) for g, v in per_dir.items()},
               "n_hvg": args.n_hvg, "n_pcs": args.n_pcs}

    # positive control + pre-declared sensitivity grid (all settings reported, none selected post hoc)
    ctrl = [treated_vs_untreated_control(Xn, meta, a, b, args.n_hvg, args.n_pcs)
            for a, b in [("C57", "MAPT"), ("MAPT", "C57")]]
    pd.DataFrame(ctrl).to_csv(OUT / "fig7_positive_control_treated_vs_untreated.csv", index=False)
    print("Positive control:", pd.DataFrame(ctrl)[["train_genotype", "balanced_accuracy", "auc"]].round(3)
          .to_dict("records"), flush=True)
    sens = []
    for centre in (True, False):
        for n_hvg in (2000, 5000):
            for a, b in [("C57", "MAPT"), ("MAPT", "C57")]:
                yte, proba, libs = transfer(Xn, meta, a, b, n_hvg, args.n_pcs, centre=centre)
                m = metric_panel(yte, proba, CLASSES)
                sens.append({"ut_centring": centre, "n_hvg": n_hvg, "train_genotype": a, "test_genotype": b,
                             "balanced_accuracy": m["balanced_accuracy"], "macro_auc_ovr": m["macro_auc_ovr"],
                             "library_calls_correct": sum(r["correct"] for r in libs)})
    pd.DataFrame(sens).to_csv(OUT / "fig7_sensitivity_grid.csv", index=False)

    # supplementary: the same transfer restricted to each Seurat cluster with >=30 cells in all 6 libraries
    dis = meta[meta["condition"].isin(CLASSES)]
    ct = pd.crosstab(dis["seurat_clusters"], dis["sample"])
    clus_rows = []
    for cl in ct[(ct >= 30).all(axis=1)].index:
        keep = ((meta["seurat_clusters"] == cl)).to_numpy()  # UT centring uses UT cells of the same cluster
        for a, b in [("C57", "MAPT"), ("MAPT", "C57")]:
            yte, proba, libs = transfer(Xn[keep], meta[keep], a, b, args.n_hvg, 30)
            m = metric_panel(yte, proba, CLASSES)
            clus_rows.append({"cluster": int(cl), "train_genotype": a, "test_genotype": b, "n_test_cells": len(yte),
                              "balanced_accuracy": m["balanced_accuracy"], "macro_auc_ovr": m["macro_auc_ovr"],
                              "library_calls_correct": sum(r["correct"] for r in libs)})
    pd.DataFrame(clus_rows).to_csv(OUT / "fig7_per_cluster_transfer.csv", index=False)
    summary["positive_control"] = ctrl

    conc = response_concordance(Xn, meta)
    conc.to_csv(OUT / "fig7_response_concordance.csv")
    diag = np.mean(np.diag(conc.to_numpy()))
    perm_diag = [np.mean([conc.to_numpy()[i, p[i]] for i in range(3)]) for p in itertools.permutations(range(3))]
    summary["concordance_mean_diagonal"] = float(diag)
    summary["concordance_exact_p_6_assignments"] = float((np.array(perm_diag) >= diag - 1e-12).mean())
    summary["concordance_n_genes"] = conc.attrs["n_genes"]
    (OUT / "fig7_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, ((tr, te), (yte, proba)) in zip(axes[:2], preds.items()):
        ConfusionMatrixDisplay(confusion_matrix(yte, proba.argmax(1), normalize="true"),
                               display_labels=CLASSES).plot(ax=ax, cmap="Blues", colorbar=False, values_format=".2f")
        ax.set_title(f"Train {tr} -> test {te} (cells)")
    im = axes[2].imshow(conc.to_numpy(dtype=float), cmap="RdBu_r", vmin=-1, vmax=1)
    axes[2].set_xticks(range(3), conc.columns)
    axes[2].set_yticks(range(3), conc.index)
    for i in range(3):
        for j in range(3):
            axes[2].text(j, i, f"{conc.iat[i, j]:.2f}", ha="center", va="center")
    axes[2].set_title("Disease-specific response correlation")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIG / "fig7_cross_genotype.png", dpi=300)
    plt.close(fig)
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
