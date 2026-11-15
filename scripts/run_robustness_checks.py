"""Robustness checks requested by a pre-submission statistical review (all exploratory).

  1. seeding_nested  - the seeding model family (logistic regression vs shrinkage LDA) is chosen INSIDE the
                       inner CV loop instead of post hoc on outer-CV accuracy, removing selection optimism.
  2. drift           - within-disease exchangeability: for every dye and disease, classify spectra from the
                       first vs second half of the acquisition order (group-aware CV, contiguous blocks).
                       Under exchangeability of measurements within a preparation the AUC is ~0.5.
  1b. seeding_fusion_sensitivity - full-panel fusion accuracy with each seeding expert.
  3. block_size      - sensitivity of the spectral NeuroFoldNet expert to the CV grouping unit
                       (block of 1 = no grouping, 5, 10 [primary], 20, 50 consecutive spectra).

Nothing here re-tunes or replaces the primary analysis; results are written to separate files.

Usage:  python scripts/run_robustness_checks.py [--only seeding_nested drift block_size] [--repeats 3]
Outputs: results/v2/robustness_seeding_nested.csv, robustness_drift.csv, robustness_block_size.csv
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurofoldnet import assays as A  # noqa: E402
from neurofoldnet import evaluation as E  # noqa: E402
from neurofoldnet import models as M  # noqa: E402

warnings.filterwarnings("ignore")
OUT = ROOT / "results" / "v2"
CLASSES = list(A.DISEASES)
SEED = 20260


def seeding_nested(args) -> pd.DataFrame:
    """Model family as a hyper-parameter of the inner search (GridSearchCV over a list of grids)."""
    s = A.load_seeding()
    X = s[["TOMA1", "TOMA2", "TOMA3", "TOMA4", "T22"]].to_numpy()
    y = s["label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    groups = np.array([f"{l}_{r}" for l, r in zip(s["label"], s["replicate"])])
    pipe = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=5000))])
    grid = [{"clf": [LogisticRegression(max_iter=5000)], "clf__C": [0.1, 1.0, 10.0]},
            {"clf": [LinearDiscriminantAnalysis(solver="lsqr")], "clf__shrinkage": ["auto", 0.5]}]
    res = E.repeated_nested_cv("nested_family", pipe, grid, X, y, groups, CLASSES, outer_splits=3,
                               inner_splits=2, repeats=20, seed=11)
    rm = res.repeat_metrics(y)
    chosen = pd.Series([str(sel.get("clf", "")).split("(")[0] for sel in res.selections]).value_counts()
    rows = [{"analysis": "family_selected_in_inner_cv", "metric": c, "mean": rm[c].mean(), "sd": rm[c].std(ddof=1)}
            for c in ["accuracy", "balanced_accuracy", "macro_auc_ovr", "log_loss"]]
    rows += [{"analysis": "inner_choice_frequency", "metric": k, "mean": v / len(res.selections), "sd": np.nan}
             for k, v in chosen.items()]
    if args.n_perm > 0:
        p = E.group_permutation_test(pipe, grid, X, y, groups, CLASSES, n_perm=args.n_perm, outer_splits=3,
                                     inner_splits=2, seed=13)
        rows.append({"analysis": "permutation_p", "metric": "balanced_accuracy", "mean": p["p_value"],
                     "sd": np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "robustness_seeding_nested.csv", index=False)
    # OOF posteriors of the honestly selected expert (file prefix keeps it out of the primary oof_seeding_* glob)
    pd.DataFrame({"label": s["label"], "replicate": s["replicate"],
                  **{f"p_{c}": res.mean_proba[:, i] for i, c in enumerate(CLASSES)}}).to_csv(
        OUT / "robustness_oof_seeding_nested.csv", index=False)
    return df


def seeding_fusion_sensitivity(args) -> pd.DataFrame:
    """Full-panel product-rule accuracy with the primary vs the honestly selected seeding expert."""
    spec = __import__("importlib.util").util.spec_from_file_location("fa", ROOT / "scripts" / "run_fusion_analyses.py")
    fa = __import__("importlib.util").util.module_from_spec(spec)
    spec.loader.exec_module(fa)
    base = fa.load_sources()
    alt = dict(base, seeding=fa.load_proba(OUT / "robustness_oof_seeding_nested.csv"))
    panels = {"AFM + dyes + seeding": list(base), "seeding only": ["seeding"],
              "AFM + dyes (no seeding)": [k for k in base if k != "seeding"]}
    rows = []
    for label, src in [("primary (shrinkage LDA, chosen post hoc)", base),
                       ("model family chosen in inner CV", alt)]:
        for pname, keys in panels.items():
            for size in [1, 3]:
                rng = np.random.default_rng(SEED)
                acc = []
                for _ in range(args.draws):
                    y, L, P = fa.draw(rng, src, keys, size)
                    acc.append((fa.softmax(L.sum(1)).argmax(1) == y).mean())
                rows.append({"seeding_expert": label, "panel": pname, "measurements_per_source": size,
                             "accuracy_mean": float(np.mean(acc)), "accuracy_q025": float(np.quantile(acc, .025)),
                             "accuracy_q975": float(np.quantile(acc, .975)), "draws": args.draws})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "robustness_seeding_fusion.csv", index=False)
    return df


def drift(args) -> pd.DataFrame:
    rows = []
    for dye in A.DYE_BLOCKS:
        s = A.load_spectra(dye)
        w = A.wavelength_columns(s)
        for disease in CLASSES:
            d = s[s["label"] == disease].sort_values("spectrum_index")
            idx = d["spectrum_index"].to_numpy()
            X = d[w].to_numpy()
            half = (np.arange(len(d)) >= len(d) // 2).astype(int)
            groups = np.arange(len(d)) // 10
            aucs = []
            for r in range(args.repeats):
                cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + r)
                p = np.full(len(d), np.nan)
                for tr, te in cv.split(X, half, groups):
                    m = Pipeline([("s", StandardScaler()),
                                  ("c", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))])
                    p[te] = m.fit(X[tr], half[tr]).predict_proba(X[te])[:, 1]
                aucs.append(roc_auc_score(half, p))
            # null: permute half-labels at block level (keeps block structure), same pipeline
            rng = np.random.default_rng(SEED)
            ublk = np.unique(groups)
            null = []
            for i in range(args.n_null):
                blk_lab = dict(zip(ublk, rng.permutation([half[groups == g][0] for g in ublk])))
                yp = np.array([blk_lab[g] for g in groups])
                cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + 999 + i)
                p = np.full(len(d), np.nan)
                for tr, te in cv.split(X, yp, groups):
                    m = Pipeline([("s", StandardScaler()),
                                  ("c", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))])
                    p[te] = m.fit(X[tr], yp[tr]).predict_proba(X[te])[:, 1]
                null.append(roc_auc_score(yp, p))
            obs = float(np.mean(aucs))
            null = np.array(null)
            rows.append({"dye": dye, "disease": disease, "n_spectra": len(d), "index_min": idx.min(),
                         "index_max": idx.max(), "auc_first_vs_second_half": obs,
                         "auc_sd_over_repeats": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
                         "null_mean": null.mean(), "null_95th": np.quantile(null, 0.95),
                         "p_value": (1 + (null >= obs).sum()) / (1 + len(null))})
            print(f"  drift {dye:9s} {disease}: AUC={obs:.3f} p={rows[-1]['p_value']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df["p_value"].to_numpy())
    df.to_csv(OUT / "robustness_drift.csv", index=False)
    return df


def holm(p):
    order = np.argsort(p)
    adj, running, m = np.empty_like(p, dtype=float), 0.0, len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def block_size(args) -> pd.DataFrame:
    rows = []
    for dye in A.DYE_BLOCKS:
        s = A.load_spectra(dye)
        w = A.wavelength_columns(s)
        wl = [int(c[3:]) for c in w]
        X = s[w].to_numpy()
        y = s["label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
        idx = s["spectrum_index"].to_numpy()
        for b in args.blocks:
            t0 = time.time()
            groups = E.contiguous_blocks(s["label"].to_numpy(), idx, b)
            res = E.repeated_nested_cv("NeuroFoldNet", M.neurofoldnet(wl), None, X, y, groups, CLASSES,
                                       repeats=args.repeats, seed=1)
            rm = res.repeat_metrics(y)
            rows.append({"dye": dye, "block_size": b, "n_groups": len(np.unique(groups)),
                         **{f"{c}_mean": rm[c].mean() for c in ["accuracy", "balanced_accuracy", "macro_auc_ovr"]},
                         "balanced_accuracy_sd": rm["balanced_accuracy"].std(ddof=1) if len(rm) > 1 else 0.0})
            print(f"  block {dye:9s} b={b:3d}: bacc={rows[-1]['balanced_accuracy_mean']:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(OUT / "robustness_block_size.csv", index=False)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*",
                    default=["seeding_nested", "seeding_fusion_sensitivity", "drift", "block_size"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--blocks", type=int, nargs="*", default=[1, 5, 10, 20, 50])
    ap.add_argument("--draws", type=int, default=500)
    args = ap.parse_args()
    for name in args.only:
        t0 = time.time()
        print(f"[{name}]", flush=True)
        df = globals()[name](args)
        print(df.to_string(index=False), flush=True)
        print(f"[{name}] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
