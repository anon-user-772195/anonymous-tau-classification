"""Decision-support extensions of NeuroFoldNet: conformal prediction with abstention, and sequential assay selection.

All analyses reuse held-out (out-of-fold) predictions and the preparation-sampling scheme of run_multiassay_panel.py,
with a split-half protocol so that anything estimated (conformal quantiles, predictive distributions) comes from one
half of the held-out measurements and is evaluated on the other half.

A. Conformal prediction (class-conditional / Mondrian split conformal; score = 1 - fused posterior of the class).
   Prediction sets with guaranteed per-class coverage 1 - alpha; an empty set is an abstention.
B. Out-of-distribution rejection. Recombinant tau oligomers (rTauO), which belong to no class, are scored by the frozen
   spectral experts (never trained on rTauO) and fused over the four dyes. We report how often their prediction set is
   empty (rejected) versus disease preparations, and the AUC of the conformal OOD score.
C. Sequential assay selection. Starting from no measurements, the next measurement is the one with the largest expected
   reduction in posterior entropy (value of information), with predictive distributions estimated from the other half;
   acquisition stops when the fused posterior reaches a confidence threshold. Compared with fixed and random orders.

Usage: python scripts/run_decision_support.py [--splits 100]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from neurofoldnet import assays as A  # noqa: E402
from run_fusion_analyses import load_sources, softmax, draw, EPS  # noqa: E402

R = ROOT / "results" / "v2"
DYES = ["ThT", "Curcumin", "Bis-ANS", "FSB"]
ALPHAS = [0.10, 0.05]


# ----------------------------------------------------------------------------- conformal helpers
def mondrian_quantiles(y_cal, p_cal, alpha):
    q = np.zeros(3)
    for c in range(3):
        s = np.sort(1 - p_cal[y_cal == c, c])
        n = len(s)
        k = int(np.ceil((n + 1) * (1 - alpha)))
        q[c] = s[min(k, n) - 1] if k <= n else 1.0
    return q


def conformal_pvalues(y_cal, p_cal, p_test):
    """Class-conditional conformal p-value of each class for each test point."""
    pv = np.zeros_like(p_test)
    for c in range(3):
        s_cal = 1 - p_cal[y_cal == c, c]
        s_t = 1 - p_test[:, c]
        pv[:, c] = (1 + (s_cal[None, :] >= s_t[:, None]).sum(1)) / (len(s_cal) + 1)
    return pv


def split_pools(rng, src, keys):
    a, b = {}, {}
    for k in keys:
        a[k], b[k] = {}, {}
        for i in range(3):
            perm = rng.permutation(len(src[k][i]))
            h = len(perm) // 2
            a[k][i], b[k][i] = perm[:h], perm[h:]
    return a, b


def novelty_detection(rng, splits, sizes=(1, 3, 5)):
    """Conformal novelty detection on dye spectra: distance of a preparation to the disease spectra distribution.

    Per dye, standardisation, PCA (8 components) and a Ledoit-Wolf covariance are fitted on half of the disease
    spectra; the preparation score is the sum over dyes of the mean Mahalanobis distance of its N spectra. A quarter of
    the disease spectra calibrates the score (conformal p-value); the last quarter and all rTauO spectra are tested.
    """
    from sklearn.covariance import LedoitWolf
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    data = {}
    for dye in DYES:
        df = A.load_spectra(dye, include_control=True)
        w = A.wavelength_columns(df)
        data[dye] = (df.loc[df.label != "rTauO", w].to_numpy(float), df.loc[df.label == "rTauO", w].to_numpy(float))
    rows = []
    for _ in range(splits):
        dist = {}
        for dye, (dis, rt) in data.items():
            perm = rng.permutation(len(dis))
            q = len(dis) // 4
            fit, cal, test = dis[perm[:2 * q]], dis[perm[2 * q:3 * q]], dis[perm[3 * q:]]
            sc = StandardScaler().fit(fit)
            pca = PCA(n_components=8).fit(sc.transform(fit))
            lw = LedoitWolf().fit(pca.transform(sc.transform(fit)))
            dist[dye] = {k: lw.mahalanobis(pca.transform(sc.transform(v))) for k, v in
                         [("cal", cal), ("test", test), ("rtau", rt)]}
        for size in sizes:
            def prep_scores(part):
                n = min(len(dist[d][part]) for d in DYES) // size
                perms = {d: rng.permutation(len(dist[d][part])) for d in DYES}
                return np.array([sum(dist[d][part][perms[d][b * size:(b + 1) * size]].mean() for d in DYES)
                                 for b in range(n)])
            s_cal, s_test, s_rt = prep_scores("cal"), prep_scores("test"), prep_scores("rtau")
            pv = lambda s: (1 + (s_cal[None, :] >= s[:, None]).sum(1)) / (len(s_cal) + 1)
            p_test, p_rt = pv(s_test), pv(s_rt)
            auc = roc_auc_score(np.r_[np.zeros(len(s_test)), np.ones(len(s_rt))], np.r_[s_test, s_rt])
            for alpha in ALPHAS:
                rows.append({"spectra_per_dye": size, "alpha": alpha, "disease_rejected": (p_test <= alpha).mean(),
                             "rTauO_rejected": (p_rt <= alpha).mean(), "auc": auc})
    return pd.DataFrame(rows).groupby(["spectra_per_dye", "alpha"], sort=False).mean().reset_index()


def fit_temperatures(src, keys, pools):
    """One temperature per expert (p^(1/T), renormalised), fitted by log loss on the given half."""
    grid = np.linspace(0.3, 6, 58)
    temps = {}
    for k in keys:
        P = np.vstack([src[k][c][pools[k][c]] for c in range(3)])
        y = np.concatenate([np.full(len(pools[k][c]), c) for c in range(3)])
        losses = []
        for t in grid:
            q = P ** (1 / t)
            q = q / q.sum(1, keepdims=True)
            losses.append(-np.log(q[np.arange(len(y)), y] + 1e-12).mean())
        temps[k] = grid[int(np.argmin(losses))]
    return temps


def temper(src, temps):
    out = {}
    for k, t in temps.items():
        out[k] = {}
        for c in range(3):
            q = src[k][c] ** (1 / t)
            out[k][c] = q / q.sum(1, keepdims=True)
    return out


def rtau_proba():
    """Frozen spectral experts applied to rTauO spectra (out-of-distribution; never seen in training)."""
    bundle = joblib.load(ROOT / "models" / "neurofoldnet_frozen.joblib")
    out = {}
    for dye in DYES:
        df = A.load_spectra(dye, include_control=True)
        df = df[df.label == "rTauO"]
        spec = bundle["spectra"][dye]
        p = np.clip(spec["model"].predict_proba(df[spec["columns"]]), EPS, 1)
        out[dye] = p / p.sum(1, keepdims=True)
    return out


def rtau_preps(rng, rp, size):
    n = min(len(v) for v in rp.values()) // size
    perm = {d: rng.permutation(len(rp[d])) for d in DYES}
    L = np.array([[np.log(rp[d][perm[d][b * size:(b + 1) * size]]).mean(0) for d in DYES] for b in range(n)])
    return softmax(L.sum(1))


# ----------------------------------------------------------------------------- sequential acquisition
def run_sequential(rng, src, keys, pools_a, pools_b, true_class, policy, tau, max_per_source, n_mc=20):
    """Acquire measurements one at a time for one preparation; return (correct, n_measurements, reached_tau)."""
    queues = {k: list(rng.permutation(pools_b[k][true_class])) for k in keys}
    cap = {k: (1 if k == "seeding" else max_per_source) for k in keys}
    acquired = {k: [] for k in keys}
    order = [k for _ in range(max_per_source) for k in keys]
    step = 0

    def posterior(acq):
        z = np.zeros(3)
        for k, rows in acq.items():
            if rows:
                z += np.log(np.array(rows)).mean(0)
        return softmax(z[None])[0]

    while True:
        post = posterior(acquired)
        if post.max() >= tau:
            return int(post.argmax() == true_class), step, 1
        cands = [k for k in keys if len(acquired[k]) < cap[k] and queues[k]]
        if not cands:
            return int(post.argmax() == true_class), step, 0
        if policy == "voi":
            best, best_h = None, np.inf
            for k in cands:
                h = 0.0
                for c in range(3):
                    if post[c] < 1e-4:
                        continue
                    pool = pools_a[k][c]
                    sel = src[k][c][rng.choice(pool, size=min(n_mc, len(pool)), replace=False)]
                    for row in sel:
                        trial = {kk: v + [row] if kk == k else v for kk, v in acquired.items()}
                        p2 = posterior(trial)
                        h += post[c] * (-(p2 * np.log(p2 + 1e-12)).sum()) / len(sel)
                if h < best_h:
                    best, best_h = k, h
            k = best
        elif policy == "fixed":
            k = next(kk for kk in order[step:] + keys if kk in cands)
        else:
            k = cands[rng.integers(len(cands))]
        acquired[k].append(src[k][true_class][queues[k].pop()])
        step += 1


def make_figure() -> None:
    """Redraw the decision-support figure from the saved result files."""
    conf = pd.read_csv(R / "conformal_prediction.csv")
    seq = pd.read_csv(R / "sequential_acquisition_runs.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    ax = axes[0]
    sub = conf[conf.alpha.round(2) == 0.10]
    short = {"AFM + 4 dyes + seeding": "All assays", "AFM + 4 dyes": "AFM + dyes", "4-dye spectra": "Dyes only"}
    labels = [short[p] + "\nN=" + str(n) for p, n in zip(sub.panel, sub.per_source_N)]
    x = np.arange(len(sub))
    ax.bar(x - 0.2, sub.coverage, 0.4, label="Coverage", color="#4C72B0")
    ax.bar(x + 0.2, sub.singleton_rate, 0.4, label="Single-label sets", color="#55A868")
    ax.axhline(0.90, color="grey", ls="--", lw=0.8)
    ax.set_xticks(x, labels, fontsize=7.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction of preparations")
    ax.set_title("A  Conformal prediction sets (target 90%)", loc="left", fontsize=10)
    ax.legend(fontsize=7, loc="lower left")
    ax = axes[1]
    names = {"voi_calibrated": "Value of information, calibrated", "voi": "Value of information, raw",
             "fixed": "Fixed order", "random": "Random order"}
    for policy, style in [("voi_calibrated", "o-"), ("voi", "o:"), ("fixed", "s--"), ("random", "^:")]:
        s = seq[(seq.policy == policy) & (seq.threshold.round(2) == 0.95)]
        counts = np.arange(0, 23)
        frac = [((s.n_measurements <= k) & (s.reached_threshold == 1)).mean() for k in counts]
        ax.plot(counts, frac, style, ms=3, label=f"{names[policy]} (acc. {100 * s.correct.mean():.0f}%)")
    ax.set_xlabel("Measurements acquired")
    ax.set_ylabel("Fraction reaching 95% confidence")
    ax.set_ylim(0, 1.0)
    ax.set_title("B  Sequential assay selection", loc="left", fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(R / "figures" / "decision_support.png", dpi=300)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=100)
    ap.add_argument("--figure-only", action="store_true")
    args = ap.parse_args()
    if args.figure_only:
        make_figure()
        return
    src = load_sources()
    keys_all = list(src)
    dye_keys = [k for k in keys_all if k.startswith("dye")]
    panels = {"AFM + 4 dyes + seeding": keys_all, "AFM + 4 dyes": [k for k in keys_all if k != "seeding"],
              "4-dye spectra": dye_keys}
    rp = rtau_proba()

    # ----------------------------------------------------------------- A + B: conformal and OOD
    rng = np.random.default_rng(20261115)
    conf_rows, ood_rows = [], []
    for panel, keys in panels.items():
        for size in [1, 3]:
            for _ in range(args.splits):
                pa, pb = split_pools(rng, src, keys)
                yc, pc = [], []
                for _ in range(20):
                    y, L, _ = draw(rng, src, keys, size, pa)
                    yc.append(y)
                    pc.append(softmax(L.sum(1)))
                y_cal, p_cal = np.concatenate(yc), np.vstack(pc)
                y_t, L, _ = draw(rng, src, keys, size, pb)
                p_t = softmax(L.sum(1))
                for alpha in ALPHAS:
                    q = mondrian_quantiles(y_cal, p_cal, alpha)
                    sets = (1 - p_t) <= q[None, :]
                    size_ = sets.sum(1)
                    cov = sets[np.arange(len(y_t)), y_t]
                    single = size_ == 1
                    row = {"panel": panel, "per_source_N": size, "alpha": alpha, "coverage": cov.mean(),
                           "mean_set_size": size_.mean(), "singleton_rate": single.mean(), "empty_rate": (size_ == 0).mean(),
                           "singleton_accuracy": (sets[single].argmax(1) == y_t[single]).mean() if single.any() else np.nan}
                    for c, name in enumerate(["AD", "DLB", "PSP"]):
                        row[f"coverage_{name}"] = cov[y_t == c].mean()
                    conf_rows.append(row)
                if panel == "4-dye spectra":
                    p_r = rtau_preps(rng, rp, size)
                    pv_d = conformal_pvalues(y_cal, p_cal, p_t)
                    pv_r = conformal_pvalues(y_cal, p_cal, p_r)
                    score = np.r_[pv_d.max(1), pv_r.max(1)]
                    lab = np.r_[np.zeros(len(pv_d)), np.ones(len(pv_r))]
                    for alpha in ALPHAS:
                        ood_rows.append({"per_source_N": size, "alpha": alpha,
                                         "disease_rejected": (pv_d.max(1) <= alpha).mean(),
                                         "rTauO_rejected": (pv_r.max(1) <= alpha).mean(),
                                         "ood_auc": roc_auc_score(lab, -score),
                                         "rTauO_max_posterior": p_r.max(1).mean(),
                                         "disease_max_posterior": p_t.max(1).mean()})
    conf = pd.DataFrame(conf_rows).groupby(["panel", "per_source_N", "alpha"], sort=False).mean().reset_index()
    conf.to_csv(R / "conformal_prediction.csv", index=False)
    ood = pd.DataFrame(ood_rows).groupby(["per_source_N", "alpha"], sort=False).mean().reset_index()
    ood.to_csv(R / "conformal_ood_rejection.csv", index=False)
    nov = novelty_detection(np.random.default_rng(3), args.splits)
    nov.to_csv(R / "novelty_detection.csv", index=False)

    # ----------------------------------------------------------------- C: sequential acquisition
    rng = np.random.default_rng(7)
    seq_rows, temp_rows = [], []
    for tau in [0.90, 0.95]:
        for _ in range(max(10, args.splits // 2)):
            pa, pb = split_pools(rng, src, keys_all)
            temps = fit_temperatures(src, keys_all, pa)
            temp_rows.append(temps)
            src_cal = temper(src, temps)
            for c in range(3):
                for policy, source in [("voi", src), ("voi_calibrated", src_cal), ("fixed", src_cal),
                                       ("random", src_cal)]:
                    pol = "voi" if policy.startswith("voi") else policy
                    correct, n, reached = run_sequential(np.random.default_rng(rng.integers(1e9)), source, keys_all,
                                                         pa, pb, c, pol, tau, max_per_source=3)
                    seq_rows.append({"threshold": tau, "policy": policy, "true_class": c, "correct": correct,
                                     "n_measurements": n, "reached_threshold": reached})
    pd.DataFrame(temp_rows).agg(["mean", "std"]).T.rename_axis("expert").reset_index().to_csv(
        R / "expert_temperatures.csv", index=False)
    seq = pd.DataFrame(seq_rows)
    seq_sum = seq.groupby(["threshold", "policy"], sort=False).agg(
        accuracy=("correct", "mean"), mean_measurements=("n_measurements", "mean"),
        median_measurements=("n_measurements", "median"), reached_threshold=("reached_threshold", "mean"),
        n=("correct", "size")).reset_index()
    seq_sum.to_csv(R / "sequential_acquisition.csv", index=False)
    seq.to_csv(R / "sequential_acquisition_runs.csv", index=False)

    make_figure()

    pd.set_option("display.width", 220)
    print(conf.round(3).to_string(index=False))
    print(ood.round(3).to_string(index=False))
    print(nov.round(3).to_string(index=False))
    print(pd.read_csv(R / "expert_temperatures.csv").round(2).to_string(index=False))
    print(seq_sum.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
