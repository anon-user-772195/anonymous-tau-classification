"""Fusion-rule comparison, assay ablation and calibration of NeuroFoldNet Level 2.

All analyses reuse the out-of-fold (held-out) predictions written by run_assay_models.py and
run_morphology_models.py, and the same preparation-sampling scheme as run_multiassay_panel.py:
a virtual preparation of a disease is N held-out measurements per source (one replicate for seeding),
drawn without replacement within the disease.

1. Fusion rules. Fixed rules (product = NeuroFoldNet, sum, max, median, majority vote) are compared with
   a learned fusion (multinomial logistic regression on the per-source mean log posteriors, i.e. stacking
   at preparation level). To keep the learned rule leakage-free, every source's held-out measurements are
   split into two disjoint halves per disease: the learned rule is fitted on preparations built from one
   half and all rules are evaluated on preparations built from the other half. Repeated over random splits.
2. Ablation. NeuroFoldNet fusion with one source (or one assay family) removed.
3. Calibration. Reliability of the fused posterior: expected calibration error (ECE, 10 equal-width bins of
   top-label confidence), reliability-diagram data, Brier score; also each single expert at N = 1.

Usage: python scripts/run_fusion_analyses.py [--splits 100] [--draws 500]
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results" / "v2"
CLASSES = ["AD", "DLB", "PSP"]
DYES = ["ThT", "Curcumin", "Bis-ANS", "FSB"]
MEASURES = ["height", "area", "diameter"]
EPS = 1e-6


def load_proba(path: Path) -> dict[int, np.ndarray]:
    df = pd.read_csv(path)
    p = np.clip(df[[f"p_{c}" for c in CLASSES]].to_numpy(), EPS, 1)
    p = p / p.sum(1, keepdims=True)
    return {i: p[df["label"].to_numpy() == c] for i, c in enumerate(CLASSES)}


def load_sources() -> dict[str, dict[int, np.ndarray]]:
    src = {f"AFM {m}": load_proba(R / f"oof_morphology_{m}.csv") for m in MEASURES}
    src.update({f"dye {d}": load_proba(R / f"oof_spectra_{d}_NeuroFoldNet.csv") for d in DYES})
    src["seeding"] = load_proba(sorted(R.glob("oof_seeding_*.csv"))[0])
    return src


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def draw(rng, src, keys, size, pools=None):
    """Return y (n,), mean log posteriors L (n, A, 3) and mean probabilities P (n, A, 3)."""
    ys, Ls, Ps = [], [], []
    for i in range(3):
        idx = {k: (pools[k][i] if pools else np.arange(len(src[k][i]))) for k in keys}
        perm = {k: rng.permutation(idx[k]) for k in keys}
        per = {k: (1 if k == "seeding" else size) for k in keys}
        n = min(len(perm[k]) // per[k] for k in keys)
        for b in range(n):
            L, P = [], []
            for k in keys:
                rows = src[k][i][perm[k][b * per[k]:(b + 1) * per[k]]]
                L.append(np.log(rows).mean(0))
                P.append(rows.mean(0))
            ys.append(i)
            Ls.append(L)
            Ps.append(P)
    return np.array(ys), np.array(Ls), np.array(Ps)


def fixed_rules(L: np.ndarray, P: np.ndarray) -> dict[str, np.ndarray]:
    out = {"Product rule (NeuroFoldNet)": softmax(L.sum(1))}
    s = P.mean(1)
    out["Sum rule"] = s / s.sum(1, keepdims=True)
    m = P.max(1)
    out["Max rule"] = m / m.sum(1, keepdims=True)
    md = np.median(P, axis=1) + EPS
    out["Median rule"] = md / md.sum(1, keepdims=True)
    votes = np.zeros((len(L), 3))
    for a in range(L.shape[1]):
        votes[np.arange(len(L)), L[:, a].argmax(1)] += 1
    votes += 1e-3 * softmax(L.sum(1))  # ties broken by summed evidence
    out["Majority vote"] = (votes + 0.5) / (votes + 0.5).sum(1, keepdims=True)
    return out


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    out = {"accuracy": float((p.argmax(1) == y).mean()), "log_loss": float(log_loss(y, p, labels=[0, 1, 2])),
           "brier": float(((p - np.eye(3)[y]) ** 2).sum(1).mean()), "ece": ece(y, p)}
    out["macro_auc"] = float(roc_auc_score(y, p, multi_class="ovr", average="macro")) if len(set(y)) == 3 else np.nan
    return out


def summarise(rows: list[dict], keys: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    num = [c for c in df.columns if c not in keys]
    g = df.groupby(keys, sort=False)[num]
    out = g.mean().add_suffix("_mean").join(g.quantile(0.025).add_suffix("_p2.5")).join(g.quantile(0.975).add_suffix("_p97.5"))
    return out.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=100)
    ap.add_argument("--draws", type=int, default=500)
    args = ap.parse_args()
    src = load_sources()
    all_keys = list(src)
    panels = {"AFM + 4 dyes + seeding": all_keys, "AFM + 4 dyes": [k for k in all_keys if k != "seeding"]}

    # ---------------------------------------------------------------- 1. fusion rules (split-half)
    rng = np.random.default_rng(20260801)
    rule_rows = []
    for panel, keys in panels.items():
        for size in [1, 3]:
            for _ in range(args.splits):
                pools_a, pools_b = {}, {}
                for k in keys:
                    pools_a[k], pools_b[k] = {}, {}
                    for i in range(3):
                        perm = rng.permutation(len(src[k][i]))
                        h = len(perm) // 2
                        pools_a[k][i], pools_b[k][i] = perm[:h], perm[h:]
                ytr, Ltr = [], []
                for _ in range(10):
                    y, L, _P = draw(rng, src, keys, size, pools_a)
                    ytr.append(y)
                    Ltr.append(L.reshape(len(L), -1))
                stack = LogisticRegression(C=1.0, max_iter=2000).fit(np.vstack(Ltr), np.concatenate(ytr))
                y, L, P = draw(rng, src, keys, size, pools_b)
                probs = fixed_rules(L, P)
                probs["Learned fusion (stacked logistic regression)"] = np.clip(stack.predict_proba(L.reshape(len(L), -1)), EPS, 1)
                for rule, p in probs.items():
                    rule_rows.append({"panel": panel, "per_source_N": size, "rule": rule, **metrics(y, p)})
    rules = summarise(rule_rows, ["panel", "per_source_N", "rule"])
    rules.to_csv(R / "fusion_rules_comparison.csv", index=False)

    # ---------------------------------------------------------------- 2. ablation (full data)
    rng = np.random.default_rng(7)
    families = {"AFM morphology (all 3 measures)": [k for k in all_keys if k.startswith("AFM")],
                "Dye spectra (all 4 dyes)": [k for k in all_keys if k.startswith("dye")], "Seeding": ["seeding"]}
    variants = {"None (full panel)": []}
    variants.update({f"{k}": [k] for k in all_keys})
    variants.update({f"Family: {f}": ks for f, ks in families.items()})
    abl_rows = []
    for size in [1, 3]:
        for name, removed in variants.items():
            keys = [k for k in all_keys if k not in removed]
            for _ in range(args.draws // 5):
                y, L, P = draw(rng, src, keys, size)
                abl_rows.append({"removed": name, "per_source_N": size, **metrics(y, softmax(L.sum(1)))})
    abl = summarise(abl_rows, ["removed", "per_source_N"])
    full = abl[abl.removed == "None (full panel)"].set_index("per_source_N")["accuracy_mean"]
    abl["accuracy_drop"] = abl.apply(lambda r: full[r.per_source_N] - r.accuracy_mean, axis=1)
    abl.to_csv(R / "fusion_ablation.csv", index=False)

    # ---------------------------------------------------------------- 3. calibration (full data)
    rng = np.random.default_rng(11)
    cal_rows, rel_rows = [], []
    settings = [(p, s) for p in panels for s in [1, 3]] + [(f"Single expert: {k}", 1) for k in all_keys]
    for name, size in settings:
        keys = panels.get(name, [name.replace("Single expert: ", "")])
        ys, ps = [], []
        for _ in range(args.draws // 5):
            y, L, P = draw(rng, src, keys, size)
            ys.append(y)
            ps.append(softmax(L.sum(1)))
        y, p = np.concatenate(ys), np.vstack(ps)
        m = metrics(y, p)
        cal_rows.append({"setting": name, "per_source_N": size, "n_predictions": len(y),
                         "mean_confidence": float(p.max(1).mean()), **m})
        conf, correct = p.max(1), (p.argmax(1) == y)
        for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
            sel = (conf > lo) & (conf <= hi)
            if sel.any():
                rel_rows.append({"setting": name, "per_source_N": size, "bin_low": lo, "bin_high": hi, "n": int(sel.sum()),
                                 "mean_confidence": float(conf[sel].mean()), "accuracy": float(correct[sel].mean())})
    pd.DataFrame(cal_rows).to_csv(R / "fusion_calibration.csv", index=False)
    rel = pd.DataFrame(rel_rows)
    rel.to_csv(R / "fusion_reliability_bins.csv", index=False)

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    ax = axes[0]
    sub = abl[(abl.per_source_N == 1) & (~abl.removed.isin(["None (full panel)", "Family: Seeding"]))].sort_values("accuracy_drop")
    labels = (sub.removed.str.replace("Family: ", "All ", regex=False).str.replace("dye ", "", regex=False)
              .str.replace("seeding", "Seeding", regex=False))
    colors = ["#C44E52" if lab.startswith("All ") else "#4C72B0" for lab in labels]
    ax.barh(labels, 100 * sub.accuracy_drop, color=colors)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("Accuracy drop when removed (percentage points)")
    ax.set_title("A  Ablation (1 measurement per source)", loc="left", fontsize=10)
    ax.tick_params(axis="y", labelsize=7)
    ax = axes[1]
    for (name, size), style in zip([("AFM + 4 dyes + seeding", 1), ("AFM + 4 dyes + seeding", 3), ("AFM + 4 dyes", 1)],
                                   ["o-", "s-", "^--"]):
        s = rel[(rel.setting == name) & (rel.per_source_N == size) & (rel.n >= 20)]
        e = [r for r in cal_rows if r["setting"] == name and r["per_source_N"] == size][0]["ece"]
        ax.plot(s.mean_confidence, s.accuracy, style, ms=4, label=f"{name}, N={size} (ECE {e:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", ls=":", lw=1)
    ax.set_xlim(0.3, 1.01)
    ax.set_ylim(0.3, 1.01)
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_title("B  Reliability of the fused posterior", loc="left", fontsize=10)
    ax.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(R / "figures" / "fusion_ablation_calibration.png", dpi=300)
    plt.close(fig)

    pd.set_option("display.width", 200)
    cols = ["panel", "per_source_N", "rule", "accuracy_mean", "macro_auc_mean", "log_loss_mean", "ece_mean"]
    print(rules[cols].round(3).to_string(index=False))
    print(abl[["removed", "per_source_N", "accuracy_mean", "accuracy_drop", "macro_auc_mean"]].round(3).to_string(index=False))
    print(pd.DataFrame(cal_rows)[["setting", "per_source_N", "accuracy", "mean_confidence", "ece", "brier"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
