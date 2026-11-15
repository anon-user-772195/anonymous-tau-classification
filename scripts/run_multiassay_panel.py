"""Multi-assay preparation profile from held-out (out-of-fold) predictions.

A 'virtual preparation sample' of a disease consists of N held-out particles per AFM measure (height,
area, diameter), N held-out spectra per dye (ThT, Curcumin, Bis-ANS, FSB) and, optionally, one held-out
seeding replicate. Per-assay evidence is the mean log posterior over the N measurements; assays are
combined by a fixed, unweighted sum (no learned weights). Every probability used here is out-of-fold,
so combination adds no leakage.

Assumption (stated in the manuscript): within a disease, measurements from different assays are
exchangeable draws from that disease's oligomer preparation(s), because the source data contain no
sample identifier linking assays. Results therefore describe preparation-level separability.

Usage:  python scripts/run_multiassay_panel.py [--draws 500]
Requires outputs of run_assay_models.py and run_morphology_models.py.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurofoldnet.evaluation import metric_panel  # noqa: E402
R = ROOT / "results" / "v2"
CLASSES = ["AD", "DLB", "PSP"]
DYES = ["ThT", "Curcumin", "Bis-ANS", "FSB"]
MEASURES = ["height", "area", "diameter"]
SIZES = [1, 2, 3, 5, 10]


def load_logp(path: Path) -> dict[int, np.ndarray]:
    df = pd.read_csv(path)
    lp = np.log(np.clip(df[[f"p_{c}" for c in CLASSES]].to_numpy(), 1e-6, 1))
    return {i: lp[df["label"].to_numpy() == c] for i, c in enumerate(CLASSES)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=500)
    args = ap.parse_args()
    sources = {f"AFM {m}": load_logp(R / f"oof_morphology_{m}.csv") for m in MEASURES}
    sources.update({f"dye {d}": load_logp(R / f"oof_spectra_{d}_NeuroFoldNet.csv") for d in DYES})
    seed_file = sorted(R.glob("oof_seeding_*.csv"))
    seeding = load_logp(seed_file[0]) if seed_file else None

    panels = {
        "AFM morphology (3 measures)": [k for k in sources if k.startswith("AFM")],
        "4-dye spectra": [k for k in sources if k.startswith("dye")],
        "AFM + 4 dyes": list(sources),
    }
    rng = np.random.default_rng(7)
    rows, metric_rows = [], []
    for size in SIZES:
        for panel, keys in panels.items():
            for with_seed in ([False, True] if seeding is not None else [False]):
                accs = {"evidence_sum": [], "majority_vote": []}
                panels_per_draw = []
                for _ in range(args.draws):
                    correct = {r: [] for r in accs}
                    y_draw, fused_draw = [], []
                    for i in range(3):
                        # disjoint draws within each source; number of samples limited by the smallest source
                        perm = {k: rng.permutation(len(sources[k][i])) for k in keys}
                        n = min(len(p) for p in perm.values()) // size
                        if with_seed:
                            n = min(n, len(seeding[i]))
                            seed_order = rng.permutation(len(seeding[i]))
                        for b in range(n):
                            scores = [sources[k][i][perm[k][b * size:(b + 1) * size]].mean(0) for k in keys]
                            if with_seed:
                                scores.append(seeding[i][seed_order[b]])
                            total = sum(scores)
                            y_draw.append(i)
                            fused_draw.append(total)
                            correct["evidence_sum"].append(total.argmax() == i)
                            votes = np.bincount([s.argmax() for s in scores], minlength=3).astype(float)
                            votes += 1e-6 * (total - total.min())  # ties broken by summed evidence
                            correct["majority_vote"].append(votes.argmax() == i)
                    for r in accs:
                        accs[r].append(np.mean(correct[r]))
                    # full metric panel of the fused posterior (softmax of the summed log evidence)
                    z = np.array(fused_draw)
                    proba = np.exp(z - z.max(axis=1, keepdims=True))
                    proba /= proba.sum(axis=1, keepdims=True)
                    m = metric_panel(np.array(y_draw), proba, CLASSES)
                    for stat in ["sensitivity", "specificity", "precision"]:
                        m[f"macro_{stat}"] = float(np.mean([m[f"{c}_{stat}"] for c in CLASSES]))
                    panels_per_draw.append(m)
                mdf = pd.DataFrame(panels_per_draw)
                for metric in mdf.columns:
                    metric_rows.append({"panel": panel + (" + seeding" if with_seed else ""),
                                        "per_source_N": size, "metric": metric,
                                        "mean": mdf[metric].mean(), "p2.5": mdf[metric].quantile(0.025),
                                        "p97.5": mdf[metric].quantile(0.975)})
                for rule, a in accs.items():
                    a = np.array(a)
                    rows.append({"panel": panel + (" + seeding" if with_seed else ""), "fusion_rule": rule,
                                 "per_source_N": size, "samples_per_draw": len(correct[rule]),
                                 "accuracy_mean": a.mean(), "accuracy_p2.5": np.quantile(a, 0.025),
                                 "accuracy_p97.5": np.quantile(a, 0.975)})
    df = pd.DataFrame(rows)
    df.to_csv(R / "multiassay_panel.csv", index=False)
    print(df.round(3).to_string(index=False))
    mt = pd.DataFrame(metric_rows)
    mt.to_csv(R / "multiassay_panel_metrics.csv", index=False)
    key = mt[(mt.panel == "AFM + 4 dyes + seeding") & (mt.per_source_N.isin([1, 3]))]
    print(key.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4.4))
    for panel, sub in df[df.fusion_rule == "evidence_sum"].groupby("panel", sort=False):
        ax.plot(sub["per_source_N"], sub["accuracy_mean"], marker="o", label=panel,
                lw=2.4 if panel.startswith("AFM + 4 dyes") else 1.2)
    ax.axhline(1 / 3, color="grey", ls="--", lw=0.8, label="chance")
    ax.set_xlabel("Held-out measurements per source")
    ax.set_ylabel("Preparation-level accuracy")
    ax.set_ylim(0.3, 1.02)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(R / "figures" / "multiassay_panel.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
