"""Frozen NeuroFoldNet models for external validation on independent oligomer preparations.

  python scripts/neurofoldnet_external.py freeze
      Fits every assay expert on all source data and writes models/neurofoldnet_frozen.joblib
      plus blank input templates in models/templates/.

  python scripts/neurofoldnet_external.py predict --spectra-dir DIR --morphology CSV --seeding CSV [--out CSV]
      Applies the frozen experts to new measurements from one or more preparations and fuses the
      evidence with the fixed NeuroFoldNet rule (unweighted sum of mean log posteriors). Any subset of
      assays may be supplied. Input formats (see templates):
        spectra-dir/<Dye>.csv : preparation_id + 30 columns wl_416 ... wl_706 (max-normalised emission)
        morphology CSV        : preparation_id, measure (height|area|diameter), value (nm / nm^2)
        seeding CSV           : preparation_id, TOMA1, TOMA2, TOMA3, TOMA4, T22
      Output: one row per preparation with per-assay and fused probabilities and the called disease.

The frozen file is created once and must not be refit on external data, so that predictions are a
genuine external test.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurofoldnet import assays as A  # noqa: E402
from neurofoldnet import models as M  # noqa: E402

MODELS = ROOT / "models"
FROZEN = MODELS / "neurofoldnet_frozen.joblib"
CLASSES = list(A.DISEASES)
SEED_FEATS = ["TOMA1", "TOMA2", "TOMA3", "TOMA4", "T22"]


def freeze() -> None:
    MODELS.mkdir(exist_ok=True)
    (MODELS / "templates").mkdir(exist_ok=True)
    bundle = {"classes": CLASSES, "spectra": {}, "morphology": {}, "version": "2.0"}
    for dye in A.DYE_BLOCKS:
        s = A.load_spectra(dye)
        w = A.wavelength_columns(s)
        y = s["label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
        bundle["spectra"][dye] = {"columns": w, "model": M.neurofoldnet([int(c[3:]) for c in w]).fit(s[w], y)}
        pd.DataFrame(columns=["preparation_id"] + w).to_csv(MODELS / "templates" / f"{dye}.csv", index=False)
    h = A.load_morphology_histograms()
    rng = np.random.default_rng(100)
    for (label, measure), sub in h.groupby(["label", "measure"]):
        vals = A.reconstruct_log_particles(sub["bin_centre"].to_numpy(), sub["count"].to_numpy(), rng)
        bundle["morphology"].setdefault(measure, {})[label] = gaussian_kde(vals)
    pd.DataFrame(columns=["preparation_id", "measure", "value"]).to_csv(MODELS / "templates" / "morphology.csv",
                                                                        index=False)
    sd = A.load_seeding()
    est, _ = M.small_n_grids()["shrinkage_lda"]
    est.set_params(clf__shrinkage="auto")
    bundle["seeding"] = est.fit(sd[SEED_FEATS].to_numpy(),
                                sd["label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy())
    pd.DataFrame(columns=["preparation_id"] + SEED_FEATS).to_csv(MODELS / "templates" / "seeding.csv", index=False)
    joblib.dump(bundle, FROZEN, compress=("xz", 9))  # compressed so the model fits in a GitHub upload
    print(f"Frozen models written to {FROZEN}")


def _logp(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, 1e-6, 1))


def predict(args) -> pd.DataFrame:
    b = joblib.load(FROZEN)
    evidence: dict[str, dict[str, np.ndarray]] = {}  # preparation -> assay -> mean log posterior
    if args.spectra_dir:
        for dye, spec in b["spectra"].items():
            f = Path(args.spectra_dir) / f"{dye}.csv"
            if not f.exists():
                continue
            df = pd.read_csv(f)
            lp = _logp(spec["model"].predict_proba(df[spec["columns"]]))
            for prep, idx in df.groupby("preparation_id").groups.items():
                evidence.setdefault(str(prep), {})[f"dye_{dye}"] = lp[np.asarray([df.index.get_loc(i) for i in idx])].mean(0)
    if args.morphology:
        df = pd.read_csv(args.morphology)
        for (prep, measure), sub in df.groupby(["preparation_id", "measure"]):
            kdes = b["morphology"][measure]
            x = np.log(np.clip(sub["value"].to_numpy(float), 1e-3, None))
            ll = np.column_stack([np.log(kdes[c](x) + 1e-12) for c in CLASSES])
            post = ll - np.logaddexp.reduce(ll, axis=1, keepdims=True)
            evidence.setdefault(str(prep), {})[f"afm_{measure}"] = post.mean(0)
    if args.seeding:
        df = pd.read_csv(args.seeding)
        lp = _logp(b["seeding"].predict_proba(df[SEED_FEATS].to_numpy()))
        for prep, idx in df.groupby("preparation_id").groups.items():
            evidence.setdefault(str(prep), {})["seeding"] = lp[np.asarray([df.index.get_loc(i) for i in idx])].mean(0)
    rows = []
    for prep, ev in evidence.items():
        fused = sum(ev.values())
        post = np.exp(fused - np.logaddexp.reduce(fused))
        row = {"preparation_id": prep, "assays_used": ";".join(sorted(ev)), "called": CLASSES[int(fused.argmax())],
               **{f"p_{c}": post[i] for i, c in enumerate(CLASSES)}}
        for k, v in ev.items():
            row[f"{k}_call"] = CLASSES[int(v.argmax())]
        rows.append(row)
    out = pd.DataFrame(rows)
    if args.out:
        out.to_csv(args.out, index=False)
    print(out.to_string(index=False))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("freeze")
    p = sub.add_parser("predict")
    p.add_argument("--spectra-dir")
    p.add_argument("--morphology")
    p.add_argument("--seeding")
    p.add_argument("--out")
    args = ap.parse_args()
    freeze() if args.cmd == "freeze" else predict(args)


if __name__ == "__main__":
    main()
