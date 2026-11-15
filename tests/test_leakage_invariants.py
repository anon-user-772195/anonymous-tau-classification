"""Leakage and correctness invariants of the NeuroFoldNet pipeline.

Run:  python -m pytest -q tests
Each test guards one property the manuscript relies on; none of them looks at model accuracy.
"""
from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurofoldnet import assays as A  # noqa: E402
from neurofoldnet import evaluation as E  # noqa: E402

CLASSES = list(A.DISEASES)


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------- grouping / CV
class GroupSpy(ClassifierMixin, BaseEstimator):
    """Column 0 of X carries the group id; predicting a row whose group was seen in fit is a leak.

    Leaks are recorded in a class-level list (not raised) because GridSearchCV swallows exceptions
    raised during inner scoring.
    """
    leaks: list = []
    calls: list = []

    def fit(self, X, y):
        self.seen_ = set(np.asarray(X)[:, 0].tolist())
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        GroupSpy.calls.append(len(X))
        GroupSpy.leaks.extend(self.seen_ & set(np.asarray(X)[:, 0].tolist()))
        return np.full((len(X), 3), 1 / 3)


@pytest.fixture(autouse=True)
def _reset_spy():
    GroupSpy.leaks, GroupSpy.calls = [], []
    yield
    assert not GroupSpy.leaks, f"groups {sorted(set(GroupSpy.leaks))[:5]} were in both training and test"


def _toy(n_per=60, block=10):
    y = np.repeat([0, 1, 2], n_per)
    idx = np.tile(np.arange(n_per), 3)
    labels = np.array(CLASSES)[y]
    groups = E.contiguous_blocks(labels, idx, block)
    gid = np.unique(groups, return_inverse=True)[1].astype(float)
    X = np.column_stack([gid, np.random.default_rng(0).normal(size=len(y))])
    return X, y, groups


def test_contiguous_blocks_are_label_pure_and_bounded():
    X, y, groups = _toy(n_per=57, block=10)
    for g in np.unique(groups):
        assert len(np.unique(y[groups == g])) == 1
        assert (groups == g).sum() <= 10


def test_outer_folds_never_share_groups():
    X, y, groups = _toy()
    E.repeated_nested_cv("spy", GroupSpy(), None, X, y, groups, CLASSES, outer_splits=5, repeats=3, seed=0)


def test_inner_search_folds_never_share_groups():
    X, y, groups = _toy()
    # a grid forces GridSearchCV; GroupSpy must also hold inside every inner split
    E.repeated_nested_cv("spy", GroupSpy(), [{}], X, y, groups, CLASSES, outer_splits=5, inner_splits=3,
                         repeats=1, seed=0)
    assert len(GroupSpy.calls) == 5 * 3 + 5          # inner scoring calls + outer test predictions


def test_every_sample_gets_one_oof_prediction_per_repeat():
    X, y, groups = _toy()
    res = E.repeated_nested_cv("spy", GroupSpy(), None, X, y, groups, CLASSES, outer_splits=5, repeats=2, seed=0)
    assert res.oof_proba.shape == (2, len(y), 3)
    assert not np.isnan(res.oof_proba).any()
    assert len(res.fold_scores) == 10 and res.fold_scores["n_test"].sum() == 2 * len(y)


def test_group_permutation_keeps_one_label_per_group():
    X, y, groups = _toy()
    uniq = np.unique(groups)
    lab = np.array([y[groups == g][0] for g in uniq])
    rng = np.random.default_rng(1)
    perm = dict(zip(uniq, rng.permutation(lab)))
    yp = np.array([perm[g] for g in groups])
    for g in uniq:
        assert len(np.unique(yp[groups == g])) == 1
    assert np.bincount(yp).tolist() == np.bincount(y).tolist()


# ----------------------------------------------------------------------------- source-data parsers
@pytest.mark.parametrize("dye", list(A.DYE_BLOCKS))
def test_spectra_parser_shapes(dye):
    s = A.load_spectra(dye)
    assert set(s["label"]) == set(CLASSES)
    assert 430 <= len(s) <= 460
    assert len(A.wavelength_columns(s)) == A.N_WAVELENGTHS
    assert not s[A.wavelength_columns(s)].isna().any().any()
    assert "rTauO" not in set(s["label"])            # the control is only loaded on request


def test_seeding_parser_shape():
    s = A.load_seeding()
    assert len(s) == 18
    assert s.groupby("label").size().tolist() == [6, 6, 6]
    assert {"TOMA1", "TOMA2", "TOMA3", "TOMA4", "T22"} <= set(s.columns)


# ----------------------------------------------------------------------------- morphology bin grid
def test_particle_reconstruction_removes_bin_grid():
    mm = _load_script("run_morphology_models")
    h = A.load_morphology_histograms()
    data = mm.reconstruct(h, np.random.default_rng(0))
    for (label, measure), s in h.groupby(["label", "measure"]):
        c = np.sort(s["bin_centre"].to_numpy())
        width = np.median(np.diff(c))
        vals = np.exp(data[(label, measure)])
        assert len(vals) == s["count"].sum()
        assert (vals > 0).all()
        assert np.unique(vals).size == vals.size, "point mass in reconstructed particles (e.g. clipping)"
        inner = vals >= c[0] + width / 2                          # bins that do not straddle zero
        nearest = c[np.abs(vals[inner, None] - c[None, :]).argmin(1)]
        frac = (vals[inner] - nearest) / width + 0.5              # position within the bin, should be U(0,1)
        assert np.isclose(vals, c[np.abs(vals[:, None] - c[None, :]).argmin(1)]).mean() < 0.01
        if len(frac) > 50:
            assert stats.kstest(frac, "uniform").pvalue > 1e-4


# ----------------------------------------------------------------------------- fusion resampling
def test_fusion_draw_uses_each_measurement_once():
    fa = _load_script("run_fusion_analyses")
    rng = np.random.default_rng(0)
    src = {}
    for k, n in [("a", 40), ("b", 25)]:
        src[k] = {}
        for i in range(3):
            p = rng.dirichlet(np.ones(3), size=n)               # unique rows identify the measurement
            src[k][i] = p
    y, L, P = fa.draw(np.random.default_rng(1), src, ["a", "b"], size=1)
    for a_i, k in enumerate(["a", "b"]):
        for i in range(3):
            rows = np.exp(L[y == i, a_i])
            ids = [int(np.argmin(np.abs(src[k][i] - r).sum(1))) for r in rows]
            assert len(ids) == len(set(ids)), "a measurement was reused within one draw"
    assert np.bincount(y).tolist() == [25, 25, 25]              # limited by the smallest pool


def test_product_rule_is_normalised_product_of_geometric_means():
    fa = _load_script("run_fusion_analyses")
    rng = np.random.default_rng(2)
    P = rng.dirichlet(np.ones(3), size=(50, 4))
    L = np.log(P)
    prod = np.prod(P, axis=1)
    expected = prod / prod.sum(1, keepdims=True)
    assert np.allclose(fa.fixed_rules(L, P)["Product rule (NeuroFoldNet)"], expected)


# ----------------------------------------------------------------------------- conformal
def test_mondrian_conformal_coverage_on_exchangeable_data():
    ds = _load_script("run_decision_support")
    rng = np.random.default_rng(3)
    alpha, cover = 0.1, []
    for _ in range(200):
        y = rng.integers(0, 3, 600)
        logits = rng.normal(size=(600, 3)) + 1.5 * np.eye(3)[y]
        p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
        q = ds.mondrian_quantiles(y[:300], p[:300], alpha)
        sets = (1 - p[300:]) <= q[None, :]
        cover.append(sets[np.arange(300), y[300:]].mean())
    assert np.mean(cover) >= 1 - alpha - 0.01


# ----------------------------------------------------------------------------- determinism
def test_paper_number_export_is_deterministic():
    target = ROOT / "results" / "v2" / "paper_numbers.tex"
    if not target.exists():
        pytest.skip("results not generated")
    before = target.read_bytes()
    try:
        digests = []
        for _ in range(2):
            subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "scripts" / "export_paper_numbers.py")],
                           check=True, capture_output=True, cwd=ROOT)
            digests.append(hashlib.sha256(target.read_bytes()).hexdigest())
        assert digests[0] == digests[1]
        assert target.read_bytes() == before, "archived paper_numbers.tex is stale relative to the result files"
    finally:
        target.write_bytes(before)
