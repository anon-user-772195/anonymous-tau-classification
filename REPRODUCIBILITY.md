# Reproducibility guide and checklist

## 1. Environment

| Item | Value |
|---|---|
| Python | 3.14.1 (CPython, Windows 11; no GPU needed) |
| Packages | pinned in `requirements.txt`: numpy 2.3.5, pandas 2.3.3, scipy 1.17.0, scikit-learn 1.8.0, matplotlib 3.10.8, openpyxl 3.1.5, joblib 1.5.3; pytest 9.0.1 for the tests |
| Hardware used | laptop CPU; the full run takes about 2-3 h, dominated by permutation tests |

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt pytest==9.0.1
```

## 2. Data provenance

| Source | Identifier | Used for |
|---|---|---|
| Lo Cascio et al. 2025, Commun Biol 8:53, Supplementary Data (source-data workbook) | `data/raw/42003_2025_7499_MOESM3_ESM.xlsx`, doi:10.1038/s42003-025-07499-y | all assay experts |
| GEO GSE277105 | NCBI GEO | Fig 7 cross-genotype analysis |

The workbook holds figure-level source data. It has **no donor or sample identifier shared between figures**,
and the source study used three donor brains per disease. No identifiable patient information is present
or used.

## 3. One-command reproduction

```bash
python scripts/run_all.py            # every result, figure, frozen model and LaTeX macro, in dependency order
python -m pytest -q tests            # 15 leakage/correctness invariants (about 20 s)
```

`run_all.py` logs each step's exact command line, timing and exit code to `results/v2/logs/`.
`python scripts/run_all.py --quick` runs a smoke test.

## 4. Split definitions and model selection

| Assay | Unit | CV grouping | Outer / inner folds | Repeats | Model selection |
|---|---|---|---|---|---|
| Dye spectra (4 dyes) | spectrum | contiguous blocks of 10 spectra in acquisition order, within disease | 5 / 3 (StratifiedGroupKFold) | 5 | NeuroFoldNet ensemble fixed a priori; baselines tuned by inner grid search on log loss |
| AFM morphology | reconstructed particle | none (particles reconstructed from histograms) | 5 (StratifiedKFold) | 5 reconstructions | KDE bandwidth: Scott's rule, no tuning |
| Seeding | replicate (5 antibody readouts) | replicate | 3 / 2 | 20 | hyper-parameters by inner CV; **model family chosen post hoc** (see `results/v2/AUDIT_LOG.md` A2; the nested alternative is in `robustness_seeding_nested.csv`) |
| Fusion | virtual preparation | measurements drawn without replacement from held-out OOF posteriors | — | 500 draws | product rule (no parameters); learned stacking is evaluated split-half |

Seeds are fixed in each script: spectra `seed=1`, seeding `seed=11`, morphology `100 + repeat`, robustness `20260`.

## 5. Output map

| Manuscript item | Produced by | File |
|---|---|---|
| every number in the text (LaTeX macros) | `export_paper_numbers.py` | `results/v2/paper_numbers.tex` → `manuscript/paper_numbers.tex` |
| Supplementary Tables S1-S20 | `export_supplementary_tables.py` | `manuscript/supplementary_tables.tex` |
| fusion rules, ablation, calibration | `run_fusion_analyses.py` | `fusion_*.csv`, `figures/fusion_ablation_calibration.png` |
| conformal sets, novelty detection, sequential acquisition | `run_decision_support.py` | `conformal_*.csv`, `novelty_detection.csv`, `sequential_*.csv`, `figures/decision_support.png` |
| robustness (audit) | `run_robustness_checks.py` | `robustness_*.csv` |

## 6. Tests (`tests/test_leakage_invariants.py`)

* No CV group appears in both the training and test data, in outer folds or in the inner grid search.
* Every sample gets exactly one out-of-fold prediction per repeat.
* Group-level permutation keeps one label per group and the class counts.
* The parsers return the expected shapes (4 dyes x about 443 spectra x 30 wavelengths; 18 seeding replicates).
* Particle reconstruction removes the bin grid and creates no point masses. This test caught the bug
  described in AUDIT_LOG A1.
* Fusion draws never reuse a measurement within one draw.
* The product rule equals the normalised product of geometric-mean posteriors.
* Mondrian conformal sets reach nominal coverage on exchangeable synthetic data.
* `export_paper_numbers.py` is deterministic, and the archived macros match the result files.

## 7. TRIPOD+AI mapping (Collins et al., BMJ 2024;385:e078378)

TRIPOD+AI is written for clinical prediction models in patients. The present work classifies
**disease-derived protein preparations** and does not predict anything for individual patients, so several
items cannot apply. They are listed here rather than left out silently.

| Item | Topic | Status | Where / gap |
|---|---|---|---|
| 1 | Title identifies a prediction-model study, target population and outcome | partial | title says "preparations"; should not imply patients |
| 2 | Abstract | yes | main text |
| 3a-c | Background, rationale, existing models | yes | Introduction |
| 4 | Objectives, development vs validation | yes | development and internal validation only |
| 5a-b | Data sources, dates | partial | source study and GEO; acquisition dates not available |
| 6a-c | Setting, eligibility, treatment | **no** | donors, their selection and clinical details are only in the source study; 3 donors per disease |
| 7 | Data preparation | yes | Methods, Supplementary S2 |
| 8a-c | Outcome definition and blinding | partial | neuropathological diagnosis per source study; assessor blinding unknown |
| 9a-c | Predictors | yes | per-assay features |
| 10 | Sample size | **no** | not justified; n is fixed by the source data |
| 11 | Missing data | yes | none imputed; assays never merged |
| 12a-g | Analytical methods | yes | Methods; nested CV; fusion |
| 12f | Fairness | n/a | no patient-level demographics |
| 13 | Class imbalance | yes | balanced by design |
| 14 | Fairness approach | n/a | — |
| 15 | Model output | yes | calibrated posteriors, conformal sets |
| 16 | Training vs evaluation data differences | partial | rTauO and Fig 7 genotype checks only |
| 17 | Ethics | yes | public data only |
| 18a-c | Open science: funding, conflicts, protocol, registration | partial | no protocol or registration; analyses exploratory |
| 19 | Patient and public involvement | **no** | not applicable or not done; must be stated |
| 20a-b | Participants, flow | **no** | the donor level cannot be reported |
| 21 | Model development counts | yes | |
| 22 | Full model specification | yes | code and frozen model |
| 23a-b | Performance with CIs, heterogeneity | yes | cluster bootstrap; per-assay |
| 24 | Model updating | n/a | |
| 25-27 | Interpretation, limitations, usability | yes | Discussion |

## 8. Known non-reproducible elements

* None of the reported numbers depends on unseeded randomness. Minor floating-point differences across BLAS
  builds can change the last digit of SVM probabilities.
