# Pre-submission audit log (2026-11-15)

Every entry gives the command that was run, the file it wrote and what changed. Results that existed before
this audit are preserved in `results/v2_snapshot_before_audit_2026-11-15/` (including the frozen model in
`_models/`); nothing in that folder was modified.

## A1. Bug: point mass in reconstructed AFM area particles (fixed)

* **Found by:** `tests/test_leakage_invariants.py::test_particle_reconstruction_removes_bin_grid` (new).
* **What:** particles are reconstructed by drawing uniformly within each published histogram bin. The first PSP
  area bin (centre 3.85 nm², width 18.15 nm²) straddles zero; negative draws were clipped to 1e-3 nm². This
  put 185 of 963 PSP area particles (19%) on one identical value that no other disease had, a class-specific
  artefact the KDE expert could exploit. No other disease/measure was affected.
* **Fix:** `neurofoldnet/assays.py::reconstruct_log_particles` truncates a bin that straddles zero to its
  positive part; `run_morphology_models.py` and `neurofoldnet_external.py` now share this function.
* **Rerun:** `run_morphology_models`, `run_multiassay_panel`, `run_fusion_analyses`, `run_decision_support`,
  `neurofoldnet_external freeze`, `export_paper_numbers`, `export_supplementary_tables`
  (log: `results/v2/rerun_after_morphology_fix.log`). Before/after values: section A5.

## A2. Seeding expert chosen post hoc (quantified, not hidden)

* **What:** `run_assay_models.py::run_seeding` evaluates logistic regression and shrinkage LDA with nested CV
  and keeps the one with the higher **outer-CV** balanced accuracy (0.683 vs 0.675). The manuscript said the
  model was "selected by inner cross-validation", which is not what the code did.
* **Check:** `python scripts/run_robustness_checks.py --only seeding_nested` selects the model family inside
  the inner CV (20 repeats x 3 outer folds). Output: `robustness_seeding_nested.csv`.
* **Result:** balanced accuracy 0.669 (SD 0.117), macro AUC 0.815; inner CV chose logistic regression in 88%
  of folds. Whole-pipeline group permutation test (1000 permutations): **p = 0.261**. The primary
  shrinkage-LDA expert had p = 0.020 (1000 permutations) and logistic regression p = 0.262
  (`permutation_tests_seeding.csv`). Fusion effect (`--only seeding_fusion_sensitivity`,
  `robustness_seeding_fusion.csv`): AFM + dyes + seeding with one measurement per source reached 93.5%
  with the primary expert and 91.4% with the nested one; with three measurements both reached 99.2%.
  The primary shrinkage-LDA OOF posteriors are saturated (many exactly 1.0/0.0). The significance of seeding on its own therefore depends on a choice
  made after seeing outer-CV results. It must be reported as fragile.

## A3. Within-preparation acquisition drift (exchangeability test)

* **Command:** `python scripts/run_robustness_checks.py --only drift --repeats 5 --n-null 200`
  → `robustness_drift.csv`.
* **Design:** within each dye x disease, shrinkage LDA separates the first from the second half of the
  acquisition order (group-aware 5-fold CV, blocks of 10); null = block-level permutation of half labels.
* **Result:** AUC 0.44-0.74 (mean 0.59). 7 of 12 tests have p <= 0.035, none survives Holm
  (min adjusted p 0.12), and Fisher's combined test is p = 4.1e-6. **Spectra drift mildly along the
  acquisition order.** Measurements of a preparation are therefore not fully exchangeable. Random
  resampling into "virtual preparations" may be optimistic, and the contiguous-block grouping is necessary
  but probably not sufficient.

## A4. Spectral CV block-size sensitivity

* **Command:** `python scripts/run_robustness_checks.py --only block_size --repeats 3`
  → `robustness_block_size.csv`. NeuroFoldNet spectral expert, balanced accuracy (chance 0.33):

  | Block size (groups per dye) | ThT | Curcumin | Bis-ANS | FSB |
  |---|---|---|---|---|
  | 1 (no grouping, about 443) | 0.41 | 0.45 | 0.40 | 0.49 |
  | 5 (about 90) | 0.39 | 0.47 | 0.39 | 0.47 |
  | **10 (45, primary)** | 0.37 | 0.44 | 0.38 | 0.48 |
  | 20 (24) | 0.34 | 0.40 | 0.30 | 0.45 |
  | 50 (9) | 0.07 | 0.20 | 0.06 | 0.15 |

  The ungrouped estimate is the most optimistic, which supports the manuscript's claim. Accuracy keeps
  falling as blocks grow: at 20, ThT and Bis-ANS reach chance while Curcumin and FSB keep about 0.40-0.45.
  Blocks of 50 leave 3 groups per class, so whole classes drop out of training folds and accuracy falls
  far below chance. That design is degenerate (prior shift, "anti-learning") and not interpretable.
  Together with A3, part of the spectral signal is local to the acquisition sequence. The block of 10 is a
  choice the data do not justify, and the manuscript should show this sensitivity.

## A5. Before/after values of the morphology fix

`results/v2_snapshot_before_audit_2026-11-15/paper_numbers.tex` vs `results/v2/paper_numbers.tex`
(selected; the full diff comes from `diff` on the two files):

| Macro | Before | After |
|---|---|---|
| morphology area balanced accuracy | 0.56 | 0.54 |
| AFM panel, 1 particle per measure | 80.4% | 79.8% |
| AFM + dyes, 1 per source | 84.0% | 83.4% |
| **AFM + dyes + seeding, 1 per source** | **93.7%** | **93.8%** |
| AFM + dyes + seeding, 3 per source | 99.3% | 99.2% |
| majority vote, all, 1 per source | 80.4% | 80.0% |
| product rule (split-half), all, 1 | 92.1% | 92.7% |
| learned stacking (split-half), all, 1 | 84.6% | 84.3% |
| ECE, all, 1 | 0.014 | 0.015 |
| conformal coverage, all, 1, 90% target | 86.6% | 86.1% |
| VOI vs fixed order, median measurements to 95% | 2 vs 8 | 2 vs 8 (unchanged) |

The bug was real but did not inflate the headline results, because area was the weakest AFM measure.
`manuscript/paper_numbers.tex` (used by the original manuscripts) was **not** overwritten. The revision copy
uses `manuscript/paper_numbers_rev1.tex`.

## A6. Citations and literal numbers in the manuscript

`python -X utf8 scripts/audit_manuscript_numbers.py manuscript/main_aim.tex` lists 52 literal numbers.
Each was traced by hand. Corrections:

| Text | Problem | Correction | Evidence |
|---|---|---|---|
| "more than 55 million people worldwide [Jack 2018]" | Jack et al. 2018 (NIA-AA framework) does not report prevalence | "57 million people were living with dementia in 2021" citing WHO Dementia fact sheet | https://www.who.int/news-room/fact-sheets/detail/dementia (accessed 2026-11-15) |
| AFM heights "2-8 nm" | uncited; the oligomers were seeded from AD brain oligomers using recombinant 2N4R tau | cite Dasari AKR, Bhatt N, Haque MA, Irving R, Kayed R, Lim KH. Biochem Biophys Res Commun 2025;776:152189. doi:10.1016/j.bbrc.2025.152189; describe as "brain-seeded" | PMC12335351 |
| "3.75 ± 0.86 nm for amplified AD oligomers" | uncited | cite Arar S, Haque MA, Kayed A, Khan S, Bhatt N, Zhao Y, et al. ACS Chem Neurosci 2025;16(15):2829-43. doi:10.1021/acschemneuro.5c00122 | PMC12333032 |
| "about three times those in the source data" | approximate | source-data AD height mean 1.41 nm (SD 0.40; bin centres 1.00-2.64 nm), so 3.75 nm is about 2.7-fold | `load_morphology_histograms()` |
| "selected by inner cross-validation" (seeding) | false (A2) | state post hoc choice and report the nested result | A2 |


KS D range 0.404-0.76 (`stats_fig1_morphology.csv`), rTauO ranges and the "18 GEO series"
(`external_dataset_search.csv`) were verified. With Holm correction across the four dyes, the
spectral permutation p-values are FSB 0.020, Curcumin 0.030, Bis-ANS 0.040 and ThT 0.095
(`permutation_tests_spectra.csv`).
