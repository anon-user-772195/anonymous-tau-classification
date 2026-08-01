# NeuroFoldNet Research Workflow

NeuroFoldNet is a research-grade biomedical ML workflow for multi-class classification of tau-associated neurodegenerative disease subtype:

- Alzheimer's disease (AD)
- Dementia with Lewy Bodies (DLB)
- Progressive Supranuclear Palsy (PSP)

The original tau polymorph features are reconstructed from the Nature Communications Biology supplement `42003_2025_7499_MOESM3_ESM.xlsx`. This repository also supports Figure 7 / GSE277105 preprocessing that derives sample-level transcriptomic summaries from scRNA-seq data.

## Required Data

Place these files in the project root data folders, or leave them in `C:\Users\shrey\Downloads` where the scripts can auto-discover them:

- `data/raw/42003_2025_7499_MOESM3_ESM.xlsx`
- `data/figure7_geo/GSE277105_counts.csv.gz`
- `data/figure7_geo/GSE277105_metadata.csv.gz`

The script detects the 10x ZIP name `GSE277105_combined_filtered_feature_bc_matrix.zip`, but the preferred and currently supported path is the counts/metadata CSV.gz pair because it preserves metadata needed for sample-level aggregation.

## Figure 7 / GSE277105 Features

Figure 7 adds auxiliary transcriptomic response features from GSE277105. Raw cell-level expression is not treated as independent disease-classification data. Cells are aggregated to biological sample/treatment level before any model training.

The preprocessing script:

- inspects count and metadata files,
- identifies shared cell barcodes,
- detects sample and disease/condition labels,
- uses excitatory neuron annotations if a clear cell-type column exists,
- otherwise continues with all relevant cells and logs a warning,
- creates sample-level pseudobulk expression profiles,
- applies CPM-style library-size normalization and `log1p`,
- computes compact variable-gene, module-score, global inspection PCA, and untreated-contrast features,
- writes `data/processed/figure7_geo_sample_features.csv`.

Run:

```bash
python scripts/prepare_figure7_geo_features.py
```

## Merge Tau And GEO Features

Run:

```bash
python scripts/merge_tau_and_geo_features.py
```

Output:

```text
data/processed/tau_features.csv
data/processed/tau_plus_figure7_features.csv
```

If exact tau and GEO sample IDs do not match, the merge script does not invent one-to-one pairings. It creates disease-level GEO summary features and marks them with `geo_auxiliary_level=disease_mean`. This mode is for exploratory analysis, not deployable diagnostic inference.

By default, `train.py` ignores unmatched disease-level auxiliary summaries during model evaluation. This prevents disease-level summaries from acting like label encodings. To run an explicitly exploratory separability analysis with those columns, add `--allow_disease_level_aux`.

## Model Evaluation

Run the original tau-only model:

```bash
python train.py --feature_mode tau_only
```

Run the tau + Figure 7 GEO exploratory model:

```bash
python train.py --feature_mode tau_plus_geo
```

Run the all-figures table while excluding unmatched disease-level summaries, which is the default behavior:

```bash
python train.py --feature_mode tau_full_supplement_plus_geo
```

Run an explicitly exploratory disease-summary separability analysis:

```bash
python train.py --feature_mode tau_full_supplement_plus_geo --classifier logreg_balanced --allow_disease_level_aux
```

Run nested model selection for the optimized tau-polymorph classifier:

```bash
python scripts/nested_model_selection.py --feature-mode tau_full_supplement_plus_geo
```

Create and evaluate the tau model with one disease-blind Figure 7 context feature:

```bash
python scripts/create_tau_plus_one_fig7_nonleaking_feature.py
python scripts/nested_model_selection.py --feature-mode tau_plus_one_fig7_nonleaking --output results/nested_model_selection_tau_plus_one_fig7_nonleaking_results.csv
```

Current optimized result:

```text
Accuracy: 95.0% +/- 11.2%
Macro-F1: 0.920 +/- 0.179
Log loss: 0.318 +/- 0.169
```

This result uses matched tau polymorph measurements. The optional one-feature Figure 7 context table uses `fig7_global_tau_neuronal_response`, a single disease-blind constant derived from treated Figure 7 profiles. It is included only as non-predictive context and does not assign disease-specific transcriptomic values to tau samples.

Outputs:

```text
results/cv_results_tau_only.csv
results/cv_results_tau_plus_geo.csv
results/model_comparison_summary.csv
results/cv_predictions_tau_only.csv
results/cv_predictions_tau_plus_geo.csv
```

Evaluation uses 5-fold `StratifiedGroupKFold`. The script asserts that no group appears in both train and validation folds. Scaling, imputation, and fold-local GEO PCA are fit inside each training fold only.

## Scientific Warning

This integration is intended for defensible research exploration. It is not a clinical model and should not be presented as diagnostic. The scRNA-seq branch avoids cell-level pseudoreplication by aggregating expression to biological sample/treatment level before modeling.

When tau and GEO samples do not map by exact sample ID, disease-level GEO summaries are auxiliary biological context. Any performance from that mode should be interpreted cautiously because the feature assignment uses disease-level labels rather than a prospective sample-level assay.

Current default training ignores those disease-level summary columns. If all Figure 1-7 summaries are enabled with `--allow_disease_level_aux`, inflated or perfect performance should be interpreted only as disease-profile separability, not validated sample-level classification.

## Web Demo

The existing Next.js/Flask demo remains under `UI design/`. The research scripts above are root-level and do not rewrite the UI.
