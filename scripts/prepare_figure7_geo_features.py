from __future__ import annotations

import argparse
import csv
import gzip
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_INPUT_DIRS = [
    PROJECT_ROOT / "data" / "figure7_geo",
    Path.home() / "Downloads",
]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "figure7_geo_sample_features.csv"

DISEASES = ("AD", "DLB", "PSP")
CONTROL_TOKENS = ("UNTREATED", "CONTROL", "CTRL")
CELL_ID_CANDIDATES = ("cell", "cell_id", "barcode", "barcodes")
SAMPLE_CANDIDATES = ("sample", "orig.ident", "sample_id", "gsm", "condition", "treatment")
CELL_TYPE_CANDIDATES = ("cell_type", "celltype", "annotation", "predicted.celltype", "seurat_clusters")
EXCITATORY_TOKENS = ("excit", "glutamatergic", "neuron_exc")

GENE_MODULES = {
    "synaptic": ["Snap25", "Syp", "Syn1", "Dlg4", "Camk2a", "Grin1", "Gria1", "Map2"],
    "stress": ["Fos", "Jun", "Egr1", "Atf3", "Ddit3", "Hspa1a", "Hspa1b"],
    "apoptosis": ["Bax", "Bcl2", "Casp3", "Casp8", "Trp53", "Bad"],
    "tau_neuronal": ["Mapt", "Tubb3", "Rbfox3", "Neun", "Mapt", "App", "Psen1"],
}


def canonical_cell_id(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"([.-])\d+$", "", text)
    return text.replace(".", "-")


def find_geo_files(input_dir: Path | None) -> dict[str, Path | None]:
    search_dirs = [input_dir] if input_dir else DEFAULT_INPUT_DIRS
    files = {"counts": None, "metadata": None, "zip": None}
    for directory in search_dirs:
        if not directory.exists():
            continue
        counts = sorted(directory.glob("GSE277105_counts.csv.gz"))
        metadata = sorted(directory.glob("GSE277105_metadata.csv.gz"))
        zipped = sorted(directory.glob("GSE277105_combined_filtered_feature_bc_matrix.zip"))
        if counts and metadata:
            files["counts"] = counts[0]
            files["metadata"] = metadata[0]
            return files
        if zipped and files["zip"] is None:
            files["zip"] = zipped[0]

    if files["zip"] and not (files["counts"] and files["metadata"]):
        raise FileNotFoundError(
            "Found the 10x ZIP but not the preferred counts/metadata CSV.gz pair. "
            "This script expects GSE277105_counts.csv.gz and GSE277105_metadata.csv.gz "
            "so sample-level metadata can be joined unambiguously."
        )

    searched = "\n".join(f"  - {d}" for d in search_dirs)
    raise FileNotFoundError(
        "Could not find GSE277105_counts.csv.gz and GSE277105_metadata.csv.gz. "
        f"Searched:\n{searched}"
    )


def detect_cell_id_column(metadata: pd.DataFrame) -> str:
    unnamed = [c for c in metadata.columns if str(c).startswith("Unnamed") or str(c) == ""]
    candidates = unnamed + [
        c
        for c in metadata.columns
        if any(token in str(c).lower() for token in CELL_ID_CANDIDATES)
    ]
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        print("Ambiguous cell ID columns:", candidates)
        raise ValueError("Pass --cell-id-column to choose the metadata cell/barcode column.")
    return metadata.columns[0]


def detect_sample_column(metadata: pd.DataFrame, requested: str | None = None) -> str:
    if requested:
        if requested not in metadata.columns:
            raise ValueError(f"Requested sample column {requested!r} is not in metadata.")
        return requested
    candidates = [
        c
        for c in metadata.columns
        if any(token == str(c).lower() or token in str(c).lower() for token in SAMPLE_CANDIDATES)
    ]
    candidates = list(dict.fromkeys(candidates))
    useful = [c for c in candidates if metadata[c].nunique(dropna=True) > 1]
    if len(useful) == 1:
        return useful[0]
    if "sample" in metadata.columns:
        return "sample"
    print("Sample/condition column candidates:")
    for col in useful:
        print(f"  {col}: {metadata[col].nunique(dropna=True)} unique values")
        print(metadata[col].value_counts(dropna=False).head(10).to_string())
    raise ValueError("Could not uniquely identify sample/condition column. Pass --sample-column.")


def detect_cell_type_column(metadata: pd.DataFrame, requested: str | None = None) -> str | None:
    if requested:
        if requested not in metadata.columns:
            raise ValueError(f"Requested cell-type column {requested!r} is not in metadata.")
        return requested
    candidates = [
        c
        for c in metadata.columns
        if any(token in str(c).lower() for token in CELL_TYPE_CANDIDATES)
    ]
    text_candidates = [
        c for c in candidates if metadata[c].astype(str).str.contains("|".join(EXCITATORY_TOKENS), case=False).any()
    ]
    if len(text_candidates) == 1:
        return text_candidates[0]
    if len(text_candidates) > 1:
        print("Ambiguous cell-type columns with excitatory-like values:", text_candidates)
        raise ValueError("Pass --cell-type-column to choose one.")
    return None


def disease_from_sample(sample: object) -> str | None:
    upper = str(sample).upper()
    for disease in DISEASES:
        if re.search(rf"(^|[^A-Z]){disease}([^A-Z]|$)", upper):
            return disease
    if upper.startswith("UT_") or upper.endswith("_UT") or "_UT_" in upper:
        return "UNTREATED"
    if any(token in upper for token in CONTROL_TOKENS):
        return "UNTREATED"
    return None


def genotype_from_sample(sample: object) -> str:
    upper = str(sample).upper()
    if "HTAU" in upper or "MAPT" in upper:
        return "hTau"
    if "C57" in upper:
        return "C57BL/6"
    return "unknown"


def inspect_inputs(counts_path: Path, metadata: pd.DataFrame, sample_col: str) -> None:
    with gzip.open(counts_path, "rt", newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        first = next(reader)
    print("=" * 80)
    print("GSE277105 input inspection")
    print("=" * 80)
    print(f"Counts file: {counts_path}")
    print(f"Counts columns: {len(header)} (gene column + {len(header) - 1} cells)")
    print(f"First gene row preview: {first[:8]}")
    print(f"Metadata shape: {metadata.shape}")
    print(f"Metadata columns: {list(metadata.columns)}")
    print("Sample distribution:")
    print(metadata[sample_col].value_counts(dropna=False).to_string())
    diseases = metadata[sample_col].map(disease_from_sample)
    print("Parsed condition distribution:")
    print(diseases.value_counts(dropna=False).to_string())


def prepare_metadata(
    metadata_path: Path,
    sample_column: str | None,
    cell_id_column: str | None,
    cell_type_column: str | None,
) -> tuple[pd.DataFrame, str, str]:
    metadata = pd.read_csv(metadata_path, compression="gzip")
    if cell_id_column is None:
        cell_id_column = detect_cell_id_column(metadata)
    if sample_column is None:
        sample_column = detect_sample_column(metadata)
    detected_cell_type = detect_cell_type_column(metadata, cell_type_column)

    metadata = metadata.copy()
    metadata["canonical_cell_id"] = metadata[cell_id_column].map(canonical_cell_id)
    metadata["sample_id"] = metadata[sample_column].astype(str)
    metadata["disease"] = metadata["sample_id"].map(disease_from_sample)
    metadata["genotype"] = metadata["sample_id"].map(genotype_from_sample)

    metadata = metadata[metadata["disease"].isin([*DISEASES, "UNTREATED"])].copy()
    if detected_cell_type:
        mask = metadata[detected_cell_type].astype(str).str.contains("|".join(EXCITATORY_TOKENS), case=False, na=False)
        if mask.any():
            print(f"Using excitatory-neuron subset from column {detected_cell_type!r}: {int(mask.sum())} cells")
            metadata = metadata[mask].copy()
        else:
            print(f"WARNING: no excitatory-like entries found in {detected_cell_type!r}; continuing with all cells.")
    else:
        print("WARNING: no clear cell-type annotation found; continuing with all relevant cells.")

    if metadata.empty:
        raise ValueError("No AD/DLB/PSP/untreated cells remain after filtering.")

    print(f"Selected metadata rows: {metadata.shape[0]}")
    print("Selected sample distribution:")
    print(metadata["sample_id"].value_counts().to_string())
    return metadata, sample_column, cell_id_column


def build_column_mapping(counts_path: Path, metadata: pd.DataFrame) -> tuple[list[str], dict[int, str]]:
    wanted = dict(zip(metadata["canonical_cell_id"], metadata["sample_id"]))
    with gzip.open(counts_path, "rt", newline="", encoding="utf-8", errors="replace") as handle:
        header = next(csv.reader(handle))

    index_to_sample = {}
    for idx, cell_id in enumerate(header[1:], start=1):
        sample = wanted.get(canonical_cell_id(cell_id))
        if sample is not None:
            index_to_sample[idx] = sample

    matched_cells = len(index_to_sample)
    if matched_cells == 0:
        raise ValueError("No shared cell barcodes found between counts and metadata.")
    print(f"Matched count matrix cells to metadata: {matched_cells:,}")
    return header, index_to_sample


def aggregate_expression(
    counts_path: Path,
    metadata: pd.DataFrame,
    index_to_sample: dict[int, str],
    top_genes: int,
) -> pd.DataFrame:
    print("Aggregating cell-level raw counts into sample-level pseudobulk profiles...", flush=True)
    with gzip.open(counts_path, "rt", newline="", encoding="utf-8", errors="replace") as handle:
        header = next(csv.reader(handle))

    unique_samples = sorted(set(index_to_sample.values()))
    sample_to_code = {sample: idx for idx, sample in enumerate(unique_samples)}
    group_codes = np.full(len(header) - 1, -1, dtype=np.int32)
    for one_based_idx, sample in index_to_sample.items():
        group_codes[one_based_idx - 1] = sample_to_code[sample]
    selected_mask = group_codes >= 0
    selected_codes = group_codes[selected_mask]

    genes: list[str] = []
    rows: list[np.ndarray] = []
    with gzip.open(counts_path, "rb") as handle:
        next(handle)
        for gene_idx, raw_line in enumerate(handle, start=1):
            gene_raw, values_raw = raw_line.rstrip(b"\r\n").split(b",", 1)
            values = np.fromstring(values_raw, sep=",", dtype=np.float64)
            if values.shape[0] != len(group_codes):
                raise ValueError(
                    f"Count row for {gene_raw!r} has {values.shape[0]} values; expected {len(group_codes)}."
                )
            sample_counts = np.bincount(
                selected_codes,
                weights=values[selected_mask],
                minlength=len(unique_samples),
            )
            genes.append(gene_raw.decode("utf-8", errors="replace"))
            rows.append(sample_counts)
            if gene_idx % 5000 == 0:
                print(f"  aggregated {gene_idx:,} genes", flush=True)

    raw_pseudobulk = pd.DataFrame(rows, index=genes, columns=unique_samples)
    library_sizes = raw_pseudobulk.sum(axis=0).replace(0, 1.0)
    expr = np.log1p(raw_pseudobulk.div(library_sizes, axis=1) * 10000.0).T
    print(f"Pseudobulk expression matrix: {expr.shape[0]} samples × {expr.shape[1]} genes", flush=True)
    variances = expr.var(axis=0).sort_values(ascending=False)
    keep_genes = variances.head(top_genes).index.tolist()
    print(f"Selected top {len(keep_genes)} variable genes for compact expression features.")

    features = expr[keep_genes].copy()
    features.columns = [f"geo_expr_{safe_feature_name(gene)}" for gene in features.columns]

    for module_name, genes_in_module in GENE_MODULES.items():
        present = [gene for gene in genes_in_module if gene in expr.columns]
        col = f"geo_module_{module_name}"
        features[col] = expr[present].mean(axis=1) if present else np.nan
        print(f"Module {module_name}: {len(present)} genes present")

    n_components = min(5, features.shape[0] - 1, len(keep_genes))
    if n_components >= 1:
        scaled = StandardScaler().fit_transform(expr[keep_genes])
        pca_values = PCA(n_components=n_components, random_state=42).fit_transform(scaled)
        for idx in range(n_components):
            features[f"geo_pca_global_{idx + 1}"] = pca_values[:, idx]
        print(
            "Added unsupervised global GEO PCA columns for inspection. "
            "train.py recomputes fold-local PCA from geo_expr_* columns."
        )

    sample_meta = metadata.groupby("sample_id").agg(
        disease=("disease", "first"),
        genotype=("genotype", "first"),
        n_cells=("canonical_cell_id", "count"),
    )
    out = sample_meta.join(features, how="inner").reset_index()
    out["group"] = out["sample_id"]
    out["is_untreated_control"] = out["disease"].eq("UNTREATED")
    out["geo_feature_level"] = "sample"
    ordered = ["sample_id", "group", "disease", "genotype", "n_cells", "is_untreated_control", "geo_feature_level"]
    return out[ordered + [c for c in out.columns if c not in ordered]]


def safe_feature_name(gene: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", gene).strip("_")


def add_contrast_features(features: pd.DataFrame) -> pd.DataFrame:
    controls = features[features["is_untreated_control"]]
    treated = features[~features["is_untreated_control"]].copy()
    numeric_cols = [c for c in features.columns if c.startswith(("geo_expr_", "geo_module_", "geo_pca_global_"))]
    if controls.empty:
        print("WARNING: untreated controls were not available; contrast features were not added.")
        return treated

    control_by_genotype = controls.groupby("genotype")[numeric_cols].mean()
    global_control = controls[numeric_cols].mean()
    for idx, row in treated.iterrows():
        baseline = control_by_genotype.loc[row["genotype"]] if row["genotype"] in control_by_genotype.index else global_control
        for col in numeric_cols:
            treated.loc[idx, f"{col}_vs_untreated"] = row[col] - baseline[col]

    print("Added treatment-minus-untreated contrast features by genotype where available.")
    return treated


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare sample-level Figure 7/GSE277105 features.")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-column", default=None)
    parser.add_argument("--cell-id-column", default=None)
    parser.add_argument("--cell-type-column", default=None)
    parser.add_argument("--top-genes", type=int, default=50)
    parser.add_argument("--include-controls", action="store_true", help="Keep untreated controls in output.")
    args = parser.parse_args()

    files = find_geo_files(args.input_dir)
    counts_path = files["counts"]
    metadata_path = files["metadata"]
    assert counts_path is not None and metadata_path is not None

    metadata, sample_col, cell_col = prepare_metadata(
        metadata_path, args.sample_column, args.cell_id_column, args.cell_type_column
    )
    inspect_inputs(counts_path, pd.read_csv(metadata_path, compression="gzip"), sample_col)
    _, index_to_sample = build_column_mapping(counts_path, metadata)
    features = aggregate_expression(counts_path, metadata, index_to_sample, args.top_genes)
    features = add_contrast_features(features)

    if not args.include_controls:
        features = features[features["disease"].isin(DISEASES)].copy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False)
    print("=" * 80)
    print(f"Saved Figure 7 GEO sample-level features: {args.output}")
    print(f"Output shape: {features.shape}")
    print("Output disease distribution:")
    print(features["disease"].value_counts().to_string())
    print("Cells were aggregated to biological sample/treatment level; no cell is a classifier sample.")


if __name__ == "__main__":
    main()
