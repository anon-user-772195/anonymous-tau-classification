"""Audit Figure 7 / GSE277105 metadata for biological replicate identifiers.

This script checks whether the downloaded scRNA-seq files contain mouse, brain,
donor, replicate, or library identifiers that would support brain-held-out
cell-level evaluation. It does not train a model.
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import pandas as pd


REPLICATE_HINTS = (
    "brain",
    "mouse",
    "mice",
    "animal",
    "donor",
    "replicate",
    "rep",
    "subject",
    "biosample",
    "sample",
    "library",
    "run",
    "batch",
)


def read_tsv_gz_head(path: Path, n: int = 5) -> list[str]:
    rows: list[str] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for _ in range(n):
            line = handle.readline()
            if not line:
                break
            rows.append(line.strip())
    return rows


def infer_sample_label(value: str) -> str:
    value = str(value)
    if value.endswith("_filtered_feature_bc_matrix"):
        value = value.removesuffix("_filtered_feature_bc_matrix")
    return value


def summarize_column(df: pd.DataFrame, column: str) -> dict[str, object]:
    series = df[column]
    return {
        "column": column,
        "non_null": int(series.notna().sum()),
        "unique_values": int(series.nunique(dropna=True)),
        "example_values": "; ".join(map(str, series.dropna().astype(str).unique()[:8])),
    }


def audit_metadata(metadata_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_csv(metadata_path)
    column_summary = pd.DataFrame(
        [summarize_column(metadata, column) for column in metadata.columns]
    )
    candidate_columns = [
        column
        for column in metadata.columns
        if any(hint in column.lower() for hint in REPLICATE_HINTS)
    ]
    candidate_summary = column_summary[column_summary["column"].isin(candidate_columns)].copy()
    return column_summary, candidate_summary


def audit_runinfo(runinfo_path: Path | None) -> pd.DataFrame:
    if runinfo_path is None or not runinfo_path.exists():
        return pd.DataFrame()
    runinfo = pd.read_csv(runinfo_path)
    candidate_columns = [
        column
        for column in runinfo.columns
        if any(hint in column.lower() for hint in REPLICATE_HINTS)
    ]
    return pd.DataFrame([summarize_column(runinfo, column) for column in candidate_columns])


def audit_10x_matrix(matrix_root: Path | None) -> pd.DataFrame:
    if matrix_root is None or not matrix_root.exists():
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for barcode_path in sorted(matrix_root.rglob("barcodes.tsv.gz")):
        sample = barcode_path.parent.name
        first_barcodes = read_tsv_gz_head(barcode_path, n=5)
        rows.append(
            {
                "condition_folder": sample,
                "sample": infer_sample_label(sample),
                "barcode_file": str(barcode_path),
                "example_barcodes": "; ".join(first_barcodes),
                "barcode_has_extra_replicate_token": any(
                    len(barcode.split("_")) > 1 for barcode in first_barcodes
                ),
            }
        )
    return pd.DataFrame(rows)


def write_note(
    note_path: Path,
    metadata_columns: pd.DataFrame,
    metadata_candidates: pd.DataFrame,
    runinfo_candidates: pd.DataFrame,
    tenx_summary: pd.DataFrame,
) -> None:
    note_path.parent.mkdir(parents=True, exist_ok=True)

    sample_columns = metadata_columns[metadata_columns["column"].isin(["sample", "orig.ident"])]
    sample_lines = []
    for _, row in sample_columns.iterrows():
        sample_lines.append(
            f"- {row['column']}: {row['unique_values']} unique values "
            f"({row['example_values']})"
        )

    candidate_lines = []
    if metadata_candidates.empty:
        candidate_lines.append("- Metadata candidate replicate columns: none")
    else:
        for _, row in metadata_candidates.iterrows():
            candidate_lines.append(
                f"- Metadata {row['column']}: {row['unique_values']} unique values "
                f"({row['example_values']})"
            )

    runinfo_lines = []
    if runinfo_candidates.empty:
        runinfo_lines.append("- SRA runinfo candidate replicate columns: none or unavailable")
    else:
        for _, row in runinfo_candidates.iterrows():
            runinfo_lines.append(
                f"- SRA {row['column']}: {row['unique_values']} unique values "
                f"({row['example_values']})"
            )

    tenx_lines = []
    if tenx_summary.empty:
        tenx_lines.append("- 10x barcode folders: not found")
    else:
        tenx_lines.append(f"- 10x condition folders found: {len(tenx_summary)}")
        tenx_lines.append(
            "- Barcode examples do not encode mouse/brain IDs; they are standard 10x "
            "cell barcodes within condition folders."
        )

    note = "\n".join(
        [
            "Figure 7 Replicate Audit",
            "",
            "Conclusion: the locally downloaded GSE277105 Figure 7 files do not contain "
            "brain-, mouse-, donor-, or replicate-level identifiers. The available "
            "metadata identify cells and condition-level samples, not independent brains "
            "within each disease treatment.",
            "",
            "Metadata sample fields:",
            *sample_lines,
            "",
            "Candidate replicate fields:",
            *candidate_lines,
            "",
            "SRA/runinfo fields:",
            *runinfo_lines,
            "",
            "10x barcode audit:",
            *tenx_lines,
            "",
            "Implication: cell barcodes cannot be used as biological groups for disease "
            "classification. A cell-level split would measure recognition of cells from "
            "the same condition/library and would inflate performance. A held-out-brain "
            "AD/DLB/PSP evaluation requires an external metadata table that maps each "
            "cell barcode to a mouse, brain, donor, or biological replicate.",
            "",
        ]
    )
    note_path.write_text(note, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/figure7_geo/GSE277105_metadata.csv.gz")
    parser.add_argument("--runinfo", default="data/figure7_geo/PRJNA1160803_runinfo.csv")
    parser.add_argument(
        "--tenx-root",
        default=r"C:\Users\shrey\Downloads\GSE277105_combined_filtered_feature_bc_matrix",
    )
    parser.add_argument("--out-dir", default="results")
    parser.add_argument(
        "--note", default="manuscript/figure7_replicate_audit_note.txt"
    )
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    runinfo_path = Path(args.runinfo)
    tenx_root = Path(args.tenx_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    metadata_columns, metadata_candidates = audit_metadata(metadata_path)
    runinfo_candidates = audit_runinfo(runinfo_path)
    tenx_summary = audit_10x_matrix(tenx_root)

    metadata_columns.to_csv(out_dir / "figure7_metadata_column_audit.csv", index=False)
    metadata_candidates.to_csv(out_dir / "figure7_metadata_replicate_candidates.csv", index=False)
    runinfo_candidates.to_csv(out_dir / "figure7_runinfo_replicate_candidates.csv", index=False)
    tenx_summary.to_csv(out_dir / "figure7_10x_barcode_audit.csv", index=False)

    write_note(
        Path(args.note),
        metadata_columns,
        metadata_candidates,
        runinfo_candidates,
        tenx_summary,
    )

    print(f"Metadata columns audited: {len(metadata_columns)}")
    print("Sample/condition fields:")
    print(metadata_columns[metadata_columns["column"].isin(["sample", "orig.ident"])].to_string(index=False))
    print("\nMetadata replicate candidates:")
    if metadata_candidates.empty:
        print("None")
    else:
        print(metadata_candidates.to_string(index=False))
    print("\nSRA/runinfo replicate candidates:")
    if runinfo_candidates.empty:
        print("None")
    else:
        print(runinfo_candidates.to_string(index=False))
    print(f"\n10x barcode folders audited: {len(tenx_summary)}")
    if not tenx_summary.empty:
        print(tenx_summary[["sample", "example_barcodes"]].to_string(index=False))
    print(f"\nWrote audit note: {args.note}")


if __name__ == "__main__":
    main()
