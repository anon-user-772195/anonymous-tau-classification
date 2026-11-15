"""Convert the GSE277105 dense genes x cells CSV into a compressed sparse matrix.

Output (data/figure7_geo/):
  GSE277105_counts_cells_x_genes.npz  scipy.sparse CSR, cells x genes, raw UMI counts
  GSE277105_genes.txt                 gene names (column order)
  GSE277105_cells.txt                 cell IDs (row order, canonicalised to match metadata)

Run once:  python scripts/build_fig7_sparse.py
"""
from __future__ import annotations

import gzip
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
GEO = ROOT / "data" / "figure7_geo"
SRC = GEO / "GSE277105_counts.csv.gz"


def canonical(cell: str) -> str:
    # counts header uses "BARCODE.1"; metadata uses "BARCODE-1"
    cell = cell.strip().strip('"')
    if cell.endswith(".1"):
        cell = cell[:-2] + "-1"
    return cell


def main() -> None:
    t0 = time.time()
    genes: list[str] = []
    data_chunks, row_chunks = [], []  # gene-major triplets, transposed at the end
    col_chunks = []
    with gzip.open(SRC, "rt", encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split(",")
        cells = [canonical(c) for c in header[1:]]
        n_cells = len(cells)
        for g, line in enumerate(fh):
            name, rest = line.split(",", 1)
            values = np.fromstring(rest, sep=",", dtype=np.float32)  # noqa: NPY201 (fast path)
            if values.size != n_cells:
                raise ValueError(f"gene {name}: {values.size} values, expected {n_cells}")
            nz = np.flatnonzero(values)
            genes.append(name.strip().strip('"'))
            data_chunks.append(values[nz])
            row_chunks.append(nz.astype(np.int32))
            col_chunks.append(np.full(nz.size, g, dtype=np.int32))
            if g % 2000 == 0:
                print(f"  {g} genes parsed ({time.time() - t0:.0f}s)", flush=True)
    data = np.concatenate(data_chunks)
    rows = np.concatenate(row_chunks)
    cols = np.concatenate(col_chunks)
    mat = sp.csr_matrix((data, (rows, cols)), shape=(n_cells, len(genes)), dtype=np.float32)
    sp.save_npz(GEO / "GSE277105_counts_cells_x_genes.npz", mat, compressed=True)
    (GEO / "GSE277105_genes.txt").write_text("\n".join(genes), encoding="utf-8")
    (GEO / "GSE277105_cells.txt").write_text("\n".join(cells), encoding="utf-8")
    print(f"Saved {mat.shape[0]} cells x {mat.shape[1]} genes, nnz={mat.nnz} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
