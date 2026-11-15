"""Layout-correct parsers for the Lo Cascio et al. 2025 (Commun Biol 8:53) source-data workbook.

Every parser returns observations exactly as they appear in the workbook. No values are
imputed, interpolated, broadcast across rows, or joined across assays: the workbook
contains no sample identifier shared between figures, so assays are never merged into
a single per-sample vector.

Layout notes (verified against the sheet headers):
  Fig 1D  per-disease histograms: (bin centre, count) for height / area / diameter.
  Fig 1E  modal values only (one number per disease, repeated) -> not used as features.
  Fig 2A  normalised emission spectra of four conformation-sensitive dyes (ThT, Curcumin,
          Bis-ANS, FSB); 30 wavelengths x 150 spectra per disease (+50 recombinant tau, rTauO).
  Fig 3B  proteolysis time course (0/1/5/15 min) at two protease doses; one curve per disease.
  Fig 4D  seeding readouts: 5 antibodies (TOMA1-4, T22) x 6 replicate columns per disease.
  Fig 5E-G  per-recording electrophysiology values at 12 h and 24 h (unpaired between panels).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "data" / "raw" / "42003_2025_7499_MOESM3_ESM.xlsx"
DISEASES = ("AD", "DLB", "PSP")

# Fig 2: row index of the first wavelength in each dye block, 30 wavelengths each
DYE_BLOCKS = {"ThT": 3, "Curcumin": 36, "Bis-ANS": 69, "FSB": 102}
N_WAVELENGTHS = 30
SPECTRA_COLUMNS = {"AD": range(1, 151), "DLB": range(151, 301), "PSP": range(301, 451), "rTauO": range(451, 501)}


def _sheet(name: str, workbook: Path = WORKBOOK) -> pd.DataFrame:
    return pd.read_excel(workbook, sheet_name=name, header=None)


def load_spectra(dye: str, include_control: bool = False, workbook: Path = WORKBOOK) -> pd.DataFrame:
    """One row per measured spectrum. Columns: label, spectrum_index, block, then one column per wavelength."""
    if dye not in DYE_BLOCKS:
        raise KeyError(f"dye must be one of {list(DYE_BLOCKS)}")
    df = _sheet("Fig 2", workbook)
    start = DYE_BLOCKS[dye]
    header_row = df.iloc[start - 1]
    if not str(header_row.iloc[0]).lower().startswith("emission"):
        raise ValueError(f"Unexpected Fig 2 layout above row {start} for {dye}")
    wl = pd.to_numeric(df.iloc[start:start + N_WAVELENGTHS, 0], errors="raise").astype(int).to_numpy()
    rows = []
    labels = list(DISEASES) + (["rTauO"] if include_control else [])
    for label in labels:
        for k, col in enumerate(SPECTRA_COLUMNS[label]):
            values = pd.to_numeric(df.iloc[start:start + N_WAVELENGTHS, col], errors="coerce").to_numpy(float)
            if np.isnan(values).all():
                continue
            if np.isnan(values).any():
                raise ValueError(f"Partially missing spectrum: {dye} {label} column {col}")
            rows.append({"label": label, "spectrum_index": k, **{f"wl_{w}": v for w, v in zip(wl, values)}})
    out = pd.DataFrame(rows)
    out.insert(2, "dye", dye)
    return out


def wavelength_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith("wl_")]


def load_seeding(workbook: Path = WORKBOOK) -> pd.DataFrame:
    """Fig 4D: one row per replicate column (6 per disease), five antibody readouts per row."""
    df = _sheet("Fig 4", workbook)
    antibodies = ["".join(str(v).split()) for v in df.iloc[2:7, 0]]  # also drops non-breaking spaces
    if antibodies != ["TOMA1", "TOMA2", "TOMA3", "TOMA4", "T22"]:
        raise ValueError(f"Unexpected Fig 4 antibody rows: {antibodies}")
    columns = {"AD": range(1, 7), "DLB": range(7, 13), "PSP": range(13, 19)}
    rows = []
    for label, cols in columns.items():
        for k, col in enumerate(cols):
            vals = pd.to_numeric(df.iloc[2:7, col], errors="raise").to_numpy(float)
            rows.append({"label": label, "replicate": k, **dict(zip(antibodies, vals))})
    return pd.DataFrame(rows)


def load_electrophysiology(workbook: Path = WORKBOOK) -> pd.DataFrame:
    """Fig 5E-G in long format: panel, timepoint, label, value (one row per recording)."""
    df = _sheet("Fig 5", workbook)
    panels = {"5E": {"12hr": 0, "24hr": 4}, "5F": {"12hr": 8, "24hr": 12}, "5G": {"12hr": 16, "24hr": 20}}
    rows = []
    for panel, times in panels.items():
        for tp, start in times.items():
            for j, label in enumerate(DISEASES):
                if str(df.iat[2, start + j]).strip() != label:
                    raise ValueError(f"Fig 5 header mismatch at panel {panel} {tp}")
                vals = pd.to_numeric(df.iloc[3:, start + j], errors="coerce").dropna()
                rows += [{"panel": panel, "timepoint": tp, "label": label, "value": float(v)} for v in vals]
    return pd.DataFrame(rows)


def load_morphology_histograms(workbook: Path = WORKBOOK) -> pd.DataFrame:
    """Fig 1D histograms in long format: label, measure, bin_centre, count."""
    df = _sheet("Fig 1", workbook)
    starts = {"AD": 0, "DLB": 7, "PSP": 14}
    rows = []
    for label, s in starts.items():
        if str(df.iat[1, s]).strip() != label:
            raise ValueError(f"Fig 1 header mismatch for {label}")
        for offset, measure in [(0, "height"), (2, "area"), (4, "diameter")]:
            centre = pd.to_numeric(df.iloc[3:, s + offset], errors="coerce")
            count = pd.to_numeric(df.iloc[3:, s + offset + 1], errors="coerce")
            ok = centre.notna() & count.notna()
            rows += [
                {"label": label, "measure": measure, "bin_centre": float(c), "count": int(n)}
                for c, n in zip(centre[ok], count[ok])
            ]
    return pd.DataFrame(rows)


def reconstruct_log_particles(centres, counts, rng) -> np.ndarray:
    """Log particle values drawn uniformly within each published histogram bin.

    The draw removes the per-disease bin grid (which would otherwise identify the class). A bin that
    straddles zero is truncated to its positive part. An earlier version clipped negative draws to
    1e-3, which created an artificial point mass that occurred only for PSP area (about 19% of those
    particles) and leaked the label; see results/v2/AUDIT_LOG.md.
    """
    order = np.argsort(centres)
    c = np.asarray(centres, dtype=float)[order]
    n = np.asarray(counts, dtype=int)[order]
    width = np.median(np.diff(c))
    lo = np.maximum(c - width / 2, 0.0)
    hi = c + width / 2
    vals = rng.uniform(np.repeat(lo, n), np.repeat(hi, n))
    return np.log(np.maximum(vals, 1e-12))   # uniform(0, hi) returns exactly 0 with probability ~0


def load_proteolysis(workbook: Path = WORKBOOK) -> pd.DataFrame:
    """Fig 3B time course: label, minutes, dose, value (a single curve per disease)."""
    df = _sheet("Fig 3", workbook)
    rows = []
    for label, s in {"AD": 0, "DLB": 4, "PSP": 8}.items():
        for r in range(3, 7):
            minutes = float(df.iat[r, s])
            rows.append({"label": label, "minutes": minutes, "dose": "0.5ug/mL", "value": float(df.iat[r, s + 1])})
            rows.append({"label": label, "minutes": minutes, "dose": "1.0ug/mL", "value": float(df.iat[r, s + 2])})
    return pd.DataFrame(rows)
