from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_TAU_XLSX = Path("data/raw/42003_2025_7499_MOESM3_ESM.xlsx")


def find_tau_workbook(path: str | Path | None = None) -> Path:
    """Find the Nature supplement workbook used by the original notebook."""
    candidates: list[Path] = []
    if path:
        candidates.append(Path(path))
    candidates.extend(
        [
            DEFAULT_TAU_XLSX,
            Path.home() / "Downloads" / "42003_2025_7499_MOESM3_ESM.xlsx",
            Path.home() / "Downloads" / "42003_2025_7499_MOESM3_ESM (1).xlsx",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n".join(f"  - {c}" for c in candidates)
    raise FileNotFoundError(
        "Could not find the tau polymorph Excel supplement. Searched:\n"
        f"{searched}\n"
        "Place 42003_2025_7499_MOESM3_ESM.xlsx in data/raw/ or pass --tau-input."
    )


def load_tau_feature_table(path: str | Path | None = None) -> pd.DataFrame:
    """
    Reconstruct the original tau polymorph feature table from the notebook.

    The workbook parsing mirrors flask_backend/Classification_Complete_Pipeline.ipynb:
    Fig 1 morphology, Fig 3 proteolytic resistance, Fig 4 seeding density,
    and Fig 5 electrophysiology are outer-merged by disease and replicate index.
    """
    workbook = find_tau_workbook(path)
    print(f"Loading tau polymorph workbook: {workbook}")

    df1 = pd.read_excel(workbook, sheet_name="Fig 1", header=None)
    morphology_rows = []
    for disease, h_col, a_col, d_col in [
        ("AD", 0, 2, 29),
        ("DLB", 22, 26, 30),
        ("PSP", 23, 27, 31),
    ]:
        height = pd.to_numeric(df1[h_col].iloc[3:100], errors="coerce").dropna().values
        area = pd.to_numeric(df1[a_col].iloc[3:100], errors="coerce").dropna().values
        diameter = pd.to_numeric(df1[d_col].iloc[2:100], errors="coerce").dropna().values
        for i in range(min(len(height), len(area), len(diameter))):
            morphology_rows.append(
                {
                    "disease": disease,
                    "height": height[i],
                    "area": area[i],
                    "diameter": diameter[i],
                }
            )
    df_morph = pd.DataFrame(morphology_rows)

    df3 = pd.read_excel(workbook, sheet_name="Fig 3", header=None)
    proteo_rows = []
    for disease, c1, c2 in [("AD", 1, 2), ("DLB", 5, 6), ("PSP", 9, 10)]:
        p1 = pd.to_numeric(df3[c1].iloc[3:7], errors="coerce").dropna().values
        p2 = pd.to_numeric(df3[c2].iloc[3:7], errors="coerce").dropna().values
        for i in range(min(len(p1), len(p2))):
            proteo_rows.append(
                {
                    "disease": disease,
                    "proteo_resist_05": p1[i],
                    "proteo_resist_1": p2[i],
                }
            )
    df_proteo = pd.DataFrame(proteo_rows)

    df4 = pd.read_excel(workbook, sheet_name="Fig 4", header=None)
    seed_rows = []
    for disease, col in [("AD", 1), ("DLB", 5), ("PSP", 9)]:
        seeds = pd.to_numeric(df4[col].iloc[3:7], errors="coerce").dropna().values
        for value in seeds:
            seed_rows.append({"disease": disease, "seed_density": value})
    df_seed = pd.DataFrame(seed_rows)

    df5 = pd.read_excel(workbook, sheet_name="Fig 5", header=None)
    electro_rows = []
    for disease, c1, c2 in [("AD", 0, 1), ("DLB", 4, 5), ("PSP", 8, 9)]:
        basal = pd.to_numeric(df5[c1].iloc[3:13], errors="coerce").dropna().values
        induced = pd.to_numeric(df5[c2].iloc[3:13], errors="coerce").dropna().values
        for i in range(min(len(basal), len(induced))):
            electro_rows.append(
                {
                    "disease": disease,
                    "basal_trans": basal[i],
                    "induced_trans": induced[i],
                }
            )
    df_electro = pd.DataFrame(electro_rows)

    for frame in [df_morph, df_proteo, df_seed, df_electro]:
        frame["replicate"] = frame.groupby("disease").cumcount()

    data = df_morph.merge(df_proteo, on=["disease", "replicate"], how="outer")
    data = data.merge(df_seed, on=["disease", "replicate"], how="outer")
    data = data.merge(df_electro, on=["disease", "replicate"], how="outer")
    data = data.dropna(thresh=6).copy()

    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if data[col].isnull().any():
            data[col] = data[col].fillna(data[col].median())

    data["sample_id"] = data["disease"].astype(str) + "_tau_rep" + data["replicate"].astype(int).astype(str)
    data["group"] = data["sample_id"]
    ordered = ["sample_id", "group", "disease"] + [
        c for c in data.columns if c not in {"sample_id", "group", "disease"}
    ]
    data = data[ordered]
    print(f"Tau feature table shape: {data.shape}")
    print("Tau disease distribution:")
    print(data["disease"].value_counts().to_string())
    return data


def save_tau_feature_table(output: str | Path = "data/processed/tau_features.csv") -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = load_tau_feature_table()
    data.to_csv(output_path, index=False)
    print(f"Saved tau features: {output_path}")
    return output_path

