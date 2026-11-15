"""Reproduce every result file, figure and manuscript number in dependency order.

Usage:
  python scripts/run_all.py            # full run (about 2-3 h on a laptop CPU, dominated by permutation tests)
  python scripts/run_all.py --quick    # smoke test of the spectral/seeding models (minutes), then the rest
  python scripts/run_all.py --from run_fusion_analyses   # resume from a step

Each step is logged to results/v2/logs/<step>.log with its exact command line, start/end time and exit code.
All random seeds are fixed inside the scripts; see REPRODUCIBILITY.md.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "results" / "v2" / "logs"

STEPS = [
    ("build_fig7_sparse", []),
    ("run_assay_models", ["QUICK"]),           # spectral + seeding experts, permutations, OOF posteriors
    ("run_fig7_cross_genotype", []),
    ("run_morphology_models", []),             # AFM experts on reconstructed particles
    ("run_multiassay_panel", []),              # product-rule panels (needs the two expert steps)
    ("run_fusion_analyses", []),               # fusion rules, ablation, calibration
    ("run_decision_support", []),              # conformal sets, novelty detection, sequential acquisition
    ("run_spectral_preprocessing_sensitivity", []),
    ("run_assay_statistics", []),
    # audit checks, with the exact settings used for results/v2/AUDIT_LOG.md
    ("run_robustness_checks", ["--only", "seeding_nested", "seeding_fusion_sensitivity", "--n-perm", "1000"]),
    ("run_robustness_checks", ["--only", "drift", "--repeats", "5", "--n-null", "200"]),
    ("run_robustness_checks", ["--only", "block_size", "--repeats", "3"]),
    ("export_robustness_numbers", []),         # LaTeX macros for the robustness section (rev1)
    ("audit_legacy_pipeline", []),
    ("neurofoldnet_external", ["freeze"]),     # frozen model + input templates in models/
    ("export_paper_numbers", []),              # LaTeX macros used by the manuscript
    ("export_supplementary_tables", []),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--from", dest="start", default=None)
    args = ap.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    names = [s for s, _ in STEPS]
    first = names.index(args.start) if args.start else 0
    for step_no, (name, extra) in enumerate(STEPS[first:], start=first + 1):
        extra = [a for a in extra if a != "QUICK"] + (["--quick"] if "QUICK" in extra and args.quick else [])
        cmd = [sys.executable, "-X", "utf8", str(ROOT / "scripts" / f"{name}.py"), *extra]
        t0 = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] {name} ...", flush=True)
        logfile = LOGS / f"{step_no:02d}_{name}.log"
        with open(logfile, "w", encoding="utf-8") as log:
            log.write(f"$ {' '.join(cmd)}\nstart {time.ctime()}\n\n")
            log.flush()
            rc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
            log.write(f"\nend {time.ctime()}  exit {rc}  ({time.time() - t0:.0f}s)\n")
        if rc != 0:
            sys.exit(f"{name} failed (exit {rc}); see {logfile}")
        print(f"    done in {time.time() - t0:.0f}s", flush=True)
    print("All steps finished. Run `python -m pytest -q tests` to check the leakage invariants.")


if __name__ == "__main__":
    main()
