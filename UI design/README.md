# NeuroFoldNet Inference Console

> **Outdated demo, not part of the manuscript.** This interface was built for an earlier version of the
> project. Its figures (`public/`) and the Flask backend model come from the earlier pipeline, whose
> accuracy was a leakage artefact (see `legacy/` and `scripts/audit_legacy_pipeline.py`). The interface
> outputs are simulated. For the analyses reported in the manuscript, use the code in the repository root
> and the frozen model in `models/`.

A single-page, client-side interface for running simulated NeuroFoldNet inference and reviewing offline evaluation details.

## How to Run

```bash
npm install
npm run dev
```

Then open http://localhost:3000

## Build

```bash
npm run build
npm run start
```

## Deploy (Vercel)

1. Push this repo to GitHub (or another git provider).
2. In Vercel, click **New Project** and import the repo.
3. Accept the defaults (Framework: Next.js) and deploy.

Vercel will run `npm install` and `npm run build` automatically.

## Notes

- All outputs are simulated for interface validation; no backend is required.
- Replace the scoring section in `components/simulator.ts` with a real inference pipeline when available.

