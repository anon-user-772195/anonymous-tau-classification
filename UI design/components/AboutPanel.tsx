"use client";

import { Badge } from "./Badge";

const performanceRows = [
  { model: "NeuroFoldNet", accuracy: 0.7901, f1: 0.7833, logLoss: 0.5109, highlight: true },
  { model: "XGBoost", accuracy: 0.6780, f1: 0.6734, logLoss: 1.0913 },
  { model: "Gradient Boosting", accuracy: 0.6348, f1: 0.6324, logLoss: 1.0833 },
  { model: "SVM", accuracy: 0.5905, f1: 0.5857, logLoss: 0.9322 },
];

export function AboutPanel() {
  return (
    <div className="space-y-6">
      <div className="section-glass rounded-2xl p-6 shadow-card">
        <div className="flex items-start justify-between">
          <div>
            <h2 className="font-display text-2xl font-semibold text-ink-900">
              NeuroFoldNet
            </h2>
            <p className="mt-2 text-sm text-ink-600">
              Triple-layer stacking ensemble for tau protein classification (AD vs DLB vs PSP).
            </p>
            <p className="mt-1 text-xs text-ink-500">
              Offline evaluation with K-Fold Cross-Validation.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <Badge>Leakage-Free</Badge>
            <Badge>5-Fold CV</Badge>
            <Badge>8 Models</Badge>
          </div>
        </div>

        <div className="mt-6 rounded-lg border border-haze-300 bg-white/80 p-4 text-sm text-ink-600">
          <p className="font-semibold text-ink-900">Model Summary</p>
          <ul className="mt-2 list-disc space-y-1 pl-5">
            <li>Layer 1: 5 base learners (XGBoost x2, GB, SVM x2)</li>
            <li>Layer 2: 2 meta-learners (XGBoost, Gradient Boosting)</li>
            <li>Layer 3: Final XGBoost combiner</li>
          </ul>
        </div>
      </div>

      <div className="section-glass overflow-hidden rounded-2xl shadow-card">
        <div className="bg-ink-900 px-6 py-4">
          <h3 className="text-lg font-semibold text-white">Cross-Validation Performance</h3>
          <p className="mt-1 text-sm text-haze-300">
            K-Fold Cross-Validation
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="bg-haze-100 text-ink-700">
              <tr>
                <th className="px-6 py-3 font-semibold">Model</th>
                <th className="px-6 py-3 font-semibold">Accuracy</th>
                <th className="px-6 py-3 font-semibold">Macro-F1</th>
                <th className="px-6 py-3 font-semibold">Log Loss</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-haze-200 bg-white">
              {performanceRows.map((row) => (
                <tr key={row.model} className={row.highlight ? "bg-accent-50" : ""}>
                  <td className="px-6 py-4 font-semibold text-ink-900">
                    {row.model}
                    {row.highlight && <Badge className="ml-2 text-xs">Best</Badge>}
                  </td>
                  <td className="px-6 py-4">{row.accuracy.toFixed(4)}</td>
                  <td className="px-6 py-4">{row.f1.toFixed(4)}</td>
                  <td className="px-6 py-4">{row.logLoss.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="section-glass rounded-2xl p-6 shadow-card">
        <h3 className="text-sm font-semibold text-ink-700">Methodology</h3>
        <p className="mt-2 text-sm text-ink-600">
          Replicate-aware validation ensures polymorphs from the same donor or imaging batch never
          cross folds, preventing leakage that would inflate performance.
        </p>
        <div className="mt-4 space-y-2 text-sm text-ink-600">
          <p><strong>Classes:</strong> AD, DLB, PSP</p>
          <p><strong>Features:</strong> 8 core features used for prediction</p>
          <p><strong>Validation:</strong> K-Fold Cross-Validation</p>
        </div>
      </div>

      <div className="section-glass rounded-2xl p-6 shadow-card">
        <h3 className="text-sm font-semibold text-ink-700">Why Proper Stacking Matters</h3>
        <p className="mt-2 text-sm text-ink-600">
          NeuroFoldNet uses <strong>out-of-fold predictions</strong> to prevent data leakage.
          Each layer is trained on predictions from models that never saw the validation data during training.
        </p>
        <div className="mt-4 rounded-lg border border-haze-300 bg-white/70 p-4">
          <p className="mb-2 text-xs font-semibold text-ink-900">Key Features:</p>
          <ul className="list-disc space-y-1 pl-5 text-xs text-ink-600">
            <li>No data leakage between training and validation</li>
            <li>Proper out-of-fold stacking across all layers</li>
            <li>Replicate-aware validation splits</li>
            <li>Production-ready performance estimates</li>
          </ul>
        </div>
      </div>

    </div>
  );
}
