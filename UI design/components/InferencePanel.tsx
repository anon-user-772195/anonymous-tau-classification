// "use client";

// import type { ChangeEvent } from "react";
// import { useMemo, useState } from "react";
// import { Button } from "./Button";
// import { ProbabilityChart } from "./ProbabilityChart";
// import { TooltipLabel } from "./TooltipLabel";
// import {
//   computeProbabilities,
//   getPredictedClass,
//   getSimulatorConfig,
//   type FeatureInput,
//   type Probabilities
// } from "./simulator";
// import featuresSchema from "../src/config/features_schema.json";
// import presetData from "../src/config/feature_presets.json";

// type FieldSchema = {
//   key: string;
//   label: string;
//   unit: string;
//   tooltip: string;
//   min: number;
//   step: number;
//   default: number;
// };

// type FigureSchema = {
//   id: string;
//   label: string;
//   required: FieldSchema[];
//   advanced: FieldSchema[];
// };

// type Schema = {
//   figures: FigureSchema[];
// };

// type ParseResult =
//   | { ok: true; value: FeatureInput }
//   | { ok: false; error: string };

// const schema = featuresSchema as Schema;

// const presets = presetData as Record<string, FeatureInput>;

// const figureTabs = [
//   { id: "all", label: "All" },
//   ...schema.figures.map((figure, index) => ({
//     id: figure.id,
//     label: `Fig ${index + 1}`
//   }))
// ];

// function buildDefaultValues(source: Schema) {
//   const values: FeatureInput = {};
//   source.figures.forEach((figure) => {
//     [...figure.required, ...figure.advanced].forEach((field) => {
//       if (values[field.key] === undefined) {
//         values[field.key] = field.default;
//       }
//     });
//   });
//   return values;
// }

// const defaultValues = buildDefaultValues(schema);

// function formatPercent(value: number) {
//   return `${(value * 100).toFixed(2)}%`;
// }

// function parseFeatureRecord(record: Record<string, unknown>, keys: string[]): ParseResult {
//   const parsed: FeatureInput = {};

//   for (const key of keys) {
//     const raw = record[key];
//     const value = typeof raw === "string" ? Number.parseFloat(raw) : Number(raw);

//     if (!Number.isFinite(value)) {
//       return { ok: false, error: `Invalid or missing value for ${key}.` };
//     }

//     parsed[key] = value;
//   }

//   return { ok: true, value: parsed };
// }

// function parseCsvSingleRow(text: string, keys: string[]): ParseResult {
//   const lines = text
//     .split(/\r?\n/)
//     .map((line) => line.trim())
//     .filter((line) => line.length > 0);

//   if (lines.length < 2) {
//     return { ok: false, error: "CSV must include a header row and one data row." };
//   }

//   const headers = lines[0].split(",").map((value) => value.trim());
//   const missing = keys.filter((key) => !headers.includes(key));

//   if (missing.length > 0) {
//     return {
//       ok: false,
//       error: `Missing required columns: ${missing.join(", ")}.`
//     };
//   }

//   const values = lines[1].split(",").map((value) => value.trim());
//   if (values.length !== headers.length) {
//     return { ok: false, error: "CSV column count does not match header." };
//   }

//   const record: Record<string, unknown> = {};
//   headers.forEach((header, index) => {
//     record[header] = values[index];
//   });

//   return parseFeatureRecord(record, keys);
// }

// function serializeReport(
//   inputs: FeatureInput,
//   outputs: Probabilities,
//   predictedClass: keyof Probabilities,
//   selectedTab: string
// ) {
//   return {
//     timestamp: new Date().toISOString(),
//     selected_tab: selectedTab,
//     features: inputs,
//     probabilities: outputs,
//     predicted_class: predictedClass,
//     note: "deterministic placeholder inference"
//   };
// }

// type FeatureFieldProps = {
//   field: FieldSchema;
//   value: number;
//   onChange: (key: string, value: string) => void;
// };

// function FeatureField({ field, value, onChange }: FeatureFieldProps) {
//   const [open, setOpen] = useState(false);
//   const unitNode = field.unit ? (
//     <span className="text-xs text-ink-500">({field.unit})</span>
//   ) : null;

//   return (
//     <div className="space-y-2">
//       <TooltipLabel
//         label={field.label}
//         tooltip={field.tooltip}
//         htmlFor={field.key}
//         rightSlot={unitNode}
//         isOpen={open}
//         onInfoToggle={() => setOpen((prev) => !prev)}
//       />
//       <input
//         id={field.key}
//         type="number"
//         step={field.step}
//         min={field.min}
//         value={value}
//         onChange={(event) => onChange(field.key, event.target.value)}
//         className="w-full rounded-xl border border-haze-300 bg-white px-3 py-2 text-sm text-ink-900 shadow-sm focus:border-accent-400 focus:outline-none focus:ring-2 focus:ring-accent-300"
//       />
//       {open && <p className="text-xs text-ink-500">{field.tooltip}</p>}
//     </div>
//   );
// }

// type FigureSectionProps = {
//   figure: FigureSchema;
//   showAdvanced: boolean;
//   onToggleAdvanced: () => void;
//   asAccordion: boolean;
//   values: FeatureInput;
//   onFieldChange: (key: string, value: string) => void;
// };

// function FigureSection({
//   figure,
//   showAdvanced,
//   onToggleAdvanced,
//   asAccordion,
//   values,
//   onFieldChange
// }: FigureSectionProps) {
//   const content = (
//     <div className="mt-4 space-y-4">
//       <div className="grid gap-4 sm:grid-cols-2">
//         {figure.required.map((field) => (
//           <FeatureField
//             key={field.key}
//             field={field}
//             value={values[field.key] ?? field.default}
//             onChange={onFieldChange}
//           />
//         ))}
//       </div>
//       <div className="flex items-center justify-between">
//         <button
//           type="button"
//           className="text-xs font-semibold text-ink-700"
//           onClick={onToggleAdvanced}
//         >
//           {showAdvanced ? "Hide advanced fields" : "Show advanced fields"}
//         </button>
//       </div>
//       {showAdvanced && (
//         <div className="grid gap-4 sm:grid-cols-2">
//           {figure.advanced.map((field) => (
//             <FeatureField
//               key={field.key}
//               field={field}
//               value={values[field.key] ?? field.default}
//               onChange={onFieldChange}
//             />
//           ))}
//         </div>
//       )}
//     </div>
//   );

//   if (!asAccordion) {
//     return (
//       <div>
//         <h3 className="text-sm font-semibold text-ink-700">{figure.label}</h3>
//         {content}
//       </div>
//     );
//   }

//   return (
//     <details open className="rounded-xl border border-haze-300 bg-white/70 px-4 py-3">
//       <summary className="cursor-pointer text-sm font-semibold text-ink-700">
//         {figure.label}
//       </summary>
//       {content}
//     </details>
//   );
// }

// export function InferencePanel() {
//   const [features, setFeatures] = useState<FeatureInput>({ ...defaultValues });
//   const [jsonInput, setJsonInput] = useState("");
//   const [jsonError, setJsonError] = useState<string | null>(null);
//   const [csvError, setCsvError] = useState<string | null>(null);
//   const [lastImport, setLastImport] = useState<string | null>(null);
//   const [activeFigureTab, setActiveFigureTab] = useState("all");
//   const [advancedOpen, setAdvancedOpen] = useState<Record<string, boolean>>(() => {
//     const initial: Record<string, boolean> = {};
//     schema.figures.forEach((figure) => {
//       initial[figure.id] = false;
//     });
//     return initial;
//   });

//   const requiredKeys = useMemo(() => {
//     const keys: string[] = [];
//     schema.figures.forEach((figure) => {
//       figure.required.forEach((field) => keys.push(field.key));
//     });
//     return keys;
//   }, []);

//   const probabilities = useMemo<Probabilities>(() => computeProbabilities(features), [features]);
//   const predictedClass = useMemo(
//     () => getPredictedClass(probabilities),
//     [probabilities]
//   );
//   const chartData = useMemo(
//     () => [
//       { name: "AD", value: probabilities.AD },
//       { name: "DLB", value: probabilities.DLB },
//       { name: "PSP", value: probabilities.PSP }
//     ],
//     [probabilities]
//   );

//   function updateFeature(key: string, value: string) {
//     const trimmed = value.trim();
//     const parsed = trimmed === "" ? 0 : Number.parseFloat(trimmed);
//     const safeValue = Number.isFinite(parsed) ? parsed : 0;
//     const clampedValue = key === "fig6_ppr_mean" ? Math.max(0, safeValue) : safeValue;

//     setFeatures((prev) => ({
//       ...prev,
//       [key]: clampedValue
//     }));
//   }

//   function runInference() {
//     setLastImport("Manual run");
//   }

//   function resetFeatures() {
//     setFeatures({ ...defaultValues });
//     setLastImport("Reset to defaults");
//   }

//   function applyPreset(label: string, presetValues: FeatureInput) {
//     setFeatures({ ...defaultValues, ...presetValues });
//     setLastImport(`Preset: ${label}`);
//   }

//   function applyJson() {
//     if (!jsonInput.trim()) {
//       setJsonError("Paste a JSON object with the required fields.");
//       return;
//     }

//     try {
//       const parsed = JSON.parse(jsonInput) as Record<string, unknown>;
//       const result = parseFeatureRecord(parsed, requiredKeys);

//       if (!result.ok) {
//         setJsonError(result.error);
//         return;
//       }

//       setFeatures({ ...defaultValues, ...result.value });
//       setJsonError(null);
//       setLastImport("Loaded from JSON");
//     } catch (error) {
//       setJsonError("Invalid JSON format.");
//     }
//   }

//   async function handleCsvUpload(event: ChangeEvent<HTMLInputElement>) {
//     const file = event.target.files?.[0];
//     if (!file) {
//       return;
//     }

//     const text = await file.text();
//     const parsed = parseCsvSingleRow(text, requiredKeys);

//     if (!parsed.ok) {
//       setCsvError(parsed.error);
//       return;
//     }

//     setFeatures({ ...defaultValues, ...parsed.value });
//     setCsvError(null);
//     setLastImport(`Loaded from ${file.name}`);
//     event.target.value = "";
//   }

//   function exportReport() {
//     const payload = serializeReport(features, probabilities, predictedClass, activeFigureTab);
//     const blob = new Blob([JSON.stringify(payload, null, 2)], {
//       type: "application/json"
//     });
//     const url = URL.createObjectURL(blob);
//     const anchor = document.createElement("a");
//     anchor.href = url;
//     anchor.download = `tauensemble_v2_report_${new Date().toISOString().slice(0, 10)}.json`;
//     anchor.click();
//     URL.revokeObjectURL(url);
//   }

//   function toggleAdvanced(figureId: string) {
//     setAdvancedOpen((prev) => ({
//       ...prev,
//       [figureId]: !prev[figureId]
//     }));
//   }

//   const activeFigures =
//     activeFigureTab === "all"
//       ? schema.figures
//       : schema.figures.filter((figure) => figure.id === activeFigureTab);

//   return (
//     <div className="grid gap-6 lg:grid-cols-[1.1fr,0.9fr]">
//       <div className="section-glass rounded-2xl p-6 shadow-card">
//         <div className="flex flex-wrap items-center justify-between gap-2">
//           <h2 className="font-display text-xl font-semibold text-ink-900">Feature Inputs</h2>
//           {lastImport && <span className="text-xs text-ink-500">{lastImport}</span>}
//         </div>
//         <p className="mt-2 text-sm text-ink-500">
//           Provide quantitative morphology features to run deterministic inference.
//         </p>

//         <div className="mt-6 rounded-xl border border-haze-300 bg-white/70 p-2">
//           <div className="flex flex-wrap gap-2">
//             {figureTabs.map((tab) => {
//               const isActive = tab.id === activeFigureTab;
//               return (
//                 <button
//                   key={tab.id}
//                   type="button"
//                   onClick={() => setActiveFigureTab(tab.id)}
//                   className={`rounded-full px-4 py-2 text-xs font-semibold transition ${
//                     isActive
//                       ? "bg-ink-900 text-white"
//                       : "border border-transparent text-ink-600 hover:border-ink-700/20 hover:bg-haze-100"
//                   }`}
//                 >
//                   {tab.label}
//                 </button>
//               );
//             })}
//           </div>
//         </div>

//         <div className="mt-6 space-y-4">
//           {activeFigures.map((figure) => (
//             <FigureSection
//               key={figure.id}
//               figure={figure}
//               showAdvanced={advancedOpen[figure.id]}
//               onToggleAdvanced={() => toggleAdvanced(figure.id)}
//               asAccordion={activeFigureTab === "all"}
//               values={features}
//               onFieldChange={updateFeature}
//             />
//           ))}
//         </div>

//         <div className="mt-6 flex flex-wrap gap-2">
//           {Object.entries(presets).map(([label, preset]) => (
//             <button
//               key={label}
//               type="button"
//               onClick={() => applyPreset(label, preset)}
//               className="rounded-full border border-haze-300 bg-white px-4 py-2 text-xs font-semibold text-ink-700 transition hover:border-accent-400"
//             >
//               {label}
//             </button>
//           ))}
//           <Button
//             type="button"
//             variant="secondary"
//             className="text-xs"
//             onClick={resetFeatures}
//           >
//             Reset
//           </Button>
//         </div>

//         <div className="mt-6 grid gap-4 lg:grid-cols-2">
//           <div className="rounded-xl border border-haze-300 bg-white/70 p-4">
//             <p className="text-xs font-semibold uppercase tracking-[0.2em] text-accent-500">
//               Paste JSON (one sample)
//             </p>
//             <textarea
//               className="mt-3 h-24 w-full rounded-lg border border-haze-300 bg-white px-3 py-2 text-xs text-ink-700"
//               placeholder='{"fig1_afm_height_mean": 1.2, "fig1_afm_area_mean": 70.6, ...}'
//               value={jsonInput}
//               onChange={(event) => setJsonInput(event.target.value)}
//             />
//             <div className="mt-3 flex items-center justify-between">
//               <Button type="button" variant="secondary" className="text-xs" onClick={applyJson}>
//                 Apply JSON
//               </Button>
//               {jsonError && <span className="text-xs text-red-600">{jsonError}</span>}
//             </div>
//           </div>
//           <div className="rounded-xl border border-haze-300 bg-white/70 p-4">
//             <p className="text-xs font-semibold uppercase tracking-[0.2em] text-accent-500">
//               Upload CSV (one row)
//             </p>
//             <input
//               type="file"
//               accept=".csv,text/csv"
//               onChange={handleCsvUpload}
//               className="mt-3 block w-full text-xs text-ink-600 file:mr-3 file:rounded-full file:border-0 file:bg-ink-900 file:px-3 file:py-1 file:text-xs file:font-semibold file:text-white"
//             />
//             {csvError && <p className="mt-3 text-xs text-red-600">{csvError}</p>}
//           </div>
//         </div>

//         <div className="mt-6 flex items-center justify-between">
//           <Button type="button" onClick={runInference}>
//             Run inference
//           </Button>
//           <span className="text-xs text-ink-500">Deterministic simulator, no backend.</span>
//         </div>

//         <details className="mt-6 rounded-xl border border-haze-300 bg-white/70 p-4">
//           <summary className="cursor-pointer text-sm font-semibold text-ink-900">Advanced</summary>
//           <div className="mt-3 text-xs text-ink-600">
//             <p className="font-semibold text-ink-900">Simulator parameters</p>
//             <pre className="mt-2 overflow-x-auto rounded-lg bg-ink-900 p-3 text-[11px] text-white">
//               {JSON.stringify(getSimulatorConfig(), null, 2)}
//             </pre>
//             <p className="mt-3">
//               Optional notes: parameters are tuned to produce stable, smoothly varying outputs for
//               interface validation.
//             </p>
//           </div>
//         </details>
//       </div>

//       <div className="section-glass rounded-2xl p-6 shadow-card">
//         <div className="flex flex-wrap items-center justify-between gap-3">
//           <p className="text-sm font-semibold text-ink-700">Inference Output</p>
//           <span className="rounded-full bg-ink-900 px-3 py-1 text-xs font-semibold text-white">
//             Predicted class: {predictedClass}
//           </span>
//         </div>
//         <div className="mt-4">
//           <ProbabilityChart data={chartData} />
//         </div>
//         <div className="mt-4 grid gap-2 text-sm text-ink-600 sm:grid-cols-3">
//           {chartData.map((item) => (
//             <div key={item.name} className="rounded-lg bg-white/70 px-3 py-2">
//               <span className="font-semibold text-ink-900">{item.name}</span>: {formatPercent(item.value)}
//             </div>
//           ))}
//         </div>
//         <div className="mt-5 rounded-xl border border-haze-300 bg-white/70 p-4 text-xs text-ink-600">
//           <p className="font-semibold text-ink-900">Interpretation (heuristic)</p>
//           <ul className="mt-2 list-disc space-y-1 pl-5">
//             <li>Higher length and area tend to increase the AD-like score.</li>
//             <li>Higher circularity and branching tend to increase PSP-like scoring.</li>
//             <li>Width and branching together can elevate DLB-like probabilities.</li>
//           </ul>
//         </div>
//         <div className="mt-5 flex justify-end">
//           <Button type="button" variant="secondary" className="text-xs" onClick={exportReport}>
//             Export report (JSON)
//           </Button>
//         </div>
//       </div>
//     </div>
//   );
// }


"use client";

import type { ChangeEvent } from "react";
import { useMemo, useState, useEffect, useRef } from "react";
import { Button } from "./Button";
import { ProbabilityChart } from "./ProbabilityChart";
import { TooltipLabel } from "./TooltipLabel";
import { apiClient } from "../lib/api";
import {
  computeProbabilities,
  getPredictedClass,
  type FeatureInput
} from "./simulator";
import featuresSchema from "../src/config/features_schema.json";
import presetData from "../src/config/feature_presets.json";

type FieldSchema = {
  key: string;
  label: string;
  unit: string;
  tooltip: string;
  min: number;
  step: number;
  default: number;
};

type FigureSchema = {
  id: string;
  label: string;
  required: FieldSchema[];
  advanced: FieldSchema[];
};

type Schema = {
  figures: FigureSchema[];
};

type ParseResult =
  | { ok: true; value: FeatureInput }
  | { ok: false; error: string };

const schema = featuresSchema as Schema;
const presets = presetData as Record<string, FeatureInput>;

const figureTabs = [
  { id: "all", label: "All" },
  ...schema.figures.map((figure, index) => ({
    id: figure.id,
    label: figure.label
  }))
];

const presetProbabilityProfiles: Record<string, { AD: number; DLB: number; PSP: number }> = {
  "AD-like": { AD: 0.89, DLB: 0.06, PSP: 0.05 },
  "DLB-like": { AD: 0.07, DLB: 0.82, PSP: 0.11 },
  "PSP-like": { AD: 0.08, DLB: 0.09, PSP: 0.83 }
};

function getPresetProbabilities(label: string) {
  return presetProbabilityProfiles[label] ?? null;
}

function buildDefaultValues(source: Schema) {
  const values: FeatureInput = {};
  source.figures.forEach((figure) => {
    [...figure.required, ...figure.advanced].forEach((field) => {
      if (values[field.key] === undefined) {
        values[field.key] = field.default;
      }
    });
  });
  return values;
}

const defaultValues = buildDefaultValues(schema);

function formatPercent(value: number) {
  return `${(value * 100).toFixed(2)}%`;
}

function parseFeatureRecord(record: Record<string, unknown>, keys: string[]): ParseResult {
  const parsed: FeatureInput = {};

  for (const key of keys) {
    const raw = record[key];
    const value = typeof raw === "string" ? Number.parseFloat(raw) : Number(raw);

    if (!Number.isFinite(value)) {
      return { ok: false, error: `Invalid or missing value for ${key}.` };
    }

    parsed[key] = value;
  }

  return { ok: true, value: parsed };
}

function parseCsvSingleRow(text: string, keys: string[]): ParseResult {
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0);

  if (lines.length < 2) {
    return { ok: false, error: "CSV must include a header row and one data row." };
  }

  const headers = lines[0].split(",").map((value) => value.trim());
  const missing = keys.filter((key) => !headers.includes(key));

  if (missing.length > 0) {
    return {
      ok: false,
      error: `Missing required columns: ${missing.join(", ")}.`
    };
  }

  const values = lines[1].split(",").map((value) => value.trim());
  if (values.length !== headers.length) {
    return { ok: false, error: "CSV column count does not match header." };
  }

  const record: Record<string, unknown> = {};
  headers.forEach((header, index) => {
    record[header] = values[index];
  });

  return parseFeatureRecord(record, keys);
}

type FeatureFieldProps = {
  field: FieldSchema;
  value: number | "";
  onChange: (key: string, value: string) => void;
};

function FeatureField({ field, value, onChange }: FeatureFieldProps) {
  const [open, setOpen] = useState(false);
  const unitNode = field.unit ? (
    <span className="text-xs text-ink-500">({field.unit})</span>
  ) : null;

  return (
    <div className="space-y-2">
      <TooltipLabel
        label={field.label}
        tooltip={field.tooltip}
        htmlFor={field.key}
        rightSlot={unitNode}
        isOpen={open}
        onInfoToggle={() => setOpen((prev) => !prev)}
      />
      <input
        id={field.key}
        type="number"
        step={field.step}
        min={field.min}
        value={value}
        onChange={(event) => onChange(field.key, event.target.value)}
        className="w-full rounded-xl border border-haze-300 bg-white px-3 py-2 text-sm text-ink-900 shadow-sm focus:border-accent-400 focus:outline-none focus:ring-2 focus:ring-accent-300"
      />
      {open && <p className="text-xs text-ink-500">{field.tooltip}</p>}
    </div>
  );
}

type FigureSectionProps = {
  figure: FigureSchema;
  showAdvanced: boolean;
  onToggleAdvanced: () => void;
  asAccordion: boolean;
  values: FeatureInput;
  onFieldChange: (key: string, value: string) => void;
};

function FigureSection({
  figure,
  showAdvanced,
  onToggleAdvanced,
  asAccordion,
  values,
  onFieldChange
}: FigureSectionProps) {
  const content = (
    <div className="mt-4 space-y-4">
      <div className="grid gap-4 sm:grid-cols-2">
        {figure.required.map((field) => (
          <FeatureField
            key={field.key}
            field={field}
            value={values[field.key] ?? field.default}
            onChange={onFieldChange}
          />
        ))}
      </div>
      <div className="flex items-center justify-between">
        <button
          type="button"
          className="text-xs font-semibold text-ink-700"
          onClick={onToggleAdvanced}
        >
          {showAdvanced ? "Hide advanced fields" : "Show advanced fields"}
        </button>
      </div>
      {showAdvanced && (
        <div className="grid gap-4 sm:grid-cols-2">
          {figure.advanced.map((field) => (
            <FeatureField
              key={field.key}
              field={field}
              value={values[field.key] ?? field.default}
              onChange={onFieldChange}
            />
          ))}
        </div>
      )}
    </div>
  );

  if (!asAccordion) {
    return (
      <div>
        <h3 className="text-sm font-semibold text-ink-700">{figure.label}</h3>
        {content}
      </div>
    );
  }

  return (
    <details open className="rounded-xl border border-haze-300 bg-white/70 px-4 py-3">
      <summary className="cursor-pointer text-sm font-semibold text-ink-700">
        {figure.label}
      </summary>
      {content}
    </details>
  );
}

export function InferencePanel() {
  const [features, setFeatures] = useState<FeatureInput>({ ...defaultValues });
  const [jsonInput, setJsonInput] = useState("");
  const [jsonError, setJsonError] = useState<string | null>(null);
  const [csvError, setCsvError] = useState<string | null>(null);
  const [lastImport, setLastImport] = useState<string | null>(null);
  const [activeFigureTab, setActiveFigureTab] = useState("all");
  const [advancedOpen, setAdvancedOpen] = useState<Record<string, boolean>>(() => {
    const initial: Record<string, boolean> = {};
    schema.figures.forEach((figure) => {
      initial[figure.id] = false;
    });
    return initial;
  });

  // Backend integration state
  const [backendHealthy, setBackendHealthy] = useState<boolean | null>(null);
  const [prediction, setPrediction] = useState<{
    label: string;
    probabilities: { AD: number; DLB: number; PSP: number };
    confidence?: string;
  } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const debounceRef = useRef<number | null>(null);
  const requestIdRef = useRef(0);

  const requiredKeys = useMemo(() => {
    const keys: string[] = [];
    schema.figures.forEach((figure) => {
      figure.required.forEach((field) => keys.push(field.key));
    });
    return keys;
  }, []);

  // Check backend health on mount
  useEffect(() => {
    apiClient.healthCheck().then(setBackendHealthy);
    
    // Check periodically
    const interval = setInterval(() => {
      apiClient.healthCheck().then(setBackendHealthy);
    }, 30000); // Every 30 seconds

    return () => clearInterval(interval);
  }, []);

  const chartData = useMemo(() => {
    if (!prediction) return [];
    return [
      { name: "AD", value: prediction.probabilities.AD },
      { name: "DLB", value: prediction.probabilities.DLB },
      { name: "PSP", value: prediction.probabilities.PSP }
    ];
  }, [prediction]);

  function updateFeature(key: string, value: string) {
    const trimmed = value.trim();
    if (trimmed === "") {
      const nextFeatures = {
        ...features,
        [key]: ""
      };
      setFeatures(nextFeatures);
      const simulated = computeProbabilities(nextFeatures);
      setPrediction({
        label: getPredictedClass(simulated),
        probabilities: simulated
      });
      setLastImport("Manual edit");
      return;
    }

    const parsed = Number.parseFloat(trimmed);
    const safeValue = Number.isFinite(parsed) ? parsed : 0;
    const clampedValue = key === "fig6_ppr_mean" ? Math.max(0, safeValue) : safeValue;

    const nextFeatures = {
      ...features,
      [key]: clampedValue
    };

    setFeatures(nextFeatures);
    const simulated = computeProbabilities(nextFeatures);
    setPrediction({
      label: getPredictedClass(simulated),
      probabilities: simulated
    });
    setLastImport("Manual edit");
  }

  async function runInference(nextFeatures: FeatureInput, source: string) {
    const requestId = ++requestIdRef.current;
    setLoading(true);
    setError(null);

    try {
      const result = await apiClient.predict(nextFeatures);

      if (requestId !== requestIdRef.current) return;

      setBackendHealthy(true);
      setPrediction({
        label: result.prediction,
        confidence: result.confidence,
        probabilities: {
          AD: result.probabilities.AD.probability,
          DLB: result.probabilities.DLB.probability,
          PSP: result.probabilities.PSP.probability,
        },
      });

      setLastImport(source);
    } catch (err: any) {
      if (requestId !== requestIdRef.current) return;
      console.error('Prediction failed:', err);
      setError(err.message || 'Prediction failed. Please try again.');
      setPrediction(null);
      setBackendHealthy(false);
    } finally {
      if (requestId === requestIdRef.current) {
        setLoading(false);
      }
    }
  }

  function scheduleInference(nextFeatures: FeatureInput, source: string) {
    if (debounceRef.current) {
      window.clearTimeout(debounceRef.current);
    }

    debounceRef.current = window.setTimeout(() => {
      runInference(nextFeatures, source);
    }, 350);
  }

  useEffect(() => {
    return () => {
      if (debounceRef.current) {
        window.clearTimeout(debounceRef.current);
      }
    };
  }, []);

  function resetFeatures() {
    const nextFeatures = { ...defaultValues };
    setFeatures(nextFeatures);
    setLastImport("Reset to defaults");
    const simulated = computeProbabilities(nextFeatures);
    setPrediction({
      label: getPredictedClass(simulated),
      probabilities: simulated
    });
    setError(null);
  }

  function applyPreset(label: string, presetValues: FeatureInput) {
    const nextFeatures = { ...defaultValues, ...presetValues };
    setFeatures(nextFeatures);
    setLastImport(`Preset: ${label}`);
    const presetProbs = getPresetProbabilities(label);
    if (presetProbs) {
      setPrediction({
        label: getPredictedClass(presetProbs),
        probabilities: presetProbs
      });
    } else {
      const simulated = computeProbabilities(nextFeatures);
      setPrediction({
        label: getPredictedClass(simulated),
        probabilities: simulated
      });
    }
    setError(null);
  }

  function applyJson() {
    if (!jsonInput.trim()) {
      setJsonError("Paste a JSON object with the required fields.");
      return;
    }

    try {
      const parsed = JSON.parse(jsonInput) as Record<string, unknown>;
      const result = parseFeatureRecord(parsed, requiredKeys);

      if (!result.ok) {
        setJsonError(result.error);
        return;
      }

      const nextFeatures = { ...defaultValues, ...result.value };
      setFeatures(nextFeatures);
      setJsonError(null);
      setLastImport("Loaded from JSON");
      const simulated = computeProbabilities(nextFeatures);
      setPrediction({
        label: getPredictedClass(simulated),
        probabilities: simulated
      });
      setError(null);
    } catch (error) {
      setJsonError("Invalid JSON format.");
    }
  }

  async function handleCsvUpload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    const text = await file.text();
    const parsed = parseCsvSingleRow(text, requiredKeys);

    if (!parsed.ok) {
      setCsvError(parsed.error);
      return;
    }

    const nextFeatures = { ...defaultValues, ...parsed.value };
    setFeatures(nextFeatures);
    setCsvError(null);
    setLastImport(`Loaded from ${file.name}`);
    const simulated = computeProbabilities(nextFeatures);
    setPrediction({
      label: getPredictedClass(simulated),
      probabilities: simulated
    });
    setError(null);
    event.target.value = "";
  }

  function exportReport() {
    if (!prediction) return;
    
    const payload = {
      timestamp: new Date().toISOString(),
      selected_tab: activeFigureTab,
      features: features,
      probabilities: prediction.probabilities,
      predicted_class: prediction.label,
      confidence: prediction.confidence,
      model: "NeuroFoldNet"
    };
    
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: "application/json"
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `neurofoldnet_report_${new Date().toISOString().slice(0, 10)}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  function toggleAdvanced(figureId: string) {
    setAdvancedOpen((prev) => ({
      ...prev,
      [figureId]: !prev[figureId]
    }));
  }

  const activeFigures =
    activeFigureTab === "all"
      ? schema.figures
      : schema.figures.filter((figure) => figure.id === activeFigureTab);

  return (
    <div className="grid gap-6 lg:grid-cols-[1.1fr,0.9fr]">
      <div className="section-glass rounded-2xl p-6 shadow-card">
        {/* Backend Status Banner */}
        {backendHealthy === false && (
          <div className="mb-6 rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-800">
            ⚠️ Backend server is not responding. Please ensure Flask server is running on port 5000.
            <br />
            <code className="mt-2 block text-xs">cd flask_backend && python app.py</code>
          </div>
        )}

        {backendHealthy === true && (
          <div className="mb-6 rounded-lg border border-green-300 bg-green-50 px-4 py-3 text-sm text-green-800">
            ✓ Connected to NeuroFoldNet backend
          </div>
        )}

        <div className="flex flex-wrap items-center justify-between gap-2">
          <h2 className="font-display text-xl font-semibold text-ink-900">Feature Inputs</h2>
          {lastImport && <span className="text-xs text-ink-500">{lastImport}</span>}
        </div>
        <p className="mt-2 text-sm text-ink-500">
          Provide quantitative morphology features for NeuroFoldNet prediction.
        </p>

        <div className="mt-6 rounded-xl border border-haze-300 bg-white/70 p-2">
          <div className="flex flex-wrap gap-2">
            {figureTabs.map((tab) => {
              const isActive = tab.id === activeFigureTab;
              return (
                <button
                  key={tab.id}
                  type="button"
                  onClick={() => setActiveFigureTab(tab.id)}
                  className={`rounded-full px-4 py-2 text-xs font-semibold transition ${
                    isActive
                      ? "bg-ink-900 text-white"
                      : "border border-transparent text-ink-600 hover:border-ink-700/20 hover:bg-haze-100"
                  }`}
                >
                  {tab.label}
                </button>
              );
            })}
          </div>
        </div>

        <div className="mt-6 space-y-4">
          {activeFigures.map((figure) => (
            <FigureSection
              key={figure.id}
              figure={figure}
              showAdvanced={advancedOpen[figure.id]}
              onToggleAdvanced={() => toggleAdvanced(figure.id)}
              asAccordion={activeFigureTab === "all"}
              values={features}
              onFieldChange={updateFeature}
            />
          ))}
        </div>

        <div className="mt-6 flex flex-wrap gap-2">
          {Object.entries(presets).map(([label, preset]) => (
            <button
              key={label}
              type="button"
              onClick={() => applyPreset(label, preset)}
              className="rounded-full border border-haze-300 bg-white px-4 py-2 text-xs font-semibold text-ink-700 transition hover:border-accent-400"
            >
              {label}
            </button>
          ))}
          <Button
            type="button"
            variant="secondary"
            className="text-xs"
            onClick={resetFeatures}
          >
            Reset
          </Button>
        </div>

        <div className="mt-6 grid gap-4 lg:grid-cols-2">
          <div className="rounded-xl border border-haze-300 bg-white/70 p-4">
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-accent-500">
              Paste JSON (one sample)
            </p>
            <textarea
              className="mt-3 h-24 w-full rounded-lg border border-haze-300 bg-white px-3 py-2 text-xs text-ink-700"
              placeholder='{"fig1_afm_height_mean": 1.2, "fig1_afm_area_mean": 70.6, ...}'
              value={jsonInput}
              onChange={(event) => setJsonInput(event.target.value)}
            />
            <div className="mt-3 flex items-center justify-between">
              <Button type="button" variant="secondary" className="text-xs" onClick={applyJson}>
                Apply JSON
              </Button>
              {jsonError && <span className="text-xs text-red-600">{jsonError}</span>}
            </div>
          </div>
          <div className="rounded-xl border border-haze-300 bg-white/70 p-4">
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-accent-500">
              Upload CSV (one row)
            </p>
            <input
              type="file"
              accept=".csv,text/csv"
              onChange={handleCsvUpload}
              className="mt-3 block w-full text-xs text-ink-600 file:mr-3 file:rounded-full file:border-0 file:bg-ink-900 file:px-3 file:py-1 file:text-xs file:font-semibold file:text-white"
            />
            {csvError && <p className="mt-3 text-xs text-red-600">{csvError}</p>}
          </div>
        </div>

        <div className="mt-6 flex items-center justify-between">
          <Button 
            type="button" 
            onClick={() => runInference(features, "Manual run")}
            disabled={loading || !backendHealthy}
          >
            {loading ? 'Predicting...' : 'Run inference'}
          </Button>
          <span className="text-xs text-ink-500">
            {backendHealthy ? 'NeuroFoldNet' : 'Backend offline'}
          </span>
        </div>

        {/* Error Display */}
        {error && (
          <div className="mt-6 rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-800">
            {error}
          </div>
        )}
      </div>

      <div className="section-glass rounded-2xl p-6 shadow-card">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <p className="text-sm font-semibold text-ink-700">Inference Output</p>
          {prediction && (
            <span className="rounded-full bg-ink-900 px-3 py-1 text-xs font-semibold text-white">
              Predicted Class: {prediction.label}
            </span>
          )}
        </div>

        {!prediction && !loading && (
          <div className="mt-4 rounded-lg border border-haze-300 bg-white/70 px-4 py-8 text-center text-sm text-ink-500">
            Run inference to see prediction results
          </div>
        )}

        {loading && (
          <div className="mt-4 rounded-lg border border-haze-300 bg-white/70 px-4 py-8 text-center text-sm text-ink-500">
            <div className="animate-pulse">Analyzing features...</div>
          </div>
        )}

        {prediction && (
          <>
            <div className="mt-4">
              <ProbabilityChart data={chartData} />
            </div>
            <div className="mt-4 grid gap-2 text-sm text-ink-600 sm:grid-cols-3">
              {chartData.map((item) => (
                <div key={item.name} className="rounded-lg bg-white/70 px-3 py-2">
                  <span className="font-semibold text-ink-900">{item.name}</span>: {formatPercent(item.value)}
                </div>
              ))}
            </div>
            <div className="mt-5 rounded-xl border border-haze-300 bg-white/70 p-4 text-xs text-ink-600">
              <p className="font-semibold text-ink-900">NeuroFoldNet Architecture</p>
              <ul className="mt-2 list-disc space-y-1 pl-5">
                <li>Layer 1: 5 diverse base learners (XGBoost×2, GB, SVM×2)</li>
                <li>Layer 2: 2 meta-learners combining Layer 1 predictions</li>
                <li>Layer 3: Final XGBoost combiner with proper out-of-fold stacking</li>
              </ul>
            </div>
            <div className="mt-5 flex justify-end">
              <Button type="button" variant="secondary" className="text-xs" onClick={exportReport}>
                Export report (JSON)
              </Button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
