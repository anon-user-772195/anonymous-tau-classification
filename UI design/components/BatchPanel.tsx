// "use client";

// import type { ChangeEvent } from "react";
// import { useState } from "react";
// import { Button } from "./Button";
// import {
//   computeProbabilities,
//   getPredictedClass,
//   type FeatureInput,
//   type Probabilities
// } from "./simulator";
// import featuresSchema from "../src/config/features_schema.json";

// type FieldSchema = {
//   key: string;
//   default: number;
// };

// type FigureSchema = {
//   required: FieldSchema[];
//   advanced: FieldSchema[];
// };

// type Schema = {
//   figures: FigureSchema[];
// };

// type BatchRow = {
//   raw: Record<string, string>;
//   features: FeatureInput;
// };

// type BatchResult = {
//   raw: Record<string, string>;
//   probabilities: Probabilities;
//   predicted_class: keyof Probabilities;
//   missing_fields_count: number;
// };

// type ParseResult =
//   | {
//       ok: true;
//       rows: BatchRow[];
//       headers: string[];
//       recognizedKeys: string[];
//       missingKeys: string[];
//       unknownKeys: string[];
//     }
//   | { ok: false; error: string };

// const schema = featuresSchema as Schema;
// const schemaKeys = Array.from(
//   new Set(
//     schema.figures.flatMap((figure) => [...figure.required, ...figure.advanced]).map((f) => f.key)
//   )
// );
// const schemaDefaults: Record<string, number> = {};
// schema.figures.forEach((figure) => {
//   [...figure.required, ...figure.advanced].forEach((field) => {
//     schemaDefaults[field.key] = field.default;
//   });
// });

// function parseCsv(text: string): ParseResult {
//   const lines = text
//     .split(/\r?\n/)
//     .map((line) => line.trim())
//     .filter((line) => line.length > 0);

//   if (lines.length < 2) {
//     return { ok: false, error: "CSV must include a header and at least one data row." };
//   }

//   const headers = lines[0].split(",").map((value) => value.trim());
//   if (headers.length === 0) {
//     return { ok: false, error: "CSV header row is empty." };
//   }

//   const schemaKeySet = new Set(schemaKeys);
//   const recognizedKeys = headers.filter((header) => schemaKeySet.has(header));
//   const unknownKeys = headers.filter((header) => !schemaKeySet.has(header));
//   const missingKeys = schemaKeys.filter((key) => !headers.includes(key));

//   const rows: BatchRow[] = [];

//   for (const line of lines.slice(1)) {
//     const values = line.split(",").map((value) => value.trim());
//     if (values.length !== headers.length) {
//       return { ok: false, error: "CSV column count does not match header." };
//     }

//     const raw: Record<string, string> = {};
//     headers.forEach((header, index) => {
//       raw[header] = values[index] ?? "";
//     });

//     const features: FeatureInput = {};
//     schemaKeys.forEach((key) => {
//       if (raw[key] !== undefined) {
//         const rawValue = raw[key];
//         if (rawValue === "") {
//           features[key] = schemaDefaults[key] ?? 0;
//           return;
//         }
//         const parsed = Number.parseFloat(rawValue);
//         if (!Number.isFinite(parsed)) {
//           throw new Error(`Invalid numeric value for ${key}.`);
//         }
//         features[key] = parsed;
//       } else {
//         features[key] = schemaDefaults[key] ?? 0;
//       }
//     });

//     rows.push({ raw, features });
//   }

//   return { ok: true, rows, headers, recognizedKeys, missingKeys, unknownKeys };
// }

// function buildResultsCsv(
//   results: BatchResult[],
//   headers: string[],
//   missingCount: number
// ) {
//   const header = [
//     ...headers,
//     "prob_AD",
//     "prob_DLB",
//     "prob_PSP",
//     "predicted_class",
//     "missing_fields_count"
//   ];

//   const lines = results.map((result) => {
//     const rowValues = [
//       ...headers.map((headerName) => result.raw[headerName] ?? ""),
//       result.probabilities.AD.toFixed(4),
//       result.probabilities.DLB.toFixed(4),
//       result.probabilities.PSP.toFixed(4),
//       result.predicted_class,
//       String(missingCount)
//     ];
//     return rowValues.join(",");
//   });

//   return [header.join(","), ...lines].join("\n");
// }

// function downloadTemplateCsv() {
//   const headerRow = schemaKeys.join(",");
//   const exampleRow = schemaKeys
//     .map((key) => (schemaDefaults[key] ?? 0).toString())
//     .join(",");
//   const csvText = [headerRow, exampleRow].join("\n");
//   const blob = new Blob([csvText], { type: "text/csv" });
//   const url = URL.createObjectURL(blob);
//   const anchor = document.createElement("a");
//   anchor.href = url;
//   anchor.download = "tauensemble_v2_template.csv";
//   anchor.click();
//   URL.revokeObjectURL(url);
// }

// export function BatchPanel() {
//   const [rows, setRows] = useState<BatchRow[]>([]);
//   const [results, setResults] = useState<BatchResult[] | null>(null);
//   const [parseError, setParseError] = useState<string | null>(null);
//   const [fileLabel, setFileLabel] = useState<string | null>(null);
//   const [headers, setHeaders] = useState<string[]>([]);
//   const [recognizedCount, setRecognizedCount] = useState(0);
//   const [missingCount, setMissingCount] = useState(0);
//   const [unknownCount, setUnknownCount] = useState(0);

//   async function handleUpload(event: ChangeEvent<HTMLInputElement>) {
//     const file = event.target.files?.[0];
//     if (!file) {
//       return;
//     }

//     try {
//       const text = await file.text();
//       const parsed = parseCsv(text);

//     if (!parsed.ok) {
//       setParseError(parsed.error);
//       setRows([]);
//       setResults(null);
//       setHeaders([]);
//       setRecognizedCount(0);
//       setMissingCount(0);
//       setUnknownCount(0);
//       return;
//     }

//       setParseError(null);
//       setRows(parsed.rows);
//       setResults(null);
//       setHeaders(parsed.headers);
//       setRecognizedCount(parsed.recognizedKeys.length);
//       setMissingCount(parsed.missingKeys.length);
//       setUnknownCount(parsed.unknownKeys.length);
//       setFileLabel(file.name);
//       event.target.value = "";
//     } catch (error) {
//       setParseError(error instanceof Error ? error.message : "Failed to parse CSV.");
//       setRows([]);
//       setResults(null);
//       setHeaders([]);
//       setRecognizedCount(0);
//       setMissingCount(0);
//       setUnknownCount(0);
//     }
//   }

//   function runBatch() {
//     if (rows.length === 0) {
//       setParseError("Upload a CSV with at least one row before running batch inference.");
//       return;
//     }

//     const computed = rows.map((row) => {
//       const probabilities = computeProbabilities(row.features);
//       return {
//         raw: row.raw,
//         probabilities,
//         predicted_class: getPredictedClass(probabilities),
//         missing_fields_count: missingCount
//       };
//     });

//     setResults(computed);
//     setParseError(null);
//   }

//   function downloadResults() {
//     if (!results || results.length === 0) {
//       return;
//     }

//     const csvText = buildResultsCsv(results, headers, missingCount);
//     const blob = new Blob([csvText], { type: "text/csv" });
//     const url = URL.createObjectURL(blob);
//     const anchor = document.createElement("a");
//     anchor.href = url;
//     anchor.download = "results.csv";
//     anchor.click();
//     URL.revokeObjectURL(url);
//   }

//   return (
//     <div className="grid gap-6 lg:grid-cols-[1.1fr,0.9fr]">
//       <div className="section-glass rounded-2xl p-6 shadow-card">
//         <div className="flex flex-wrap items-center justify-between gap-2">
//           <h2 className="font-display text-xl font-semibold text-ink-900">Batch Inference</h2>
//           {fileLabel && <span className="text-xs text-ink-500">{fileLabel}</span>}
//         </div>
//         <p className="mt-2 text-sm text-ink-500">
//           Upload a CSV with any subset of schema fields. Missing fields are filled with defaults.
//         </p>
//         <div className="mt-4 flex flex-wrap gap-3">
//           <input
//             type="file"
//             accept=".csv,text/csv"
//             onChange={handleUpload}
//             className="block w-full text-xs text-ink-600 file:mr-3 file:rounded-full file:border-0 file:bg-ink-900 file:px-3 file:py-1 file:text-xs file:font-semibold file:text-white"
//           />
//           <Button type="button" variant="secondary" className="text-xs" onClick={downloadTemplateCsv}>
//             Download template CSV
//           </Button>
//         </div>
//         {parseError && <p className="mt-3 text-xs text-red-600">{parseError}</p>}

//         <div className="mt-4 rounded-xl border border-haze-300 bg-white/70 p-4 text-xs text-ink-600">
//           <div className="flex flex-wrap gap-4">
//             <span>Recognized schema fields: {recognizedCount}</span>
//             <span>Missing schema fields filled with defaults: {missingCount}</span>
//             <span>Unknown columns ignored: {unknownCount}</span>
//           </div>
//         </div>

//         <div className="mt-6 overflow-x-auto rounded-xl border border-haze-300 bg-white/70">
//           <table className="min-w-full text-left text-xs text-ink-600">
//             <thead className="bg-haze-100 text-ink-700">
//               <tr>
//                 {headers.map((header) => (
//                   <th key={header} className="px-4 py-2 font-semibold">
//                     {header}
//                   </th>
//                 ))}
//               </tr>
//             </thead>
//             <tbody className="divide-y divide-haze-200">
//               {rows.slice(0, 10).map((row, index) => (
//                 <tr key={index}>
//                   {headers.map((header) => (
//                     <td key={header} className="px-4 py-2">
//                       {row.raw[header] ?? "-"}
//                     </td>
//                   ))}
//                 </tr>
//               ))}
//               {rows.length === 0 && (
//                 <tr>
//                   <td
//                     className="px-4 py-6 text-center text-xs text-ink-500"
//                     colSpan={Math.max(headers.length, 1)}
//                   >
//                     Upload a CSV to preview the first 10 rows.
//                   </td>
//                 </tr>
//               )}
//             </tbody>
//           </table>
//         </div>
//       </div>

//       <div className="section-glass rounded-2xl p-6 shadow-card">
//         <h3 className="text-sm font-semibold text-ink-700">Batch Output</h3>
//         <p className="mt-2 text-sm text-ink-500">
//           Run batch inference to generate probabilities and download a results file.
//         </p>
//         <div className="mt-4 flex flex-wrap gap-3">
//           <Button type="button" onClick={runBatch}>
//             Run batch inference
//           </Button>
//           <Button
//             type="button"
//             variant="secondary"
//             className="text-xs"
//             onClick={downloadResults}
//           >
//             Download results CSV
//           </Button>
//         </div>
//         <div className="mt-4 rounded-xl border border-haze-300 bg-white/70 p-4 text-xs text-ink-600">
//           {results ? (
//             <p>
//               Completed {results.length} samples. Output includes probabilities for AD, DLB, PSP,
//               the predicted class, and missing field counts.
//             </p>
//           ) : (
//             <p>No batch results generated yet.</p>
//           )}
//         </div>
//       </div>
//     </div>
//   );
// }

"use client";

import type { ChangeEvent } from "react";
import { useState } from "react";
import { Button } from "./Button";
import { apiClient } from "../lib/api";
import type { FeatureInput } from "./simulator";
import featuresSchema from "../src/config/features_schema.json";

type FieldSchema = {
  key: string;
  default: number;
};

type FigureSchema = {
  required: FieldSchema[];
  advanced: FieldSchema[];
};

type Schema = {
  figures: FigureSchema[];
};

type BatchRow = {
  raw: Record<string, string>;
  features: FeatureInput;
};

type ParseResult =
  | {
      ok: true;
      rows: BatchRow[];
      headers: string[];
      recognizedKeys: string[];
      missingKeys: string[];
      unknownKeys: string[];
    }
  | { ok: false; error: string };

const schema = featuresSchema as Schema;
const schemaKeys = Array.from(
  new Set(
    schema.figures.flatMap((figure) => [...figure.required, ...figure.advanced]).map((f) => f.key)
  )
);
const schemaDefaults: Record<string, number> = {};
schema.figures.forEach((figure) => {
  [...figure.required, ...figure.advanced].forEach((field) => {
    schemaDefaults[field.key] = field.default;
  });
});

function parseCsv(text: string): ParseResult {
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0);

  if (lines.length < 2) {
    return { ok: false, error: "CSV must include a header and at least one data row." };
  }

  const headers = lines[0].split(",").map((value) => value.trim());
  if (headers.length === 0) {
    return { ok: false, error: "CSV header row is empty." };
  }

  const schemaKeySet = new Set(schemaKeys);
  const recognizedKeys = headers.filter((header) => schemaKeySet.has(header));
  const unknownKeys = headers.filter((header) => !schemaKeySet.has(header));
  const missingKeys = schemaKeys.filter((key) => !headers.includes(key));

  const rows: BatchRow[] = [];

  for (const line of lines.slice(1)) {
    const values = line.split(",").map((value) => value.trim());
    if (values.length !== headers.length) {
      return { ok: false, error: "CSV column count does not match header." };
    }

    const raw: Record<string, string> = {};
    headers.forEach((header, index) => {
      raw[header] = values[index] ?? "";
    });

    const features: FeatureInput = {};
    schemaKeys.forEach((key) => {
      if (raw[key] !== undefined) {
        const rawValue = raw[key];
        if (rawValue === "") {
          features[key] = schemaDefaults[key] ?? 0;
          return;
        }
        const parsed = Number.parseFloat(rawValue);
        if (!Number.isFinite(parsed)) {
          throw new Error(`Invalid numeric value for ${key}.`);
        }
        features[key] = parsed;
      } else {
        features[key] = schemaDefaults[key] ?? 0;
      }
    });

    rows.push({ raw, features });
  }

  return { ok: true, rows, headers, recognizedKeys, missingKeys, unknownKeys };
}

function downloadTemplateCsv() {
  const headerRow = schemaKeys.join(",");
  const exampleRow = schemaKeys
    .map((key) => (schemaDefaults[key] ?? 0).toString())
    .join(",");
  const csvText = [headerRow, exampleRow].join("\n");
  const blob = new Blob([csvText], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "neurofoldnet_template.csv";
  anchor.click();
  URL.revokeObjectURL(url);
}

export function BatchPanel() {
  const [rows, setRows] = useState<BatchRow[]>([]);
  const [parseError, setParseError] = useState<string | null>(null);
  const [fileLabel, setFileLabel] = useState<string | null>(null);
  const [headers, setHeaders] = useState<string[]>([]);
  const [recognizedCount, setRecognizedCount] = useState(0);
  const [missingCount, setMissingCount] = useState(0);
  const [unknownCount, setUnknownCount] = useState(0);

  // Backend integration state
  const [batchResults, setBatchResults] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);

  async function handleUpload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    setUploadedFile(file);

    try {
      const text = await file.text();
      const parsed = parseCsv(text);

      if (!parsed.ok) {
        setParseError(parsed.error);
        setRows([]);
        setBatchResults(null);
        setHeaders([]);
        setRecognizedCount(0);
        setMissingCount(0);
        setUnknownCount(0);
        return;
      }

      setParseError(null);
      setRows(parsed.rows);
      setBatchResults(null);
      setHeaders(parsed.headers);
      setRecognizedCount(parsed.recognizedKeys.length);
      setMissingCount(parsed.missingKeys.length);
      setUnknownCount(parsed.unknownKeys.length);
      setFileLabel(file.name);
      event.target.value = "";
      setError(null);
    } catch (error) {
      setParseError(error instanceof Error ? error.message : "Failed to parse CSV.");
      setRows([]);
      setBatchResults(null);
      setHeaders([]);
      setRecognizedCount(0);
      setMissingCount(0);
      setUnknownCount(0);
    }
  }

  async function runBatch() {
    if (!uploadedFile) {
      setError("Please upload a CSV file first.");
      return;
    }

    if (rows.length === 0) {
      setError("CSV file is empty or invalid.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Send CSV to Flask backend for batch prediction
      const result = await apiClient.predictBatch(uploadedFile);
      setBatchResults(result);
      setParseError(null);
    } catch (err: any) {
      console.error('Batch prediction failed:', err);
      setError(err.message || 'Batch prediction failed. Please try again.');
      setBatchResults(null);
    } finally {
      setLoading(false);
    }
  }

  function downloadResults() {
    if (!batchResults || !batchResults.predictions || batchResults.predictions.length === 0) {
      return;
    }

    // Build CSV from results
    const csvHeaders = [...headers, "prediction", "prob_AD", "prob_DLB", "prob_PSP", "confidence"];
    const csvRows = batchResults.predictions.map((pred: any, idx: number) => {
      const originalRow = rows[idx]?.raw || {};
      const rowValues = headers.map(h => originalRow[h] || "");
      
      if (pred.error) {
        return [...rowValues, "ERROR", "", "", "", pred.error].join(",");
      }

      return [
        ...rowValues,
        pred.prediction || "",
        pred.probabilities?.AD?.toFixed(4) || "",
        pred.probabilities?.DLB?.toFixed(4) || "",
        pred.probabilities?.PSP?.toFixed(4) || "",
        pred.max_probability?.toFixed(4) || ""
      ].join(",");
    });

    const csvText = [csvHeaders.join(","), ...csvRows].join("\n");
    const blob = new Blob([csvText], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `neurofoldnet_results_${new Date().toISOString().slice(0, 10)}.csv`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="grid gap-6 lg:grid-cols-[1.1fr,0.9fr]">
      <div className="section-glass rounded-2xl p-6 shadow-card">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h2 className="font-display text-xl font-semibold text-ink-900">Batch Inference</h2>
          {fileLabel && <span className="text-xs text-ink-500">{fileLabel}</span>}
        </div>
        <p className="mt-2 text-sm text-ink-500">
          Upload a CSV with any subset of schema fields. Missing fields are filled with defaults.
        </p>
        <div className="mt-4 flex flex-wrap gap-3">
          <input
            type="file"
            accept=".csv,text/csv"
            onChange={handleUpload}
            className="block w-full text-xs text-ink-600 file:mr-3 file:rounded-full file:border-0 file:bg-ink-900 file:px-3 file:py-1 file:text-xs file:font-semibold file:text-white"
          />
          <Button type="button" variant="secondary" className="text-xs" onClick={downloadTemplateCsv}>
            Download template CSV
          </Button>
        </div>
        {parseError && <p className="mt-3 text-xs text-red-600">{parseError}</p>}

        <div className="mt-4 rounded-xl border border-haze-300 bg-white/70 p-4 text-xs text-ink-600">
          <div className="flex flex-wrap gap-4">
            <span>Recognized schema fields: {recognizedCount}</span>
            <span>Missing schema fields filled with defaults: {missingCount}</span>
            <span>Unknown columns ignored: {unknownCount}</span>
          </div>
        </div>

        <div className="mt-6 overflow-x-auto rounded-xl border border-haze-300 bg-white/70">
          <table className="min-w-full text-left text-xs text-ink-600">
            <thead className="bg-haze-100 text-ink-700">
              <tr>
                {headers.map((header) => (
                  <th key={header} className="px-4 py-2 font-semibold">
                    {header}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-haze-200">
              {rows.slice(0, 10).map((row, index) => (
                <tr key={index}>
                  {headers.map((header) => (
                    <td key={header} className="px-4 py-2">
                      {row.raw[header] ?? "-"}
                    </td>
                  ))}
                </tr>
              ))}
              {rows.length === 0 && (
                <tr>
                  <td
                    className="px-4 py-6 text-center text-xs text-ink-500"
                    colSpan={Math.max(headers.length, 1)}
                  >
                    Upload a CSV to preview the first 10 rows.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      <div className="section-glass rounded-2xl p-6 shadow-card">
        <h3 className="text-sm font-semibold text-ink-700">Batch Output</h3>
        <p className="mt-2 text-sm text-ink-500">
          Run batch inference to generate probabilities via NeuroFoldNet backend.
        </p>
        
        <div className="mt-4 flex flex-wrap gap-3">
          <Button 
            type="button" 
            onClick={runBatch}
            disabled={loading || rows.length === 0}
          >
            {loading ? 'Processing...' : 'Run batch inference'}
          </Button>
          <Button
            type="button"
            variant="secondary"
            className="text-xs"
            onClick={downloadResults}
            disabled={!batchResults}
          >
            Download results CSV
          </Button>
        </div>

        {error && (
          <div className="mt-4 rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-xs text-red-800">
            {error}
          </div>
        )}

        {batchResults && (
          <div className="mt-4 space-y-4">
            {/* Summary */}
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
              <div className="rounded-lg border border-haze-300 bg-white/80 p-3">
                <p className="text-xs text-ink-500">Total Samples</p>
                <p className="text-xl font-semibold text-ink-900">{batchResults.total_samples}</p>
              </div>
              <div className="rounded-lg border border-green-300 bg-green-50 p-3">
                <p className="text-xs text-green-600">Successful</p>
                <p className="text-xl font-semibold text-green-900">{batchResults.successful_predictions}</p>
              </div>
              <div className="rounded-lg border border-red-300 bg-red-50 p-3">
                <p className="text-xs text-red-600">Failed</p>
                <p className="text-xl font-semibold text-red-900">{batchResults.failed_predictions}</p>
              </div>
            </div>

            {/* Prediction Distribution */}
            <div className="rounded-xl border border-haze-300 bg-white/70 p-4">
              <p className="text-xs font-semibold text-ink-700 mb-3">Prediction Distribution</p>
              <div className="space-y-2">
                {Object.entries(batchResults.prediction_summary || {}).map(([disease, count]: any) => (
                  <div key={disease} className="flex items-center justify-between text-sm">
                    <span className="font-semibold text-ink-900">{disease}</span>
                    <span className="text-ink-600">{count} samples</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Sample Results */}
            <div className="rounded-xl border border-haze-300 bg-white/70 p-4">
              <p className="text-xs font-semibold text-ink-700 mb-3">
                Sample Results (first 5)
              </p>
              <div className="space-y-2 text-xs">
                {batchResults.predictions.slice(0, 5).map((pred: any) => (
                  <div key={pred.row_index} className="flex items-center justify-between border-b border-haze-200 pb-2">
                    <span className="text-ink-600">Row {pred.row_index}</span>
                    <span className="font-semibold text-ink-900">
                      {pred.prediction || 'ERROR'}
                    </span>
                    <span className="text-ink-600">
                      {pred.max_probability ? `${(pred.max_probability * 100).toFixed(1)}%` : pred.error}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {!batchResults && !loading && (
          <div className="mt-4 rounded-xl border border-haze-300 bg-white/70 p-4 text-xs text-ink-600">
            <p>No batch results generated yet. Upload a CSV and run batch inference.</p>
          </div>
        )}

        {loading && (
          <div className="mt-4 rounded-xl border border-haze-300 bg-white/70 p-4 text-center text-xs text-ink-600">
            <div className="animate-pulse">
              Processing {rows.length} samples with NeuroFoldNet...
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
