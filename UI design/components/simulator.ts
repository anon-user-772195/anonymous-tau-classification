import featuresSchema from "../src/config/features_schema.json";
import presetData from "../src/config/feature_presets.json";

export type CoreFeatureInput = {
  fig1_afm_height_mean: number;
  fig1_afm_area_mean: number;
  fig1_afm_diameter_mean: number;
  fig3_pronase_0p5ug_1min: number;
  fig3_pronase_1p0ug_1min: number;
  fig4_mean_signal: number;
  fig6_io_response_100: number;
  fig6_ltp_early_mean: number;
};

export type FeatureInput = Record<string, number | "">;

export type Probabilities = {
  AD: number;
  DLB: number;
  PSP: number;
};

export const coreFeatureKeys: Array<keyof CoreFeatureInput> = [
  "fig1_afm_height_mean",
  "fig1_afm_area_mean",
  "fig1_afm_diameter_mean",
  "fig3_pronase_0p5ug_1min",
  "fig3_pronase_1p0ug_1min",
  "fig4_mean_signal",
  "fig6_io_response_100",
  "fig6_ltp_early_mean"
];

type Schema = {
  figures: Array<{ required: Array<{ key: string; default: number }> }>;
};

const schema = featuresSchema as Schema;
const defaultsMap: Record<string, number> = {};
schema.figures.forEach((figure) => {
  figure.required.forEach((field) => {
    defaultsMap[field.key] = field.default;
  });
});

type Logits = {
  ad: number;
  dlb: number;
  psp: number;
};

const presetProfiles = presetData as Record<string, FeatureInput>;
const presetVectors: Record<keyof Probabilities, FeatureInput> = {
  AD: presetProfiles["AD-like"] ?? {},
  DLB: presetProfiles["DLB-like"] ?? {},
  PSP: presetProfiles["PSP-like"] ?? {}
};

export const defaultFeatures: CoreFeatureInput = {
  fig1_afm_height_mean: defaultsMap.fig1_afm_height_mean ?? 1,
  fig1_afm_area_mean: defaultsMap.fig1_afm_area_mean ?? 1,
  fig1_afm_diameter_mean: defaultsMap.fig1_afm_diameter_mean ?? 1,
  fig3_pronase_0p5ug_1min: defaultsMap.fig3_pronase_0p5ug_1min ?? 1,
  fig3_pronase_1p0ug_1min: defaultsMap.fig3_pronase_1p0ug_1min ?? 1,
  fig4_mean_signal: defaultsMap.fig4_mean_signal ?? 1,
  fig6_io_response_100: defaultsMap.fig6_io_response_100 ?? 1,
  fig6_ltp_early_mean: defaultsMap.fig6_ltp_early_mean ?? 1
};

const simulatorConfig = {
  temperature: 1.3,
  weights: {
    ad: {
      length: 3.0,
      area: 2.0,
      uptake: 0.5,
      circularity: 0.6,
      branching: 0.6,
      proteolysis: 0.1,
      other: 0.1,
      bias: -0.7
    },
    psp: {
      circularity: 2.8,
      branching: 2.0,
      length: 0.2,
      area: 0.7,
      electro: 0.1,
      other: 0.05,
      bias: -0.7
    },
    dlb: {
      width: 3.0,
      branching: 0.4,
      interaction: 2.0,
      spectral: 0.1,
      antibody: 0.1,
      circularity: 0.2,
      other: 0.05,
      bias: -0.8
    }
  }
};

function softmax(values: number[]) {
  const max = Math.max(...values);
  const expValues = values.map((value) => Math.exp(value - max));
  const sum = expValues.reduce((acc, value) => acc + value, 0);
  return expValues.map((value) => value / sum);
}

function normalizeFeature(key: string, features: FeatureInput) {
  const raw = features[key];
  const rawValue = Number.isFinite(raw) ? (raw as number) : defaultsMap[key];
  const baseValue = defaultsMap[key];
  const scale = Number.isFinite(baseValue) && baseValue !== 0 ? Math.abs(baseValue) : 1;
  return (Number.isFinite(rawValue) ? rawValue : 0) / scale;
}

function normalizeValue(key: string, value: number | undefined) {
  const baseValue = defaultsMap[key];
  const scale = Number.isFinite(baseValue) && baseValue !== 0 ? Math.abs(baseValue) : 1;
  const rawValue = Number.isFinite(value) ? (value as number) : 0;
  return rawValue / scale;
}

function meanByKeys(keys: string[], features: FeatureInput) {
  if (!keys.length) return 0;
  const total = keys.reduce((acc, key) => acc + normalizeFeature(key, features), 0);
  return total / keys.length;
}

function unique(keys: string[]) {
  return Array.from(new Set(keys));
}

function keysByKeywords(allKeys: string[], keywords: string[]) {
  const lowered = keywords.map((keyword) => keyword.toLowerCase());
  return allKeys.filter((key) =>
    lowered.some((keyword) => key.toLowerCase().includes(keyword))
  );
}

function keysByPrefix(allKeys: string[], prefix: string) {
  return allKeys.filter((key) => key.startsWith(prefix));
}

export function computeLogits(features: FeatureInput): Logits {
  const keys = coreFeatureKeys as string[];
  const distanceFor = (label: keyof Probabilities) => {
    const preset = presetVectors[label];
    return keys.reduce((acc, key) => {
      const current = normalizeFeature(key, features);
      const target = normalizeValue(key, preset[key]);
      const diff = current - target;
      return acc + diff * diff;
    }, 0);
  };

  // Higher similarity -> higher logit (negative distance).
  const ad = -distanceFor("AD");
  const dlb = -distanceFor("DLB");
  const psp = -distanceFor("PSP");

  return { ad, dlb, psp };
}

export function computeProbabilities(features: FeatureInput): Probabilities {
  const { temperature } = simulatorConfig;
  const { ad, dlb, psp } = computeLogits(features);
  const [adProb, dlbProb, pspProb] = softmax([
    ad * temperature,
    dlb * temperature,
    psp * temperature
  ]);

  return {
    AD: adProb,
    DLB: dlbProb,
    PSP: pspProb
  };
}

export function getPredictedClass(probabilities: Probabilities) {
  return (Object.entries(probabilities) as Array<[keyof Probabilities, number]>).sort(
    (a, b) => b[1] - a[1]
  )[0][0];
}

export function getSimulatorConfig() {
  return simulatorConfig;
}
