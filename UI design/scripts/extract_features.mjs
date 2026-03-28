import fs from "fs";
import path from "path";
import xlsx from "xlsx";

const filePath = process.argv[2];
if (!filePath || !fs.existsSync(filePath)) {
  console.error("Provide a valid Excel file path.");
  process.exit(1);
}

const outputSchemaPath = path.resolve("src", "config", "features_schema.json");
const outputPresetsPath = path.resolve("src", "config", "feature_presets.json");

const workbook = xlsx.readFile(filePath, { cellDates: false });

const isNumber = (value) => typeof value === "number" && Number.isFinite(value);
const mean = (values) => values.reduce((acc, value) => acc + value, 0) / values.length;

const weightedMean = (values, weights) => {
  const totalWeight = weights.reduce((acc, weight) => acc + weight, 0);
  if (totalWeight === 0) return 0;
  return values.reduce((acc, value, index) => acc + value * weights[index], 0) / totalWeight;
};

const weightedMedian = (values, weights) => {
  const totalWeight = weights.reduce((acc, weight) => acc + weight, 0);
  if (totalWeight === 0) return 0;
  let cumulative = 0;
  for (let i = 0; i < values.length; i += 1) {
    cumulative += weights[i];
    if (cumulative >= totalWeight / 2) {
      return values[i];
    }
  }
  return values[values.length - 1] ?? 0;
};

const round = (value) => Math.round(value * 10000) / 10000;

const averageGroups = (groupValues) =>
  round(mean([groupValues.AD, groupValues.DLB, groupValues.PSP]));

function parseFig1() {
  const sheet = workbook.Sheets["Fig 1"];
  const matrix = xlsx.utils.sheet_to_json(sheet, { header: 1, raw: true });

  const dataRows = [];
  for (let r = 3; r < matrix.length; r += 1) {
    const row = matrix[r] || [];
    if (!isNumber(row[0]) && !isNumber(row[7]) && !isNumber(row[14])) {
      break;
    }
    dataRows.push(row);
  }

  const groupColumns = {
    AD: { height: 0, area: 2, diameter: 4 },
    DLB: { height: 7, area: 9, diameter: 11 },
    PSP: { height: 14, area: 16, diameter: 18 }
  };

  const afmStats = {};
  Object.entries(groupColumns).forEach(([group, cols]) => {
    const heightValues = [];
    const heightCounts = [];
    const areaValues = [];
    const areaCounts = [];
    const diameterValues = [];
    const diameterCounts = [];

    dataRows.forEach((row) => {
      const height = row[cols.height];
      const heightCount = row[cols.height + 1];
      const area = row[cols.area];
      const areaCount = row[cols.area + 1];
      const diameter = row[cols.diameter];
      const diameterCount = row[cols.diameter + 1];

      if (isNumber(height) && isNumber(heightCount)) {
        heightValues.push(height);
        heightCounts.push(heightCount);
      }
      if (isNumber(area) && isNumber(areaCount)) {
        areaValues.push(area);
        areaCounts.push(areaCount);
      }
      if (isNumber(diameter) && isNumber(diameterCount)) {
        diameterValues.push(diameter);
        diameterCounts.push(diameterCount);
      }
    });

    afmStats[group] = {
      height_mean: round(weightedMean(heightValues, heightCounts)),
      height_median: round(weightedMedian(heightValues, heightCounts)),
      area_mean: round(weightedMean(areaValues, areaCounts)),
      area_median: round(weightedMedian(areaValues, areaCounts)),
      diameter_mean: round(weightedMean(diameterValues, diameterCounts)),
      diameter_median: round(weightedMedian(diameterValues, diameterCounts))
    };
  });

  const cdSeries = {
    PSP: { waveCol: 33, valueCol: 34 },
    DLB: { waveCol: 35, valueCol: 36 },
    AD: { waveCol: 37, valueCol: 38 }
  };

  const cdStats = {};
  Object.entries(cdSeries).forEach(([group, cols]) => {
    const points = [];
    for (let r = 2; r < matrix.length; r += 1) {
      const row = matrix[r] || [];
      const wavelength = row[cols.waveCol];
      const value = row[cols.valueCol];
      if (!isNumber(wavelength) || !isNumber(value)) {
        break;
      }
      points.push({ wavelength, value });
    }

    let minValue = Number.POSITIVE_INFINITY;
    let minWavelength = 0;
    points.forEach((point) => {
      if (point.value < minValue) {
        minValue = point.value;
        minWavelength = point.wavelength;
      }
    });

    cdStats[group] = {
      cd_min: round(minValue),
      cd_min_wavelength: round(minWavelength)
    };
  });

  return { afmStats, cdStats };
}

function parseSpectraBlock(matrix, label) {
  const labelRow = matrix.findIndex((row) => row?.[0] === label);
  if (labelRow === -1) {
    return null;
  }
  const headerRow = matrix[labelRow + 1] || [];
  const startPositions = {
    AD: headerRow.findIndex((cell) => cell === "AD"),
    DLB: headerRow.findIndex((cell) => cell === "DLB"),
    PSP: headerRow.findIndex((cell) => cell === "PSP"),
    rTauO: headerRow.findIndex((cell) => cell === "rTauO")
  };

  const groupWidth = startPositions.DLB - startPositions.AD;
  const pspWidth = startPositions.rTauO - startPositions.PSP;

  const widths = {
    AD: groupWidth,
    DLB: groupWidth,
    PSP: pspWidth
  };

  const points = { AD: [], DLB: [], PSP: [] };
  for (let r = labelRow + 2; r < matrix.length; r += 1) {
    const row = matrix[r] || [];
    const wavelength = row[0];
    if (!isNumber(wavelength)) {
      break;
    }

    (["AD", "DLB", "PSP"]).forEach((group) => {
      const start = startPositions[group];
      const slice = row.slice(start, start + widths[group]).filter(isNumber);
      const intensity = slice.length ? mean(slice) : 0;
      points[group].push({ wavelength, intensity });
    });
  }

  const peaks = {};
  (["AD", "DLB", "PSP"]).forEach((group) => {
    let maxIntensity = -Infinity;
    let peakWavelength = 0;
    points[group].forEach((point) => {
      if (point.intensity > maxIntensity) {
        maxIntensity = point.intensity;
        peakWavelength = point.wavelength;
      }
    });
    peaks[group] = {
      peak_wavelength: round(peakWavelength),
      peak_intensity: round(maxIntensity)
    };
  });

  return peaks;
}

function parseFig2() {
  const sheet = workbook.Sheets["Fig 2"];
  const matrix = xlsx.utils.sheet_to_json(sheet, { header: 1, raw: true });

  const dyes = ["ThT", "Curcumin", "Bis-ANS", "FSB"];
  const results = {};

  dyes.forEach((dye) => {
    const peaks = parseSpectraBlock(matrix, dye);
    if (peaks) {
      results[dye] = peaks;
    }
  });

  return results;
}

function parseFig3() {
  const sheet = workbook.Sheets["Fig 3"];
  const matrix = xlsx.utils.sheet_to_json(sheet, { header: 1, raw: true });
  const headerRow = matrix[1] || [];
  const subHeaderRow = matrix[2] || [];

  const groupStarts = {
    AD: headerRow.findIndex((cell) => cell === "AD"),
    DLB: headerRow.findIndex((cell) => cell === "DLB"),
    PSP: headerRow.findIndex((cell) => cell === "PSP")
  };

  const groupCols = {};
  Object.entries(groupStarts).forEach(([group, start]) => {
    const dose05 = subHeaderRow.findIndex(
      (cell, index) => index > start && typeof cell === "string" && cell.includes("0.5")
    );
    const dose10 = subHeaderRow.findIndex(
      (cell, index) => index > start && typeof cell === "string" && cell.includes("1.0")
    );
    groupCols[group] = { dose05, dose10 };
  });

  const series = {
    AD: { dose05: [], dose10: [] },
    DLB: { dose05: [], dose10: [] },
    PSP: { dose05: [], dose10: [] }
  };

  for (let r = 3; r < matrix.length; r += 1) {
    const row = matrix[r] || [];
    const time = row[0];
    if (!isNumber(time)) {
      break;
    }
    (["AD", "DLB", "PSP"]).forEach((group) => {
      const dose05Value = row[groupCols[group].dose05];
      const dose10Value = row[groupCols[group].dose10];
      if (isNumber(dose05Value)) {
        series[group].dose05.push({ time, value: dose05Value });
      }
      if (isNumber(dose10Value)) {
        series[group].dose10.push({ time, value: dose10Value });
      }
    });
  }

  const computeAuc = (points) => {
    if (points.length < 2) return 0;
    let auc = 0;
    for (let i = 1; i < points.length; i += 1) {
      const prev = points[i - 1];
      const curr = points[i];
      auc += ((prev.value + curr.value) / 2) * (curr.time - prev.time);
    }
    return auc;
  };

  const summary = {};
  (["AD", "DLB", "PSP"]).forEach((group) => {
    const dose05 = series[group].dose05;
    const dose10 = series[group].dose10;
    const valueAt = (points, target) =>
      points.find((point) => point.time === target)?.value ?? 0;

    summary[group] = {
      dose05_1min: round(valueAt(dose05, 1)),
      dose05_15min: round(valueAt(dose05, 15)),
      dose10_1min: round(valueAt(dose10, 1)),
      dose10_15min: round(valueAt(dose10, 15)),
      dose05_auc: round(computeAuc(dose05)),
      dose10_auc: round(computeAuc(dose10))
    };
  });

  return summary;
}

function parseFig4() {
  const sheet = workbook.Sheets["Fig 4"];
  const matrix = xlsx.utils.sheet_to_json(sheet, { header: 1, raw: true });
  const headerRow = matrix[1] || [];
  const starts = {
    AD: headerRow.findIndex((cell) => cell === "AD"),
    DLB: headerRow.findIndex((cell) => cell === "DLB"),
    PSP: headerRow.findIndex((cell) => cell === "PSP")
  };
  const width = starts.DLB - starts.AD;

  const antibodyRows = [];
  for (let r = 2; r < matrix.length; r += 1) {
    const row = matrix[r] || [];
    const label = row[0];
    if (!label) {
      break;
    }
    antibodyRows.push({ label: String(label).replace(/\s+/g, " ").trim(), row });
  }

  const result = {};
  (["AD", "DLB", "PSP"]).forEach((group) => {
    result[group] = {};
  });

  antibodyRows.forEach(({ label, row }) => {
    (["AD", "DLB", "PSP"]).forEach((group) => {
      const start = starts[group];
      const slice = row.slice(start, start + width).filter(isNumber);
      result[group][label] = round(slice.length ? mean(slice) : 0);
    });
  });

  (["AD", "DLB", "PSP"]).forEach((group) => {
    const values = antibodyRows.map(({ label }) => result[group][label]);
    result[group].overall_mean = round(mean(values));
  });

  return result;
}

function parseFig5() {
  const sheet = workbook.Sheets["Fig 5"];
  const matrix = xlsx.utils.sheet_to_json(sheet, { header: 1, raw: true });

  const figStarts = {
    fig5e: 0,
    fig5f: 8,
    fig5g: 16
  };

  const results = {
    AD: {},
    DLB: {},
    PSP: {}
  };

  Object.entries(figStarts).forEach(([figKey, start]) => {
    const columns = {
      "12hr": { AD: start, DLB: start + 1, PSP: start + 2 },
      "24hr": { AD: start + 4, DLB: start + 5, PSP: start + 6 }
    };

    const values = {
      AD: { "12hr": [], "24hr": [] },
      DLB: { "12hr": [], "24hr": [] },
      PSP: { "12hr": [], "24hr": [] }
    };

    for (let r = 3; r < matrix.length; r += 1) {
      const row = matrix[r] || [];
      const hasData =
        isNumber(row[start]) || isNumber(row[start + 1]) || isNumber(row[start + 2]);
      if (!hasData) {
        if (r > 3) {
          break;
        }
      }

      (["AD", "DLB", "PSP"]).forEach((group) => {
        (["12hr", "24hr"]).forEach((time) => {
          const value = row[columns[time][group]];
          if (isNumber(value)) {
            values[group][time].push(value);
          }
        });
      });
    }

    (["AD", "DLB", "PSP"]).forEach((group) => {
      results[group][`${figKey}_12hr`] = round(
        values[group]["12hr"].length ? mean(values[group]["12hr"]) : 0
      );
      results[group][`${figKey}_24hr`] = round(
        values[group]["24hr"].length ? mean(values[group]["24hr"]) : 0
      );
    });
  });

  return results;
}

function parseFig6() {
  const sheet = workbook.Sheets["Fig 6"];
  const matrix = xlsx.utils.sheet_to_json(sheet, { header: 1, raw: true });

  const ioGroups = {
    DLB: 1,
    PSP: 4,
    AD: 7
  };

  const ioSeries = { AD: [], DLB: [], PSP: [] };
  for (let r = 2; r < matrix.length; r += 1) {
    const row = matrix[r] || [];
    const intensity = row[0];
    if (!isNumber(intensity)) {
      break;
    }
    Object.entries(ioGroups).forEach(([group, col]) => {
      const value = row[col];
      if (isNumber(value)) {
        ioSeries[group].push({ intensity, value });
      }
    });
  }

  const ioSummary = {};
  (["AD", "DLB", "PSP"]).forEach((group) => {
    const series = ioSeries[group];
    const valueAt = (target) => series.find((point) => point.intensity === target)?.value ?? 0;
    const v0 = valueAt(0);
    const v100 = valueAt(100);
    ioSummary[group] = {
      io_slope_0_100: round((v100 - v0) / 100),
      io_response_100: round(v100)
    };
  });

  const pprCols = { DLB: 14, PSP: 15, AD: 17 };
  const pprValues = { AD: [], DLB: [], PSP: [] };
  for (let r = 2; r < matrix.length; r += 1) {
    const row = matrix[r] || [];
    if (!isNumber(row[pprCols.DLB]) && !isNumber(row[pprCols.PSP]) && !isNumber(row[pprCols.AD])) {
      break;
    }
    (["AD", "DLB", "PSP"]).forEach((group) => {
      const value = row[pprCols[group]];
      if (isNumber(value)) {
        pprValues[group].push(value);
      }
    });
  }

  const pprSummary = {};
  (["AD", "DLB", "PSP"]).forEach((group) => {
    pprSummary[group] = round(mean(pprValues[group]));
  });

  const ltpCols = { time: 19, AD: 23, DLB: 26, PSP: 29 };
  const ltpSeries = { AD: [], DLB: [], PSP: [] };
  for (let r = 2; r < matrix.length; r += 1) {
    const row = matrix[r] || [];
    const time = row[ltpCols.time];
    if (!isNumber(time)) {
      break;
    }
    (["AD", "DLB", "PSP"]).forEach((group) => {
      const value = row[ltpCols[group]];
      if (isNumber(value)) {
        ltpSeries[group].push({ time, value });
      }
    });
  }

  const ltpSummary = {};
  (["AD", "DLB", "PSP"]).forEach((group) => {
    const series = ltpSeries[group];
    const maxTime = Math.max(...series.map((point) => point.time));
    const lastWindow = series.filter((point) => point.time >= maxTime - 9);
    const earlyWindow = series.filter((point) => point.time <= 5);
    ltpSummary[group] = {
      ltp_last10_mean: round(mean(lastWindow.map((point) => point.value))),
      ltp_early_mean: round(mean(earlyWindow.map((point) => point.value)))
    };
  });

  const fig6ECols = { DLB: 33, PSP: 34, AD: 36 };
  const fig6EValues = { AD: [], DLB: [], PSP: [] };
  for (let r = 2; r < matrix.length; r += 1) {
    const row = matrix[r] || [];
    if (!isNumber(row[fig6ECols.DLB]) && !isNumber(row[fig6ECols.PSP]) && !isNumber(row[fig6ECols.AD])) {
      break;
    }
    (["AD", "DLB", "PSP"]).forEach((group) => {
      const value = row[fig6ECols[group]];
      if (isNumber(value)) {
        fig6EValues[group].push(value);
      }
    });
  }

  const fig6ESummary = {};
  (["AD", "DLB", "PSP"]).forEach((group) => {
    fig6ESummary[group] = round(mean(fig6EValues[group]));
  });

  const summary = {};
  (["AD", "DLB", "PSP"]).forEach((group) => {
    summary[group] = {
      ...ioSummary[group],
      ppr_mean: pprSummary[group],
      ...ltpSummary[group],
      fig6e_mean: fig6ESummary[group]
    };
  });

  return summary;
}

const fig1 = parseFig1();
const fig2 = parseFig2();
const fig3 = parseFig3();
const fig4 = parseFig4();
const fig5 = parseFig5();
const fig6 = parseFig6();

const groupValues = {
  AD: {},
  DLB: {},
  PSP: {}
};

const fields = [];

function addField(figureId, key, label, unit, tooltip, values, min = 0, step = 0.01) {
  fields.push({
    figureId,
    key,
    label,
    unit,
    tooltip,
    min,
    step,
    values
  });

  (["AD", "DLB", "PSP"]).forEach((group) => {
    groupValues[group][key] = values[group];
  });
}

// Fig 1: AFM distributions + CD summary.
addField(
  "fig1",
  "fig1_afm_height_mean",
  "AFM height (mean)",
  "a.u.",
  "Weighted mean height from Fig 1D AFM distributions.",
  {
    AD: fig1.afmStats.AD.height_mean,
    DLB: fig1.afmStats.DLB.height_mean,
    PSP: fig1.afmStats.PSP.height_mean
  }
);
addField(
  "fig1",
  "fig1_afm_height_median",
  "AFM height (median)",
  "a.u.",
  "Weighted median height from Fig 1D AFM distributions.",
  {
    AD: fig1.afmStats.AD.height_median,
    DLB: fig1.afmStats.DLB.height_median,
    PSP: fig1.afmStats.PSP.height_median
  }
);
addField(
  "fig1",
  "fig1_afm_area_mean",
  "AFM area (mean)",
  "a.u.",
  "Weighted mean area from Fig 1D AFM distributions.",
  {
    AD: fig1.afmStats.AD.area_mean,
    DLB: fig1.afmStats.DLB.area_mean,
    PSP: fig1.afmStats.PSP.area_mean
  }
);
addField(
  "fig1",
  "fig1_afm_area_median",
  "AFM area (median)",
  "a.u.",
  "Weighted median area from Fig 1D AFM distributions.",
  {
    AD: fig1.afmStats.AD.area_median,
    DLB: fig1.afmStats.DLB.area_median,
    PSP: fig1.afmStats.PSP.area_median
  }
);
addField(
  "fig1",
  "fig1_afm_diameter_mean",
  "AFM diameter (mean)",
  "a.u.",
  "Weighted mean diameter from Fig 1D AFM distributions.",
  {
    AD: fig1.afmStats.AD.diameter_mean,
    DLB: fig1.afmStats.DLB.diameter_mean,
    PSP: fig1.afmStats.PSP.diameter_mean
  }
);
addField(
  "fig1",
  "fig1_afm_diameter_median",
  "AFM diameter (median)",
  "a.u.",
  "Weighted median diameter from Fig 1D AFM distributions.",
  {
    AD: fig1.afmStats.AD.diameter_median,
    DLB: fig1.afmStats.DLB.diameter_median,
    PSP: fig1.afmStats.PSP.diameter_median
  }
);
addField(
  "fig1",
  "fig1_cd_min_signal",
  "CD minimum signal",
  "a.u.",
  "Minimum CD signal from Fig 1G spectra.",
  {
    AD: fig1.cdStats.AD.cd_min,
    DLB: fig1.cdStats.DLB.cd_min,
    PSP: fig1.cdStats.PSP.cd_min
  },
  -10,
  0.001
);
addField(
  "fig1",
  "fig1_cd_min_wavelength",
  "CD minimum wavelength",
  "nm",
  "Wavelength at minimum CD signal from Fig 1G spectra.",
  {
    AD: fig1.cdStats.AD.cd_min_wavelength,
    DLB: fig1.cdStats.DLB.cd_min_wavelength,
    PSP: fig1.cdStats.PSP.cd_min_wavelength
  },
  180,
  0.1
);

// Fig 2: fluorescence spectra peaks.
const fig2Dyes = [
  { key: "ThT", label: "ThT" },
  { key: "Curcumin", label: "Curcumin" },
  { key: "Bis-ANS", label: "Bis-ANS" },
  { key: "FSB", label: "FSB" }
];

fig2Dyes.forEach((dye) => {
  const peaks = fig2[dye.key];
  if (!peaks) return;
  const slug = dye.key.toLowerCase().replace(/[^a-z0-9]+/g, "_");

  addField(
    "fig2",
    `fig2_${slug}_peak_nm`,
    `${dye.label} peak wavelength`,
    "nm",
    `Peak emission wavelength from Fig 2 ${dye.label} spectra.`,
    {
      AD: peaks.AD.peak_wavelength,
      DLB: peaks.DLB.peak_wavelength,
      PSP: peaks.PSP.peak_wavelength
    },
    350,
    1
  );
  addField(
    "fig2",
    `fig2_${slug}_max_intensity`,
    `${dye.label} max intensity`,
    "a.u.",
    `Maximum intensity from Fig 2 ${dye.label} spectra.`,
    {
      AD: peaks.AD.peak_intensity,
      DLB: peaks.DLB.peak_intensity,
      PSP: peaks.PSP.peak_intensity
    },
    0,
    0.001
  );
});

// Fig 3: proteolysis timecourse.
addField(
  "fig3",
  "fig3_pronase_0p5ug_1min",
  "Pronase 0.5ug/mL (1 min)",
  "a.u.",
  "Signal at 1 min from Fig 3B 0.5ug/mL timecourse.",
  {
    AD: fig3.AD.dose05_1min,
    DLB: fig3.DLB.dose05_1min,
    PSP: fig3.PSP.dose05_1min
  }
);
addField(
  "fig3",
  "fig3_pronase_0p5ug_15min",
  "Pronase 0.5ug/mL (15 min)",
  "a.u.",
  "Signal at 15 min from Fig 3B 0.5ug/mL timecourse.",
  {
    AD: fig3.AD.dose05_15min,
    DLB: fig3.DLB.dose05_15min,
    PSP: fig3.PSP.dose05_15min
  }
);
addField(
  "fig3",
  "fig3_pronase_1p0ug_1min",
  "Pronase 1.0ug/mL (1 min)",
  "a.u.",
  "Signal at 1 min from Fig 3B 1.0ug/mL timecourse.",
  {
    AD: fig3.AD.dose10_1min,
    DLB: fig3.DLB.dose10_1min,
    PSP: fig3.PSP.dose10_1min
  }
);
addField(
  "fig3",
  "fig3_pronase_1p0ug_15min",
  "Pronase 1.0ug/mL (15 min)",
  "a.u.",
  "Signal at 15 min from Fig 3B 1.0ug/mL timecourse.",
  {
    AD: fig3.AD.dose10_15min,
    DLB: fig3.DLB.dose10_15min,
    PSP: fig3.PSP.dose10_15min
  }
);
addField(
  "fig3",
  "fig3_pronase_0p5ug_auc",
  "Pronase 0.5ug/mL (AUC)",
  "a.u.",
  "Area under curve from Fig 3B 0.5ug/mL timecourse.",
  {
    AD: fig3.AD.dose05_auc,
    DLB: fig3.DLB.dose05_auc,
    PSP: fig3.PSP.dose05_auc
  }
);
addField(
  "fig3",
  "fig3_pronase_1p0ug_auc",
  "Pronase 1.0ug/mL (AUC)",
  "a.u.",
  "Area under curve from Fig 3B 1.0ug/mL timecourse.",
  {
    AD: fig3.AD.dose10_auc,
    DLB: fig3.DLB.dose10_auc,
    PSP: fig3.PSP.dose10_auc
  }
);

// Fig 4: antibody signals.
const fig4Antibodies = ["TOMA 1", "TOMA2", "TOMA3", "TOMA4", "T22", "overall_mean"];
fig4Antibodies.forEach((antibody) => {
  const key =
    antibody === "overall_mean"
      ? "fig4_mean_signal"
      : `fig4_${antibody.toLowerCase().replace(/[^a-z0-9]/g, "")}_mean`;
  const label = antibody === "overall_mean" ? "Antibody mean signal" : `${antibody} mean signal`;
  const tooltip =
    antibody === "overall_mean"
      ? "Average antibody signal across Fig 4D rows."
      : `Mean signal for ${antibody} from Fig 4D.`;

  addField(
    "fig4",
    key,
    label,
    "a.u.",
    tooltip,
    {
      AD: fig4.AD[antibody],
      DLB: fig4.DLB[antibody],
      PSP: fig4.PSP[antibody]
    },
    -0.5,
    0.001
  );
});

// Fig 5: uptake counts.
const fig5Fields = [
  { figKey: "fig5e", time: "12hr" },
  { figKey: "fig5e", time: "24hr" },
  { figKey: "fig5f", time: "12hr" },
  { figKey: "fig5f", time: "24hr" },
  { figKey: "fig5g", time: "12hr" },
  { figKey: "fig5g", time: "24hr" }
];

fig5Fields.forEach(({ figKey, time }) => {
  const key = `fig5_${figKey}_${time}`;
  const label = `${figKey.toUpperCase()} ${time} mean`;
  addField(
    "fig5",
    key,
    label,
    "a.u.",
    `Mean value from Fig 5 (${figKey.toUpperCase()}) at ${time}.`,
    {
      AD: fig5.AD[`${figKey}_${time}`],
      DLB: fig5.DLB[`${figKey}_${time}`],
      PSP: fig5.PSP[`${figKey}_${time}`]
    }
  );
});

// Fig 6: electrophysiology summaries.
addField(
  "fig6",
  "fig6_io_slope_0_100",
  "IO slope (0-100 uA)",
  "a.u.",
  "Input-output slope from Fig 6B (0-100 uA).",
  {
    AD: fig6.AD.io_slope_0_100,
    DLB: fig6.DLB.io_slope_0_100,
    PSP: fig6.PSP.io_slope_0_100
  },
  -0.01,
  0.0001
);
addField(
  "fig6",
  "fig6_io_response_100",
  "IO response at 100 uA",
  "a.u.",
  "Mean response at 100 uA from Fig 6B.",
  {
    AD: fig6.AD.io_response_100,
    DLB: fig6.DLB.io_response_100,
    PSP: fig6.PSP.io_response_100
  },
  0,
  0.001
);
addField(
  "fig6",
  "fig6_ppr_mean",
  "Paired-pulse ratio (mean)",
  "a.u.",
  "Mean paired-pulse ratio from Fig 6C.",
  {
    AD: fig6.AD.ppr_mean,
    DLB: fig6.DLB.ppr_mean,
    PSP: fig6.PSP.ppr_mean
  },
  0,
  0.001
);
addField(
  "fig6",
  "fig6_ltp_early_mean",
  "LTP early mean (<=5 min)",
  "a.u.",
  "Mean early LTP response from Fig 6D (<=5 min).",
  {
    AD: fig6.AD.ltp_early_mean,
    DLB: fig6.DLB.ltp_early_mean,
    PSP: fig6.PSP.ltp_early_mean
  },
  0,
  0.01
);
addField(
  "fig6",
  "fig6_ltp_last10_mean",
  "LTP last-10-min mean",
  "a.u.",
  "Mean late LTP response from Fig 6D (last 10 min window).",
  {
    AD: fig6.AD.ltp_last10_mean,
    DLB: fig6.DLB.ltp_last10_mean,
    PSP: fig6.PSP.ltp_last10_mean
  },
  0,
  0.01
);
addField(
  "fig6",
  "fig6_fig6e_mean",
  "Fig 6E mean response",
  "a.u.",
  "Mean response values from Fig 6E.",
  {
    AD: fig6.AD.fig6e_mean,
    DLB: fig6.DLB.fig6e_mean,
    PSP: fig6.PSP.fig6e_mean
  },
  0,
  0.01
);

const figureMap = {
  fig1: { id: "fig1", label: "Figure 1", required: [], advanced: [] },
  fig2: { id: "fig2", label: "Figure 2", required: [], advanced: [] },
  fig3: { id: "fig3", label: "Figure 3", required: [], advanced: [] },
  fig4: { id: "fig4", label: "Figure 4", required: [], advanced: [] },
  fig5: { id: "fig5", label: "Figure 5", required: [], advanced: [] },
  fig6: { id: "fig6", label: "Figure 6", required: [], advanced: [] }
};

fields.forEach((field) => {
  figureMap[field.figureId].required.push({
    key: field.key,
    label: field.label,
    unit: field.unit,
    tooltip: field.tooltip,
    min: field.min,
    step: field.step,
    default: averageGroups(field.values)
  });
});

const schema = {
  figures: Object.values(figureMap)
};

fs.writeFileSync(outputSchemaPath, `${JSON.stringify(schema, null, 2)}\n`);

const presets = {
  "AD-like": groupValues.AD,
  "DLB-like": groupValues.DLB,
  "PSP-like": groupValues.PSP
};

fs.writeFileSync(outputPresetsPath, `${JSON.stringify(presets, null, 2)}\n`);

console.log(`Wrote schema: ${outputSchemaPath}`);
console.log(`Wrote presets: ${outputPresetsPath}`);
