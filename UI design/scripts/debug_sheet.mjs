import fs from "fs";
import xlsx from "xlsx";

const [filePath, sheetName, startRowRaw, endRowRaw, startColRaw, endColRaw] =
  process.argv.slice(2);

if (!filePath || !fs.existsSync(filePath)) {
  console.error("Provide a valid Excel file path.");
  process.exit(1);
}

const workbook = xlsx.readFile(filePath, { cellDates: false });
const sheet = workbook.Sheets[sheetName];
if (!sheet) {
  console.error(`Sheet not found: ${sheetName}`);
  process.exit(1);
}

const toNumber = (value, fallback) =>
  Number.isFinite(Number.parseInt(value, 10)) ? Number.parseInt(value, 10) : fallback;

const startRow = toNumber(startRowRaw, 0);
const endRow = toNumber(endRowRaw, startRow + 20);
const startCol = toNumber(startColRaw, 0);
const endCol = toNumber(endColRaw, startCol + 20);

const matrix = xlsx.utils.sheet_to_json(sheet, { header: 1, raw: true });

function normalize(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "string") return value.trim();
  return value;
}

for (let r = startRow; r <= endRow; r += 1) {
  const row = matrix[r] || [];
  const slice = [];
  for (let c = startCol; c <= endCol; c += 1) {
    slice.push(normalize(row[c]));
  }
  console.log(`${r}\t${JSON.stringify(slice)}`);
}
