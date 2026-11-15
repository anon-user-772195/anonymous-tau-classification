import fs from "fs";
import path from "path";
import xlsx from "xlsx";

const filePath = process.argv[2];
if (!filePath || !fs.existsSync(filePath)) {
  console.error("Provide a valid Excel file path.");
  process.exit(1);
}

const workbook = xlsx.readFile(filePath, { cellDates: false });

function normalizeCell(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "string") return value.trim();
  if (typeof value === "number") return value;
  return String(value).trim();
}

function isRowEmpty(row) {
  return row.every((cell) => {
    const value = normalizeCell(cell);
    return value === "";
  });
}

function getBlocks(matrix) {
  const blocks = [];
  let current = [];
  matrix.forEach((row) => {
    if (isRowEmpty(row)) {
      if (current.length) {
        blocks.push(current);
        current = [];
      }
    } else {
      current.push(row);
    }
  });
  if (current.length) blocks.push(current);
  return blocks;
}

for (const sheetName of workbook.SheetNames) {
  console.log(`\n=== ${sheetName} ===`);
  const sheet = workbook.Sheets[sheetName];
  const matrix = xlsx.utils.sheet_to_json(sheet, { header: 1, raw: true });
  const blocks = getBlocks(matrix);
  blocks.forEach((block, index) => {
    const header = block[0].map((cell) => normalizeCell(cell)).filter(Boolean);
    const labelCandidate = block[0][0] ? normalizeCell(block[0][0]) : "";
    const label = typeof labelCandidate === "string" ? labelCandidate : "";
    const sampleRows = block.slice(0, 3).map((row) => row.map(normalizeCell));
    console.log(`\n-- Block ${index + 1} --`);
    if (label) {
      console.log(`Label candidate: ${label}`);
    }
    console.log("Header sample:", header.slice(0, 12));
    console.log("Row sample:", sampleRows);
  });
}
