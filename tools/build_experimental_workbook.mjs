import fs from "node:fs/promises";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { SpreadsheetFile, Workbook } = require("@oai/artifact-tool");

const [, , payloadPath, outputPath] = process.argv;
if (!payloadPath || !outputPath) {
  throw new Error("Usage: node build_experimental_workbook.mjs PAYLOAD_JSON OUTPUT_XLSX");
}

const payload = JSON.parse(await fs.readFile(payloadPath, "utf8"));
const workbook = Workbook.create();

for (const sheetData of payload.sheets) {
  const sheet = workbook.worksheets.add(sheetData.name);
  sheet.showGridLines = false;
  const rows = sheetData.rows;
  if (!rows.length) continue;
  const headers = Object.keys(rows[0]);
  const matrix = [headers, ...rows.map((row) => headers.map((header) => row[header] ?? null))];
  const range = sheet.getRangeByIndexes(0, 0, matrix.length, headers.length);
  range.values = matrix;
  sheet.freezePanes.freezeRows(1);
  sheet.getRangeByIndexes(0, 0, 1, headers.length).format = {
    fill: "#17365D",
    font: { bold: true, color: "#FFFFFF" },
    wrapText: true,
  };
  if (matrix.length > 1) {
    sheet.getRangeByIndexes(1, 0, matrix.length - 1, headers.length).format = {
      fill: "#F7F9FC",
      font: { color: "#1F2937" },
    };
  }
  for (let col = 0; col < headers.length; col += 1) {
    const longest = Math.max(...matrix.map((row) => String(row[col] ?? "").length));
    const cap = sheetData.name === "Config" ? (col === 0 ? 52 : 46) : 34;
    sheet.getRangeByIndexes(0, col, matrix.length, 1).format.columnWidth = Math.min(Math.max(longest + 2, 12), cap);
    const header = headers[col];
    if (/map|precision|recall|delta|p_value/i.test(header)) {
      sheet.getRangeByIndexes(1, col, matrix.length - 1, 1).format.numberFormat = "0.000";
    } else if (/params_m|flops_g|fps|latency_ms/i.test(header)) {
      sheet.getRangeByIndexes(1, col, matrix.length - 1, 1).format.numberFormat = "0.0";
    }
  }
  range.format.rowHeight = 20;
  sheet.getRangeByIndexes(0, 0, matrix.length, headers.length).format.borders = {
    top: { style: "continuous", color: "#D5DCE5" },
    bottom: { style: "continuous", color: "#D5DCE5" },
    left: { style: "continuous", color: "#D5DCE5" },
    right: { style: "continuous", color: "#D5DCE5" },
    insideHorizontal: { style: "continuous", color: "#E5EAF0" },
    insideVertical: { style: "continuous", color: "#E5EAF0" },
  };
}

const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(outputPath);
console.log(`Saved ${outputPath}`);
