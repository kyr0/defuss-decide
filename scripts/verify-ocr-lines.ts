// Real-model comparison of the OCR check on the whole image and line by line on located strips:
// `node scripts/verify-ocr-lines.ts [cpu|webgpu|auto]` (local, needs models/vela3-fast-onnx/{vision-box,vision-ocr}/fp32).
//
// For every image of tests/fixtures/ocr-lines.json each line is checked as written and with one character changed
// (`corrupt`). Whole image: each line alone against the full image. Lines: the provider's `lines: true` check of all
// lines at once, a line counting as flagged when a span falls inside it. A flag on an unchanged line is a false alarm,
// on a changed line a catch. Writes reports/ocr-lines/node-<backend>.json. WHY measured per line: the line check is a
// cost (one box request plus one OCR run per line) that only pays off if it catches more without more false alarms.
// Exits 1 only when a run fails. VERIFIED: 83 lines on 2026-11-08: whole image 8 false alarms and 42 catches, line by
// line 7 and 55 (reports/ocr-lines/node-auto.json).
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { loadVela, loadVelaProvider } from "../src/node.ts";
import { ocrLines } from "../src/ocr-lines.ts";
import type { SpanAnswer } from "../src/types.ts";

/** One visible change per line: the last digit plus 3 (mod 10), else the 2nd and 3rd letters of its longest word swapped. */
export function corrupt(line: string): string {
  const d = [...line.matchAll(/\d/g)].at(-1);
  if (d) return line.slice(0, d.index) + String((Number(d[0]) + 3) % 10) + line.slice(d.index! + 1);
  const word = line.split(/\s+/).sort((a, b) => b.length - a.length)[0];
  const i = line.indexOf(word);
  return line.slice(0, i + 1) + word[2] + word[1] + line.slice(i + 3);
}

/** Per line of `text`, whether a span of `answer` overlaps it. */
export function flaggedLines(text: string, answer: SpanAnswer): boolean[] {
  return ocrLines(text).map((l) => {
    const end = l.start + Array.from(l.text).length;
    return answer.spans.some((s) => s.start < end && s.end > l.start);
  });
}

if (import.meta.main) {
  const ROOT = fileURLToPath(new URL("..", import.meta.url));
  const backend = (process.argv[2] ?? "auto") as "auto" | "cpu" | "webgpu";
  const log = (level: string, msg: string) => console.log(`${new Date().toISOString()} ${level} ${msg}`);
  const { images } = JSON.parse(readFileSync(join(ROOT, "tests/fixtures/ocr-lines.json"), "utf8")) as { images: { file: string; lines: string[] }[] };
  const bundle = (t: string) => join(ROOT, "models/vela3-fast-onnx", t, "fp32");
  const check = { type: "span" as const, task: "ocr_check" as const, criteria: ["ocr_error"] };
  const rows: { file: string; line: string; changed: string; whole: [boolean, boolean]; lines: [boolean, boolean] }[] = [];
  const whole = await loadVela(bundle("vision-ocr"), { backend });
  try {
    for (const img of images)
      for (const line of img.lines) {
        const flag = async (text: string) => ((await whole.systemOne({ state: { ocr: text }, images: [join(ROOT, img.file)], questions: { o: check } })).answers.o as SpanAnswer).spans.length > 0;
        rows.push({ file: img.file, line, changed: corrupt(line), whole: [await flag(line), await flag(corrupt(line))], lines: [false, false] });
      }
  } finally {
    await whole.dispose();
  }
  const provider = await loadVelaProvider({ vision: { box: bundle("vision-box"), ocr: bundle("vision-ocr") } }, { backend });
  let used = "";
  try {
    for (const img of images) {
      const own = rows.filter((r) => r.file === img.file);
      for (const [k, text] of [own.map((r) => r.line).join("\n"), own.map((r) => r.changed).join("\n")].entries()) {
        const r = await provider.systemOne({ state: { ocr: text }, images: [join(ROOT, img.file)], questions: { o: { ...check, lines: true } } });
        used = r.backend;
        flaggedLines(text, r.answers.o as SpanAnswer).forEach((f, i) => (own[i].lines[k] = f));
      }
      const count = (m: "whole" | "lines", k: 0 | 1) => own.filter((r) => r[m][k]).length;
      log("INFO", `ocr-lines file=${img.file} lines=${own.length} whole_false_alarms=${count("whole", 0)} whole_catches=${count("whole", 1)} lines_false_alarms=${count("lines", 0)} lines_catches=${count("lines", 1)}`);
    }
  } finally {
    await provider.dispose();
  }
  const total = (m: "whole" | "lines", k: 0 | 1) => rows.filter((r) => r[m][k]).length;
  const summary = { lines: rows.length, whole: { falseAlarms: total("whole", 0), catches: total("whole", 1) }, byLine: { falseAlarms: total("lines", 0), catches: total("lines", 1) } };
  log("INFO", `ocr-lines total=${rows.length} whole=${summary.whole.falseAlarms}/${summary.whole.catches} by_line=${summary.byLine.falseAlarms}/${summary.byLine.catches} (false alarms/catches)`);
  mkdirSync(join(ROOT, "reports/ocr-lines"), { recursive: true });
  const runtime = process.versions.bun === undefined ? `node ${process.version}` : `bun ${process.versions.bun}`;
  writeFileSync(join(ROOT, `reports/ocr-lines/node-${backend}.json`), JSON.stringify({ runtime, platform: `${process.platform}/${process.arch}`, backend: used, summary, rows }, null, 1) + "\n");
}
