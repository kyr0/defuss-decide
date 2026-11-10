// Real-model check of the typed decision pipeline: `node scripts/verify-decide.ts [cpu|webgpu|auto] [fp32|q8]` (local,
// needs models/decision/{fast,vision} and models/vela3-fast-onnx/{evidence,vision-box,vision-ocr}/<precision>; the
// decision bundles exist in FP32 only). A precision other than fp32 writes reports/decide/node-<backend>-<precision>.json.
//
// Runs the hand-written cases of tests/fixtures/decide-cases.json (yes/no, or one of `options`) through VelaProvider's `decide` questions and
// writes reports/decide/node-<backend>.json with every decision, its probabilities, evidence and observations, and the
// share answered right. WHY a recorded smoke set: the pipeline combines four models, so its answers have no Python
// reference; the cases show that the steps work together and roughly how often they answer as a person would. They are
// not a benchmark, and exits 1 only when a step fails, not for a wrong decision. The record is
// reports/decide/node-<backend>.json; documentation/decide.md states its numbers. VERIFIED: 74 of 81 answered as
// labelled on 2026-11-08 (reports/decide/node-auto.json).
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { loadVelaProvider } from "../src/node.ts";
import type { DecideAnswer } from "../src/types.ts";

interface DecideCase {
  name: string;
  instructions: string;
  expected: string;
  state: unknown;
  images?: string[];
  region?: string;
  options?: Record<string, string | null>;
  lines?: boolean;
}

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const backend = (process.argv[2] ?? "auto") as "auto" | "cpu" | "webgpu";
const precision = process.argv[3] ?? "fp32";
const log = (level: string, msg: string) => console.log(`${new Date().toISOString()} ${level} ${msg}`);
const { cases } = JSON.parse(readFileSync(join(ROOT, "tests/fixtures/decide-cases.json"), "utf8")) as { cases: DecideCase[] };
const tasks = (dir: string) => join(ROOT, "models/vela3-fast-onnx", dir, precision);
const provider = await loadVelaProvider(
  {
    fast: { decision: join(ROOT, "models/decision/fast"), evidence: tasks("evidence") },
    vision: { decision: join(ROOT, "models/decision/vision"), box: tasks("vision-box"), ocr: tasks("vision-ocr") },
  },
  { backend },
);
const rows: (DecideCase & { decision: string; correct: boolean; answer: DecideAnswer; backend: string; ms: number })[] = [];
try {
  for (const c of cases) {
    const started = performance.now();
    const r = await provider.systemOne({
      state: c.state,
      images: (c.images ?? []).map((file) => join(ROOT, file)),
      questions: { d: { type: "decide", instructions: c.instructions, ...(c.region ? { region: c.region } : {}), ...(c.options ? { options: c.options } : {}), ...(c.lines ? { lines: true } : {}) } },
    });
    const answer = r.answers.d as DecideAnswer;
    rows.push({ ...c, decision: answer.decision, correct: answer.decision === c.expected, answer, backend: r.backend, ms: Math.round(performance.now() - started) });
    log("INFO", `decide case=${c.name} expected=${c.expected} decision=${answer.decision} p=${answer.probabilities[answer.decision].toFixed(3)} evidence=${answer.evidence.length}`);
  }
} finally {
  await provider.dispose();
}
const correct = rows.filter((r) => r.correct).length;
mkdirSync(join(ROOT, "reports/decide"), { recursive: true });
const runtime = process.versions.bun === undefined ? `node ${process.version}` : `bun ${process.versions.bun}`;
writeFileSync(
  join(ROOT, `reports/decide/node-${backend}${precision === "fp32" ? "" : `-${precision}`}.json`),
  JSON.stringify({ runtime, platform: `${process.platform}/${process.arch}`, backend, precision, correct, of: rows.length, cases: rows }, null, 1) + "\n",
);
log("INFO", `decide correct=${correct} of=${rows.length} backend=${backend} precision=${precision}`);
