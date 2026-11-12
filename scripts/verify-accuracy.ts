// Accuracy gate for a quantized task bundle whose answers may differ from FP32's; usage:
//   node scripts/verify-accuracy.ts models/vela3-fast-onnx/ner/q4 [webgpu|cpu] [--cases input/eval]
// (local, needs the bundle, its FP32 bundle and the sets of `uv run python python/eval_sets.py`).
//
// Runs the FP32 bundle and the quantized one over a public labelled set and compares what a caller gets:
//   ner  entity F1 on 3,000 Few-NERD sentences (a span counts when start, end and label equal a labelled one);
//        passes with at most 1 point of F1 below FP32
//   box  share of 1,500 RefCOCO expressions whose point lies in the labelled region; the region itself is not
//        compared; passes with at most 2 points below FP32
// How often the answers differ from FP32's is recorded, not gated. On a pass the script writes `validated: true` and
// the figures into the manifest, and replaces the bundle's fixtures.json by the bundle's own answers on the FP32
// fixture inputs (ONNX Runtime CPU), so scripts/verify-quantized.ts and the browser check hold every other runtime to
// them. The report goes to reports/quantization-accuracy.json either way.
//
// WHY not "no fixture changes its answer", scripts/verify-quantized.ts's rule for the other task bundles: an answer
// next to a threshold changes under any change of the weights. VERIFIED: (2026-11-12, Node.js, WebGPU) ner q4 failed
// that rule on one fixture span with an FP32 score of 0.63 (threshold 0.6), yet has an F1 of 0.560 against 0.557 for
// FP32; of the 715 spans it drops, 671 have an FP32 score below 0.7. vision-box q4 failed it on a point that moved
// to another cell inside the same region, and points into the labelled region for 70.3 % of the expressions against
// 70.8 %. The limits were set after these measurements.
import { createHash } from "node:crypto";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { loadVela } from "../src/node.ts";
import type { TaskFixture } from "../src/task-verify.ts";
import type { BoxAnswer, ImageInput, SpanAnswer } from "../src/types.ts";
import { loadPending } from "./load-pending.ts";

type Entity = { start: number; end: number; label: string };
type Region = BoxAnswer["box"];
type Point = BoxAnswer["point"];

/** The gated figure per task and how far it may fall below FP32's. */
export const RULES = {
  ner: { metric: "f1", maxDrop: 0.01 },
  box: { metric: "pointInRegion", maxDrop: 0.02 },
} as const;
export type GatedTask = keyof typeof RULES;

const key = (s: Entity) => `${s.start}:${s.end}:${s.label}`;
const round = (x: number) => Math.round(x * 1e4) / 1e4;

/** Entity precision, recall and F1 over all sentences: a predicted span is correct when its start, end and label equal a labelled one. */
export function entityScores(gold: Entity[][], predicted: Entity[][]) {
  let correct = 0,
    spans = 0,
    labelled = 0;
  gold.forEach((g, i) => {
    const want = new Set(g.map(key));
    labelled += want.size;
    spans += predicted[i].length;
    correct += predicted[i].filter((s) => want.has(key(s))).length;
  });
  const precision = spans ? correct / spans : 0,
    recall = labelled ? correct / labelled : 0;
  return { sentences: gold.length, labelled, spans, correct, precision: round(precision), recall: round(recall), f1: round(precision + recall ? (2 * precision * recall) / (precision + recall) : 0) };
}

/** How the quantized entity answers differ from FP32's: sentences with the same spans, and the spans only one of them has. */
export function entityAgreement(fp32: Entity[][], quantized: Entity[][]) {
  let sameSentences = 0,
    onlyFp32 = 0,
    onlyQuantized = 0;
  fp32.forEach((a, i) => {
    const x = new Set(a.map(key)),
      y = new Set(quantized[i].map(key));
    const lost = [...x].filter((k) => !y.has(k)).length,
      added = [...y].filter((k) => !x.has(k)).length;
    onlyFp32 += lost;
    onlyQuantized += added;
    if (!lost && !added) sameSentences++;
  });
  return { sameSentences, spansOnlyFp32: onlyFp32, spansOnlyQuantized: onlyQuantized };
}

/** Whether a point lies in a region, borders included. */
export const inside = (p: Point, r: Region) => r.left <= p.x && p.x <= r.right && r.top <= p.y && p.y <= r.bottom;

/** The share of points that lie in their labelled region. */
export function pointScores(regions: Region[], points: Point[]) {
  const hits = points.filter((p, i) => inside(p, regions[i])).length;
  return { expressions: regions.length, hits, pointInRegion: round(regions.length ? hits / regions.length : 0) };
}

/** The rule: the task's gated figure of the quantized bundle is at most `maxDrop` below FP32's. */
export function passes(task: GatedTask, fp32: Record<string, number>, quantized: Record<string, number>): boolean {
  const { metric, maxDrop } = RULES[task];
  return quantized[metric] >= fp32[metric] - maxDrop;
}

/** `flat` in rows as long as those of `like` (a fixture's logits are words by labels, or one row of readouts). */
export function shaped(flat: number[], like: number[][] | number[]): number[][] | number[] {
  if (!Array.isArray(like[0])) return flat;
  const width = (like[0] as number[]).length;
  return Array.from({ length: flat.length / width }, (_, i) => flat.slice(i * width, (i + 1) * width));
}

if (import.meta.main) {
  const ROOT = fileURLToPath(new URL("..", import.meta.url));
  const args = process.argv.slice(2);
  const at = args.indexOf("--cases");
  const casesRoot = resolve(at < 0 ? join(ROOT, "input/eval") : args.splice(at, 2)[1]);
  if (!args[0]) throw new Error("usage: node scripts/verify-accuracy.ts <quantized bundle folder> [webgpu|cpu] [--cases <folder>]");
  const dir = resolve(args[0]),
    backend = (args[1] ?? "webgpu") as "webgpu" | "cpu";
  const log = (level: string, msg: string) => console.error(`${new Date().toISOString()} ${level} ${msg}`);
  const manifestPath = join(dir, "manifest.json");
  const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
  const task = manifest.task as GatedTask;
  if (manifest.precision === "fp32" || !(task in RULES) || !manifest.quantizedFrom) throw new Error(`${dir}: this gate covers quantized ner and box bundles`);
  const fp32Dir = resolve(dir, manifest.quantizedFrom);
  const reference = JSON.parse(readFileSync(join(fp32Dir, "manifest.json"), "utf8"));
  if (reference.precision !== "fp32" || !reference.validated || reference.task !== task) throw new Error(`${fp32Dir} is not the validated FP32 bundle of ${task}`);
  const casesFile = join(casesRoot, task, "cases.json");
  if (!existsSync(casesFile)) throw new Error(`${casesFile} is missing; create it with: uv run python python/eval_sets.py`);
  const text = readFileSync(casesFile, "utf8");
  const data = JSON.parse(text) as { set: Record<string, string>; labels?: string[]; cases: { text?: string; gold?: Entity[]; image?: string; expression?: string; box?: Region }[] };

  /** Every case's answer from one bundle, with the backend it ran on. One engine at a time: an FP32 bundle holds 1.7 GB. */
  const answers = async (bundle: string, name: string) => {
    const engine = bundle === dir ? await loadPending(dir, backend) : await loadVela(bundle, { backend });
    const started = performance.now(),
      out: (SpanAnswer | BoxAnswer)[] = [];
    let used = "";
    try {
      for (const [i, c] of data.cases.entries()) {
        const r =
          task === "ner"
            ? await engine.systemOne({ state: c.text, questions: { q: { type: "span", criteria: data.labels! } } })
            : await engine.systemOne({ state: "", images: [join(dirname(casesFile), c.image!)], questions: { q: { type: "box", instructions: c.expression! } } });
        used = r.backend;
        out.push(r.answers.q as SpanAnswer | BoxAnswer);
        if (i % 500 === 499) log("INFO", `bundle=${name} done=${i + 1} seconds=${((performance.now() - started) / 1000).toFixed(0)}`);
      }
    } finally {
      await engine.dispose();
    }
    return { out, backend: used };
  };
  const a = await answers(fp32Dir, "fp32"),
    b = await answers(dir, manifest.precision);
  let fp32: Record<string, number>, quantized: Record<string, number>, agreement: Record<string, number>;
  if (task === "ner") {
    const gold = data.cases.map((c) => c.gold!),
      x = a.out.map((r) => (r as SpanAnswer).spans),
      y = b.out.map((r) => (r as SpanAnswer).spans);
    [fp32, quantized, agreement] = [entityScores(gold, x), entityScores(gold, y), entityAgreement(x, y)];
  } else {
    const regions = data.cases.map((c) => c.box!),
      x = a.out as BoxAnswer[],
      y = b.out as BoxAnswer[];
    [fp32, quantized] = [pointScores(regions, x.map((r) => r.point)), pointScores(regions, y.map((r) => r.point))];
    agreement = { samePoint: y.filter((r, i) => r.point.x === x[i].point.x && r.point.y === x[i].point.y).length, pointInFp32Region: y.filter((r, i) => inside(r.point, x[i].box)).length };
  }
  const id = relative(join(ROOT, "models/vela3-fast-onnx"), dir).replaceAll("\\", "/");
  const entry = {
    date: new Date().toISOString().slice(0, 10),
    precision: manifest.precision as string,
    runtime: `Node ${process.version}, ${b.backend}`,
    set: { ...data.set, cases: data.cases.length, sha256: createHash("sha256").update(text).digest("hex") },
    rule: RULES[task],
    fp32,
    quantized,
    agreement,
    passed: a.backend === b.backend && passes(task, fp32, quantized),
  };
  const reportPath = join(ROOT, "reports/quantization-accuracy.json");
  const report = existsSync(reportPath) ? JSON.parse(readFileSync(reportPath, "utf8")) : { tool: "scripts/verify-accuracy.ts", bundles: {} };
  report.bundles[id] = entry;
  writeFileSync(reportPath, JSON.stringify(report, null, 1) + "\n");
  log(entry.passed ? "INFO" : "ERROR", `bundle=${id} passed=${entry.passed} ${JSON.stringify({ fp32, quantized, agreement })}`);
  if (!entry.passed) process.exit(1);

  // The bundle's own answers on the FP32 fixture inputs, from the CPU: the reference every other runtime is held to.
  const fixtures: TaskFixture[] = JSON.parse(readFileSync(join(fp32Dir, "fixtures.json"), "utf8"));
  const engine = await loadPending(dir, "cpu");
  const own: TaskFixture[] = [];
  let differing = 0;
  try {
    for (const f of fixtures) {
      const r = await engine.infer({ state: f.state, question: f.question, images: (f.images ?? []).map((file) => join(dir, file)) as ImageInput[] });
      const got = r.answer as SpanAnswer | BoxAnswer;
      const decoded = (x: SpanAnswer | BoxAnswer) => JSON.stringify(x.type === "span" ? x.spans.map(({ start, end, label }) => ({ start, end, label })) : [x.box, x.point]);
      if (decoded(got) !== decoded(f.answer)) differing++;
      own.push({ ...f, logits: shaped(Array.from(r.logits), f.logits), answer: got });
    }
  } finally {
    await engine.dispose();
  }
  writeFileSync(join(dir, "fixtures.json"), JSON.stringify(own, null, 1) + "\n");
  const { pendingVerification: _done, rejected: _old, ...rest } = manifest;
  const validation = {
    tool: report.tool,
    date: entry.date,
    rule: `${RULES[task].metric} on the labelled set at most ${RULES[task].maxDrop} below FP32`,
    set: entry.set,
    fp32,
    quantized,
    agreement,
    fixtures: `the bundle's own answers on the FP32 fixture inputs (ONNX Runtime CPU); ${differing} of ${fixtures.length} differ from FP32's`,
  };
  writeFileSync(manifestPath, JSON.stringify({ ...rest, validated: true, validation }, null, 1) + "\n");
  log("INFO", `validated bundle=${id} fixtures=${fixtures.length} differing_from_fp32=${differing}`);
}
