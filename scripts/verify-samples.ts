// Real-model run of the website's sample images: `node scripts/verify-samples.ts [cpu|webgpu|auto] [fp32|q8]` (local,
// needs models/vela3-fast-onnx/<task>/<precision> and models/decision/{fast,vision}, FP32 only). The page shows the
// FP32 record; another precision writes reports/samples/node-<backend>-<precision>.json for comparison.
//
// Runs every question of tests/fixtures/samples.json on the real FP32 bundles and writes reports/samples/node-<backend>.json
// with each answer, whether it matches what a person reads off the image (`expect`), the backend and the time.
// scripts/docs.ts shows these recorded answers on the website. WHY recorded and checked: the samples are new images
// without a Python reference, so the page may only show answers that were produced by the real bundles, and a sample
// that stops answering as expected must show up here, not on the page. Exits 1 only when a run fails, not for a miss.
// VERIFIED: 18 of 18 runs matched on 2026-11-08 (reports/samples/node-auto.json); the first draft of the questions
// matched 14 of 20, and its misses were replaced or, for OCR mismatches in decisions, documented in decide.md.
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { loadVela, loadVelaProvider } from "../src/node.ts";
import type { Answer, BoxAnswer, DecideAnswer, Question, SpanAnswer } from "../src/types.ts";

/** One question about a sample image, with what a person reads off the image. */
export interface SampleRun {
  task: string;
  state: unknown;
  question: Question & Record<string, unknown>;
  expect: { inside?: { left: number; right: number; top: number; bottom: number }; flag?: string[]; decision?: string };
}
export interface Sample {
  id: string;
  file: string;
  title: string;
  alt: string;
  runs: SampleRun[];
}

/**
 * Whether an answer matches its expectation: a box's point lies inside `inside`; a span answer flags every `flag` text
 * and nothing else (a span and a flag match when one contains the other); a decision equals `decision`.
 */
export function matches(answer: Answer, expect: SampleRun["expect"]): boolean {
  if (answer.type === "box" && expect.inside) {
    const { x, y } = (answer as BoxAnswer).point,
      r = expect.inside;
    return x >= r.left && x <= r.right && y >= r.top && y <= r.bottom;
  }
  if (answer.type === "span" && expect.flag) {
    const hit = (a: string, b: string) => a.includes(b) || b.includes(a);
    const spans = (answer as SpanAnswer).spans.map((s) => s.text);
    return expect.flag.every((f) => spans.some((s) => hit(s, f))) && spans.every((s) => expect.flag!.some((f) => hit(s, f)));
  }
  if (answer.type === "decide" && expect.decision) return (answer as DecideAnswer).decision === expect.decision;
  return false;
}

if (import.meta.main) {
  const ROOT = fileURLToPath(new URL("..", import.meta.url));
  const backend = (process.argv[2] ?? "auto") as "auto" | "cpu" | "webgpu";
  const log = (level: string, msg: string) => console.log(`${new Date().toISOString()} ${level} ${msg}`);
  const { samples } = JSON.parse(readFileSync(join(ROOT, "tests/fixtures/samples.json"), "utf8")) as { samples: Sample[] };
  const image = (s: Sample) => join(ROOT, "docs/assets/samples/synthetic", s.file);
  const precision = process.argv[3] ?? "fp32";
  const bundle = (task: string) => join(ROOT, "models/vela3-fast-onnx", task, precision);
  const rows: { sample: string; task: string; correct: boolean; answer: Answer; backend: string; ms: number }[] = [];
  const record = (s: Sample, run: SampleRun, answer: Answer, used: string, started: number) => {
    const correct = matches(answer, run.expect);
    rows.push({ sample: s.id, task: run.task, correct, answer, backend: used, ms: Math.round(performance.now() - started) });
    log(correct ? "INFO" : "WARN", `sample=${s.id} task=${run.task} correct=${correct} answer=${JSON.stringify(answer).slice(0, 160)}`);
  };
  const tasks = [...new Set(samples.flatMap((s) => s.runs.map((r) => r.task)))];
  for (const task of tasks) {
    const todo = samples.flatMap((s) => s.runs.filter((r) => r.task === task).map((r) => [s, r] as const));
    if (task === "decide") {
      const provider = await loadVelaProvider(
        {
          fast: { decision: join(ROOT, "models/decision/fast"), evidence: bundle("evidence") },
          vision: { decision: join(ROOT, "models/decision/vision"), box: bundle("vision-box"), ocr: bundle("vision-ocr") },
        },
        { backend },
      );
      try {
        for (const [s, run] of todo) {
          const started = performance.now();
          const r = await provider.systemOne({ state: run.state, images: [image(s)], questions: { d: run.question } });
          record(s, run, r.answers.d, r.backend, started);
        }
      } finally {
        await provider.dispose();
      }
      continue;
    }
    const engine = await loadVela(bundle(task), { backend });
    try {
      for (const [s, run] of todo) {
        const started = performance.now();
        const r = await engine.systemOne({ state: run.state, images: [image(s)], questions: { q: run.question } });
        record(s, run, r.answers.q, r.backend, started);
      }
    } finally {
      await engine.dispose();
    }
  }
  const correct = rows.filter((r) => r.correct).length;
  mkdirSync(join(ROOT, "reports/samples"), { recursive: true });
  const runtime = process.versions.bun === undefined ? `node ${process.version}` : `bun ${process.versions.bun}`;
  writeFileSync(
    join(ROOT, `reports/samples/node-${backend}${precision === "fp32" ? "" : `-${precision}`}.json`),
    JSON.stringify({ runtime, platform: `${process.platform}/${process.arch}`, backend, precision, correct, of: rows.length, runs: rows }, null, 1) + "\n",
  );
  log("INFO", `samples correct=${correct} of=${rows.length} backend=${backend} precision=${precision}`);
}
