// Inference latency of the exported bundles in Node.js (`make bench`; local, needs models/):
//
//   node scripts/bench.ts [fp32|q8|q4] [--backend auto|webgpu|cpu] [--threads N]
//
// Per bundle: load time, then each fixture three times; the median is reported. Text bundles get one more row: the
// first fixture's text repeated to about 1,800 tokens, because the fixtures are 20 to 60 tokens long and hide how
// text latency grows with length (the inputs may be up to 2,048 tokens). Writes output/bench-<precision>-<backend>.json.
//
// WHY fixtures: they are the inputs whose outputs the parity gate already checked, so a faster but wrong change cannot
// pass unnoticed next to its timing. The long row has no reference output; it measures time only.
import { mkdirSync, readFileSync, readdirSync, existsSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { fileURLToPath } from "node:url";
import { loadVela } from "../src/node.ts";
import type { VelaTask } from "../src/task-engine.ts";
import type { TaskFixture } from "../src/task-verify.ts";
import type { SpanQuestion } from "../src/types.ts";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const BASE = join(ROOT, "models/vela3-fast-onnx");
const log = (level: string, msg: string) => console.log(`${new Date().toISOString()} ${level} ${msg}`);
if (!existsSync(BASE)) {
  log("ERROR", "UNKNOWN[bench] BC no exported bundles in models/vela3-fast-onnx");
  process.exit(2);
}
const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: { backend: { type: "string", default: "auto" }, threads: { type: "string" } },
});
const precision = positionals[0] ?? "fp32",
  backend = values.backend as "auto" | "webgpu" | "cpu",
  threads = values.threads === undefined ? undefined : Number(values.threads);

/** The fixture's state with its target text repeated until it reaches about `tokens` tokens. */
function longState(engine: VelaTask, state: unknown, tokens: number): unknown {
  const fields: Record<string, string> = typeof state === "string" ? { user: state } : { ...(state as Record<string, string>) };
  const key = "answer" in fields ? "answer" : Object.keys(fields).at(-1)!;
  let text = fields[key];
  while (engine.tokenizer.encode(`${text} ${fields[key]}`, { add_special_tokens: false }).ids.length < tokens) text += ` ${fields[key]}`;
  return { ...fields, [key]: text };
}

/** Median of three timed runs after one warm-up run. */
async function median(run: () => Promise<unknown>): Promise<number> {
  await run();
  const times: number[] = [];
  for (let i = 0; i < 3; i++) {
    const started = performance.now();
    await run();
    times.push(performance.now() - started);
  }
  return Math.round(times.sort((a, b) => a - b)[1]);
}

const rows: { bundle: string; fixture: string; backend: string; loadMs: number; medianMs: number }[] = [];
for (const task of readdirSync(BASE)) {
  const dir = join(BASE, task, precision);
  if (!existsSync(join(dir, "manifest.json"))) continue;
  const started = performance.now();
  // Task bundles only (models/vela3-fast-onnx holds no decision bundle).
  const engine = (await loadVela(dir, { backend, threads })) as VelaTask;
  const loadMs = Math.round(performance.now() - started);
  try {
    const fixtures = JSON.parse(readFileSync(join(dir, "fixtures.json"), "utf8")) as TaskFixture[];
    const cases = fixtures.map((f) => ({ name: f.name, state: f.state, question: f.question, images: (f.images ?? []).map((x) => join(dir, x)) }));
    if (engine.manifest.kind === "fast") {
      const f = fixtures[0];
      cases.push({ name: `${f.name}, about 1,800 tokens`, state: longState(engine, f.state, 1800), question: f.question, images: [] });
    }
    for (const c of cases) {
      const medianMs = await median(() => engine.infer({ state: c.state, question: c.question as SpanQuestion, images: c.images }));
      rows.push({ bundle: `${task}/${precision}`, fixture: c.name, backend: engine.backend, loadMs, medianMs });
      log("INFO", `bench bundle=${task}/${precision} fixture="${c.name}" backend=${engine.backend} load_ms=${loadMs} median_ms=${medianMs}`);
    }
  } finally {
    await engine.dispose();
  }
}
mkdirSync(join(ROOT, "output"), { recursive: true });
writeFileSync(
  join(ROOT, `output/bench-${precision}-${backend}.json`),
  JSON.stringify({ runtime: `node ${process.version}`, platform: `${process.platform}/${process.arch}`, precision, backend, threads: threads ?? "default", rows }, null, 2) + "\n",
);
