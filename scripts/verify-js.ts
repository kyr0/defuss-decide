// Real-model parity of one v0.1 decision bundle in Node.js or Bun:
//   node|bun scripts/verify-js.ts models/decision/fast [cpu|webgpu]
// Token ids must equal the Python renderer's and probabilities stay within tolerance of native FP32 on the given
// backend (default cpu); writes parity-<runtime>.json, or parity-<runtime>-webgpu.json, into the bundle. Needs an
// exported bundle (documentation/decisions.md).
// VERIFIED: documentation/VALIDATION.md records 5 text and 4 vision cases passing in Node.js and Bun.
import { readFile, writeFile } from "node:fs/promises";
import assert from "node:assert/strict";
import { resolve } from "node:path";
import { loadVela } from "../src/node.ts";
import type { VelaDecision } from "../src/engine.ts";
import { formatFast, formatVision, probabilities } from "../src/format.ts";
import { preprocessImage } from "../src/image.ts";
import type { DecisionQuestion } from "../src/types.ts";

/** One recorded Python reference case of a decision bundle (`fixtures.json`). */
interface DecisionFixture {
  name: string;
  state?: unknown;
  question: DecisionQuestion;
  images?: string[];
  ids: number[];
  spans?: [number, number][];
  pool?: [number, number];
  logits: number[];
}

const folder = resolve(process.argv[2] ?? "models/decision/fast"),
  backend = (process.argv[3] ?? "cpu") as "cpu" | "webgpu";
const fixtures: DecisionFixture[] = JSON.parse(await readFile(resolve(folder, "fixtures.json"), "utf8"));
// A decision bundle (no `task` in its manifest) loads as a VelaDecision.
const engine = (await loadVela(folder, { backend })) as VelaDecision;
const results: Record<string, unknown>[] = [];
try {
  for (const f of fixtures) {
    const images = await Promise.all((f.images ?? []).map((name) => engine.decodeImage(resolve(folder, name))));
    const grids = images.map((rgb) => preprocessImage(rgb).grid);
    const rec =
      engine.manifest.kind === "fast"
        ? formatFast(engine.tokenizer, engine.manifest, f.state, f.question)
        : formatVision(engine.tokenizer, engine.manifest, f.state, f.question, grids);
    assert.deepEqual(rec.ids, f.ids, `${f.name}: token IDs`);
    if (f.spans) assert.deepEqual("spans" in rec ? rec.spans : undefined, f.spans);
    if (f.pool) assert.deepEqual("pool" in rec ? rec.pool : undefined, f.pool);
    const start = performance.now(),
      result = await engine.systemOne({ state: f.state, questions: { q: f.question }, images });
    const a = result.answers.q;
    const actual = a.type === "noul" ? [1 - a.noul, a.noul] : a.type === "choice" ? Object.values(a.probabilities) : [];
    const expected = probabilities(f.logits),
      error = Math.max(...actual.map((p, i) => Math.abs(p - expected[i])));
    assert.ok(error <= (engine.manifest.precision === "fp32" ? 0.001 : 0.02), `${f.name}: probability error ${error}`);
    assert.equal(actual.indexOf(Math.max(...actual)), expected.indexOf(Math.max(...expected)), `${f.name}: top choice`);
    results.push({ case: f.name, tokenIdsExact: true, maxProbabilityError: error, ms: performance.now() - start, answer: a });
    console.log(results.at(-1));
  }
} finally {
  await engine.dispose();
}
const bun = process.versions.bun;
const report = {
  passed: true,
  runtime: bun === undefined ? `node ${process.version}` : `bun ${bun}`,
  precision: engine.manifest.precision,
  backend: engine.backend,
  model: engine.manifest.model,
  results,
};
await writeFile(
  resolve(folder, `parity-${bun === undefined ? "node" : "bun"}${backend === "cpu" ? "" : "-" + backend}.json`),
  JSON.stringify(report, null, 2) + "\n",
);
