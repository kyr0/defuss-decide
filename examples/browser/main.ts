// v0.1 decision demo page (index.html, `bun run dev`): load a decision bundle, ask choice and yes/no questions, and
// run its exported parity fixtures in the browser. scripts/browser-test.ts drives it through window.velaTest.
import { loadVela } from "../../src/web.ts";
import type { VelaDecision } from "../../src/engine.ts";
import { probabilities, formatFast, formatVision } from "../../src/format.ts";
import { preprocessImage } from "../../src/image.ts";
import type { DecisionQuestion } from "../../src/types.ts";

/** One recorded Python reference case of a decision bundle (`fixtures.json`). */
interface DecisionFixture {
  name: string;
  state?: unknown;
  question: DecisionQuestion;
  images?: string[];
  ids: number[];
  logits: number[];
}

declare global {
  interface Window {
    velaTest: { load(): Promise<unknown>; parity(): Promise<Record<string, unknown>>; setBundle(path: string): string };
  }
}

const $ = <T extends HTMLElement = HTMLButtonElement>(id: string) => document.getElementById(id) as T;
const input = (id: string) => $<HTMLInputElement>(id);
let engine: VelaDecision | null = null;
function show(x: unknown) {
  $("result").textContent = JSON.stringify(x, null, 2);
}
// Release the previous engine first, so only one bundle's sessions are resident (ARCH.md: one resident bundle).
async function load() {
  if (engine) await engine.dispose();
  engine = null;
  $("run").disabled = $("parity").disabled = true;
  // Decision bundles (no `task` in their manifest) load as VelaDecision; this page serves only those.
  engine = (await loadVela(input("bundle").value, {
    backend: input("backend").value as "auto" | "wasm" | "webgpu",
    threads: Number(input("threads").value),
    onProgress: (p) => ($("status").textContent = JSON.stringify(p)),
  })) as VelaDecision;
  $("status").textContent = `Loaded ${engine.manifest.model} · ${engine.manifest.precision} · ${engine.backend}`;
  $("run").disabled = $("parity").disabled = $("unload").disabled = false;
  return engine.manifest;
}
// VERIFIED: the exported fixtures through the public API, with the checks of scripts/verify-js.ts: exact token ids, probabilities
// within 0.001 (FP32) or 0.02, and the same top option.
async function parity() {
  const e = engine!;
  const base = new URL(input("bundle").value, location.href),
    fixtures: DecisionFixture[] = await fetch(new URL("fixtures.json", base)).then((r) => r.json()),
    results: Record<string, unknown>[] = [];
  for (const f of fixtures) {
    const images = await Promise.all((f.images ?? []).map((name) => e.decodeImage(new URL(name, base))));
    const grids = images.map((rgb) => preprocessImage(rgb).grid);
    const rec = e.manifest.kind === "fast" ? formatFast(e.tokenizer, e.manifest, f.state, f.question) : formatVision(e.tokenizer, e.manifest, f.state, f.question, grids);
    if (JSON.stringify(rec.ids) !== JSON.stringify(f.ids)) throw new Error(`${f.name}: token mismatch`);
    const start = performance.now(),
      out = await e.systemOne({ state: f.state, questions: { q: f.question }, images });
    const a = out.answers.q,
      p = a.type === "noul" ? [1 - a.noul, a.noul] : a.type === "choice" ? Object.values(a.probabilities) : [],
      ref = probabilities(f.logits);
    const err = Math.max(...p.map((x, i) => Math.abs(x - ref[i])));
    if (err > (e.manifest.precision === "fp32" ? 0.001 : 0.02)) throw new Error(`${f.name}: probability error ${err}`);
    if (p.indexOf(Math.max(...p)) !== ref.indexOf(Math.max(...ref))) throw new Error(`${f.name}: argmax mismatch`);
    results.push({ case: f.name, tokenIdsExact: true, maxProbabilityError: err, ms: performance.now() - start });
    show(results);
  }
  return { passed: true, model: e.manifest.model, precision: e.manifest.precision, backend: e.backend, results };
}
// Wire a button: every control is disabled while it runs, then the result or the error is shown.
function action(id: string, fn: () => Promise<unknown>) {
  $(id).onclick = async () => {
    const controls = ["load", "run", "parity", "unload"];
    controls.forEach((k) => ($(k).disabled = true));
    try {
      show(await fn());
    } catch (e) {
      show({ error: (e as Error).message });
      $("status").textContent = "Failed; see error.";
    } finally {
      $("load").disabled = false;
      ["run", "parity", "unload"].forEach((k) => ($(k).disabled = !engine));
    }
  };
}
action("load", load);
action("run", () =>
  engine!.systemOne({ state: input("state").value, questions: JSON.parse(input("questions").value), images: [...input("images").files!] }),
);
action("parity", parity);
action("unload", async () => {
  await engine!.dispose();
  engine = null;
  $("status").textContent = "Model released.";
  return { released: true };
});
// Local Playwright uses the same public UI path and engine as an interactive run.
window.velaTest = { load, parity, setBundle: (path) => (input("bundle").value = path) };
