// Runs in real Chromium and WebKit (Vitest browser mode): the browser entry point, ONNX Runtime Web (WASM, and WebGPU
// where the browser grants an adapter), fetch-based bundle loading with external data, and the browser image decoder,
// on the committed fixture bundles.
import { describe, expect, test } from "vitest";
import { decodeBrowserImage, downloadSize, loadVela, loadVelaProvider } from "../../src/web.ts";
import type { WebOptions } from "../../src/web.ts";
import type { ProgressEvent } from "../../src/types.ts";
import { sameAnswer } from "../shared/compare.ts";

const BASE = "/tests/fixtures/bundles/";
const json = (path: string) => fetch(BASE + path).then((r) => r.json());
const reference = await json("expected.json");
const names = Object.keys(reference.bundles);

async function sha256(bytes: Uint8Array) {
  const digest = await crypto.subtle.digest("SHA-256", bytes as Uint8Array<ArrayBuffer>);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
}

const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
const adapter = await gpu?.requestAdapter().catch(() => null);

describe("every fixture bundle answers like Node.js and Bun", () => {
  // `auto` runs on WebGPU where this browser grants an adapter and falls back to WASM where it does not.
  test.each(names.flatMap((name) => [[name, "wasm"] as const, [name, "auto"] as const]))("%s on %s", async (name, backend) => {
    const engine = await loadVela(BASE + name, { backend, threads: 1 });
    expect(engine.backend).toBe(backend === "wasm" || !adapter ? "wasm" : "webgpu");
    try {
      const fixtures = await json(name + "/fixtures.json");
      for (const [i, f] of fixtures.entries()) {
        const r = await engine.systemOne({
          state: f.state,
          questions: { q: f.question },
          images: (f.images ?? []).map((file: string) => BASE + name + "/" + file),
        });
        expect(sameAnswer(r.answers.q, reference.bundles[name][i].answer), JSON.stringify(r.answers.q)).toBe(true);
        expect(r.usage.input_tokens).toBe(reference.bundles[name][i].inputTokens);
        expect(r.backend).toBe(engine.backend);
      }
    } finally {
      await engine.dispose();
    }
  });
});

describe("browser image decoding", () => {
  test.each(Object.keys(reference.images))("%s decodes to the same RGB bytes as sharp", async (key) => {
    const blob = await fetch(BASE + key).then((r) => r.blob());
    const rgb = await decodeBrowserImage(blob);
    expect([rgb.width, rgb.height]).toEqual([reference.images[key].width, reference.images[key].height]);
    expect(await sha256(rgb.data)).toBe(reference.images[key].sha256);
  });
  test("a failing image URL is an error", async () => {
    // Vite answers unknown paths with its HTML fallback, so the decoder sees HTML instead of an image.
    await expect(decodeBrowserImage(BASE + "missing.png")).rejects.toThrow();
  });
});

describe("explicit runtime options", () => {
  test("unknown backends and unisolated threads are refused instead of silently changed", async () => {
    await expect(loadVela(BASE + "ner", { backend: "cpu" as never })).rejects.toThrow(TypeError);
    if (!globalThis.crossOriginIsolated)
      await expect(loadVela(BASE + "ner", { threads: 2 })).rejects.toThrow(/COOP\/COEP/);
    await expect(loadVela(BASE + "missing")).rejects.toThrow(/manifest\.json: (HTTP 404|got an HTML page)/);
  });
  test("a pinned WebGPU backend fails where the browser has no WebGPU instead of falling back", async () => {
    if (gpu) return;
    await expect(loadVela(BASE + "ner", { backend: "webgpu" })).rejects.toThrow(/WebGPU unavailable/);
  });
});

describe("bundle files: progress, Brotli copies and the cache", () => {
  type Download = Extract<ProgressEvent, { stage: "download" }>;
  /** Load `ner` from `base`, run its first fixture, and return the download events and the answer. */
  async function run(base: string, options: WebOptions) {
    const events: Download[] = [];
    const engine = await loadVela(base, { backend: "wasm", threads: 1, ...options, onProgress: (e) => e.stage === "download" && events.push(e) });
    try {
      const [f] = await json("ner/fixtures.json");
      return { events, answer: (await engine.systemOne({ state: f.state, questions: { q: f.question } })).answers.q };
    } finally {
      await engine.dispose();
    }
  }
  /** The last event per file: what each read ended with. */
  const last = (events: Download[]) => [...new Map(events.map((e) => [e.file.split("/").slice(-2).join("/"), e])).entries()].sort();
  const files = ["bundles/tokenizer-fast.json", "bundles/tokenizer_config-fast.json", "ner/manifest.json", "ner/task.onnx", "ner/task.onnx.data"];

  test("every file read reports its bytes, and the answer stays the same", async () => {
    const { events, answer } = await run(BASE + "ner", {});
    expect(last(events).map(([file]) => file)).toEqual(files);
    for (const [, e] of last(events)) expect(e).toMatchObject({ cached: false, loaded: e.total });
    expect(sameAnswer(answer, reference.bundles.ner[0].answer)).toBe(true);
  });
  test("a second load with the same cache reads every file from it", async () => {
    const name = `defuss-decide-test-${Math.random().toString(36).slice(2)}`;
    try {
      const first = await run(BASE + "ner", { cache: name });
      expect(last(first.events).every(([, e]) => !e.cached)).toBe(true);
      const second = await run(BASE + "ner", { cache: name });
      expect(second.events.length).toBe(files.length);
      for (const [, e] of last(second.events)) expect(e).toMatchObject({ cached: true, loaded: e.total });
      expect(second.answer).toEqual(first.answer);
    } finally {
      await caches.delete(name);
    }
  });
  test("with brotli, a folder without .br copies loads the plain files", async () => {
    // A .br request answers 404 on a plain server and the app's HTML page on Vite's dev server; both fall back.
    const { events, answer } = await run(BASE + "ner", { brotli: true });
    expect(last(events).map(([file]) => file)).toEqual(files);
    expect(sameAnswer(answer, reference.bundles.ner[0].answer)).toBe(true);
  });
  test("downloadSize names the bytes a load fetches before it runs, and nothing once the cache holds them", async () => {
    const { events } = await run(BASE + "ner", {});
    expect(await downloadSize([BASE + "ner"])).toBe(last(events).reduce((sum, [, e]) => sum + (e.total ?? 0), 0));
    const name = `defuss-decide-test-${Math.random().toString(36).slice(2)}`;
    try {
      await run(BASE + "ner", { cache: name });
      expect(await downloadSize([BASE + "ner"], { cache: name })).toBe(0);
    } finally {
      await caches.delete(name);
    }
  });
  test("with brotli, the .br copies load; a server that labels them Content-Encoding: br hands over decoded bytes", async () => {
    // Vite's dev server sends .br files with `Content-Encoding: br` (VERIFIED: curl -I, 2026-11-09), so the browser decodes them.
    const { events, answer } = await run("/tmp/browser-br/ner/", { brotli: true });
    expect(last(events).map(([file]) => file)).toEqual(files.map((f) => f.replace("bundles/", "browser-br/")));
    expect(sameAnswer(answer, reference.bundles.ner[0].answer)).toBe(true);
  });
});

describe("provider in the browser", () => {
  test("routes to the right bundles and releases them", async () => {
    const provider = await loadVelaProvider(
      { fast: { ner: BASE + "ner", span_set: BASE + "span_set/" }, vision: { box: BASE + "vision-box" } },
      { backend: "wasm" },
    );
    try {
      const r = await provider.systemOne({
        state: "Contact Ada at ada@example.org.",
        questions: { who: { type: "span", criteria: ["person"] }, pii: { type: "span", criteria: ["EMAIL_ADDRESS"] } },
      });
      expect(r.answers.who.type).toBe("span");
      expect(r.answers.pii.type).toBe("span");
      const box = await provider.systemOne({
        questions: { where: { type: "box", instructions: "the red square" } },
        images: [BASE + "vision-box/sample.png"],
      });
      expect(sameAnswer(box.answers.where, reference.bundles["vision-box"][0].answer)).toBe(true);
    } finally {
      await provider.dispose();
    }
  });
  test("a text bundle whose manifest is missing fails at creation", async () => {
    await expect(loadVelaProvider({ fast: { ner: BASE + "missing" } })).rejects.toThrow(/manifest\.json: (HTTP 404|got an HTML page)/);
  });
});
