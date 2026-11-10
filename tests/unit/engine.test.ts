import { describe, expect, test } from "vitest";
import { join } from "node:path";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import sharp from "sharp";
import * as ort from "onnxruntime-node";
import { loadVela } from "../../src/node.ts";
import { VelaDecision, VelaTask } from "../../src/index.ts";
import type { Backend, ProgressEvent } from "../../src/types.ts";
import type { Fixture } from "./helpers.ts";
import { BUNDLES, TEXT_ADAPTERS, VISION_ADAPTERS, bundle, editedBundle, expected, fixtures, loadDecision, loadTask, sameAnswer } from "./helpers.ts";

const reference = expected();
const images = (name: string, f: Fixture): string[] => (f.images ?? []).map((file: string) => join(bundle(name), file));

describe("loading", () => {
  test("task bundles load as VelaTask, decision bundles as VelaDecision", async () => {
    const task = await loadVela(bundle("ner")),
      decision = await loadVela(bundle("fast-decision"));
    try {
      expect(task).toBeInstanceOf(VelaTask);
      expect(decision).toBeInstanceOf(VelaDecision);
    } finally {
      await task.dispose();
      await decision.dispose();
    }
  });
  test("a graph with a weight file shared from another folder answers like the same graph in one folder", () => {
    // The q4 bundles share one backbone file (python/quantize_gptq.py); fast-decision-split is fast-decision's graph
    // with two weights in tests/fixtures/bundles/backbone.onnx.data, which the loaders must find through the manifest.
    expect(reference.bundles["fast-decision-split"]).toEqual(reference.bundles["fast-decision"]);
  });
  test("a missing shared weight file fails the load and names the file", async () => {
    // editedBundle copies the bundle folder alone, so the shared ../backbone.onnx.data is not next to the copy.
    await expect(loadVela(editedBundle("fast-decision-split", () => {}))).rejects.toThrow(/backbone\.onnx\.data/);
  });
  test("an unvalidated or foreign bundle never reaches inference", async () => {
    await expect(loadVela(editedBundle("ner", (m) => (m.validated = false)))).rejects.toThrow(/parity validation/);
    await expect(loadVela(editedBundle("ner", (m) => (m.format = "other/1")))).rejects.toThrow(/parity validation/);
  });
  test("acceptUnvalidated loads a validated: false bundle, never a foreign or unchecked one", async () => {
    const engine = await loadVela(editedBundle("ner", (m) => (m.validated = false)), { backend: "cpu", acceptUnvalidated: true });
    await engine.dispose();
    await expect(loadVela(editedBundle("ner", (m) => (m.format = "other/1")), { acceptUnvalidated: true })).rejects.toThrow(/parity validation/);
    const pending = editedBundle("ner", (m) => {
      m.validated = false;
      m.pendingVerification = "node scripts/verify-quantized.ts models/vela3-fast-onnx/ner/q8";
    });
    await expect(loadVela(pending, { acceptUnvalidated: true })).rejects.toThrow(/not verified yet/);
  });
  test("a quantized variant stays unloadable until its decoded-output gate passed", async () => {
    // Regression: python/quantize_tasks.py used to write validated: true before any check ran.
    const pending = editedBundle("ner", (m) => {
      m.precision = "w8a32";
      m.validated = false;
      m.pendingVerification = "node scripts/verify-quantized.ts models/vela3-fast-onnx/ner/q8";
    });
    await expect(loadVela(pending)).rejects.toThrow(/not verified yet; run: node scripts\/verify-quantized\.ts/);
  });
  test("a graph that fails to load releases the session created before it", async () => {
    const dir = editedBundle("vision-span", (m) => (m.graphs.vision.file = "missing.onnx"));
    await expect(loadVela(dir)).rejects.toThrow();
    const decision = editedBundle("vision-decision", (m) => (m.graphs.vision.file = "missing.onnx"));
    await expect(loadVela(decision)).rejects.toThrow();
  });
  test("progress reports every graph a bundle loads and the backend it loads on", async () => {
    const events: unknown[] = [];
    const engine = await loadVela(bundle("vision-decision"), { backend: "cpu", onProgress: (e) => events.push(e) });
    await engine.dispose();
    expect(events).toEqual([
      { stage: "load", graph: "decision", backend: "cpu" },
      { stage: "load", graph: "vision", backend: "cpu" },
    ]);
  });
});

describe("backends", () => {
  /** Load a fixture bundle through the runtime-neutral loader with real ONNX Runtime and the given backend order. */
  const loadOn = (name: string, backends: string[], onProgress?: (e: ProgressEvent) => void) =>
    VelaDecision.load({
      ort,
      backends: backends as Backend[],
      onProgress,
      readJSON: async (file) => JSON.parse(readFileSync(join(bundle(name), file), "utf8")),
      createSession: (graph, backend) => ort.InferenceSession.create(join(bundle(name), graph.file), { executionProviders: [backend] }),
      decodeImage: async (source) => {
        const { data, info } = await sharp(source as string).removeAlpha().raw().toBuffer({ resolveWithObject: true });
        return { data, width: info.width, height: info.height };
      },
    });

  test("auto runs on WebGPU where this machine has it, else on the CPU, and every response names the backend", async () => {
    const engine = await loadTask(bundle("ner"));
    try {
      expect(["webgpu", "cpu"]).toContain(engine.backend);
      const f = fixtures("ner")[0];
      const r = await engine.systemOne({ state: f.state, questions: { q: f.question } });
      expect(r.backend).toBe(engine.backend);
      expect(sameAnswer(r.answers.q, reference.bundles.ner[0].answer)).toBe(true);
    } finally {
      await engine.dispose();
    }
  });
  test("a backend that cannot load the bundle falls back to the next one and says so", async () => {
    const events: ProgressEvent[] = [];
    const engine = await loadOn("vision-span", ["unavailable", "cpu"], (e) => events.push(e));
    try {
      expect(engine.backend).toBe("cpu");
      expect(events.filter((e) => e.stage === "fallback")).toEqual([
        { stage: "fallback", from: "unavailable", to: "cpu", error: expect.stringMatching(/backend not found/) },
      ]);
      const f = fixtures("vision-span")[0];
      const r = await engine.systemOne({ state: f.state, questions: { q: f.question }, images: [join(bundle("vision-span"), f.images[0])] });
      expect(r.backend).toBe("cpu");
    } finally {
      await engine.dispose();
    }
  });
  test("when no backend loads the bundle, the last error is thrown", async () => {
    await expect(loadOn("ner", ["unavailable"])).rejects.toThrow(/backend not found/);
    await expect(loadOn("ner", [])).rejects.toThrow(/No backend/);
  });
  test("a pinned backend is used as given; unknown names are refused", async () => {
    const engine = await loadVela(bundle("ner"), { backend: "cpu" });
    expect(engine).toMatchObject({ backend: "cpu" });
    await engine.dispose();
    await expect(loadVela(bundle("ner"), { backend: "gpu" as never })).rejects.toThrow(/backend must be/);
  });
});

describe("every fixture runs end to end and matches the cross-runtime reference", () => {
  const all = [...TEXT_ADAPTERS, ...VISION_ADAPTERS, "fast-decision", "fast-decision-split", "vision-decision"];
  test.each(all)("%s", async (name) => {
    const engine = await loadVela(bundle(name), { threads: 1 });
    const events: string[] = [];
    engine.onProgress = (e) => events.push(e.stage);
    try {
      for (const [i, f] of fixtures(name).entries()) {
        const r = await engine.systemOne({ state: f.state, questions: { q: f.question }, images: images(name, f) });
        const want = reference.bundles[name][i];
        expect(sameAnswer(r.answers.q, want.answer), JSON.stringify(r.answers.q)).toBe(true);
        expect(r.usage).toEqual({ input_tokens: want.inputTokens, output_tokens: 0 });
        expect(r.provider).toBe("vela3");
        expect(r.precision).toBe("fp32");
        if (engine instanceof VelaTask) expect(r.offsetUnit).toBe("unicode-code-point");
      }
      if ((fixtures(name)[0].images ?? []).length) expect(events).toContain("image");
    } finally {
      await engine.dispose();
    }
  });
  test("infer exposes the record the parity checks compare", async () => {
    const engine = await loadTask(bundle("ner"));
    try {
      const f = fixtures("ner")[0];
      const r = await engine.infer({ state: f.state, question: f.question });
      expect(r.record.ids).toEqual(f.ids);
      expect(r.logits).toHaveLength(f.logits.flat().length);
    } finally {
      await engine.dispose();
    }
  });
  test("decoded RGB bytes and sizes are what the browsers must reproduce", async () => {
    for (const [key, want] of Object.entries(reference.images)) {
      const { data, info } = await sharp(join(BUNDLES, key)).removeAlpha().raw().toBuffer({ resolveWithObject: true });
      expect([info.width, info.height]).toEqual([want.width, want.height]);
      expect(createHash("sha256").update(data).digest("hex")).toBe(want.sha256);
    }
  });
  test("encoded image bytes and pre-decoded RGB give the same answer as a path", async () => {
    const engine = await loadTask(bundle("vision-box"));
    try {
      const f = fixtures("vision-box")[0],
        path = images("vision-box", f)[0];
      const fromBytes = await engine.systemOne({ questions: { q: f.question }, images: [readFileSync(path)] });
      const rgb = await engine.decodeImage(path);
      const fromRGB = await engine.systemOne({ questions: { q: f.question }, images: [rgb] });
      expect(fromBytes.answers.q).toEqual(fromRGB.answers.q);
      expect(sameAnswer(fromRGB.answers.q, reference.bundles["vision-box"][0].answer)).toBe(true);
    } finally {
      await engine.dispose();
    }
  });
  test("a decision bundle answers several questions over several images with one encoder pass each", async () => {
    const engine = await loadDecision(bundle("vision-decision"));
    const tiles: number[] = [];
    engine.onProgress = (e) => e.stage === "image" && tiles.push(e.tile);
    try {
      const red = join(bundle("vision-decision"), "red.png");
      const r = await engine.systemOne({
        state: "A picture.",
        questions: { a: { type: "noul", instructions: "Is it red?" }, b: { type: "choice", criteria: { red: null, blue: "A blue color" } } },
        images: [red, red],
      });
      expect(Object.keys(r.answers)).toEqual(["a", "b"]);
      expect(r.answers.b.type).toBe("choice");
      // Both images are encoded once for the whole request, not once per question.
      expect(tiles).toEqual(tiles.map((_, i) => i + 1));
      expect(tiles.length % 2).toBe(0);
    } finally {
      await engine.dispose();
    }
  });
  test("a task bundle encodes the request's image once for all its questions", async () => {
    const engine = await loadTask(bundle("vision-span"));
    let tiles = 0;
    engine.onProgress = (e) => void (e.stage === "image" && tiles++);
    try {
      const f = fixtures("vision-span")[0],
        image = images("vision-span", f);
      await engine.systemOne({ state: f.state, questions: { a: f.question }, images: image });
      const once = tiles;
      tiles = 0;
      const r = await engine.systemOne({ state: f.state, questions: { a: f.question, b: f.question }, images: image });
      expect(once).toBeGreaterThan(0);
      expect(tiles).toBe(once);
      expect(r.answers.b).toEqual(r.answers.a);
      expect(sameAnswer(r.answers.a, reference.bundles["vision-span"][0].answer)).toBe(true);
    } finally {
      await engine.dispose();
    }
  });
  test("a task bundle checks every question before it runs the image encoder", async () => {
    const engine = await loadTask(bundle("vision-span"));
    let tiles = 0;
    engine.onProgress = (e) => void (e.stage === "image" && tiles++);
    try {
      const f = fixtures("vision-span")[0];
      const bad = { a: f.question, b: { type: "choice", criteria: { x: null } } } as never;
      await expect(engine.systemOne({ state: f.state, questions: bad, images: images("vision-span", f) })).rejects.toThrow(TypeError);
      expect(tiles).toBe(0);
    } finally {
      await engine.dispose();
    }
  });
});

describe("responses report the bundle's real precision", () => {
  // Regression: VelaTask used to report "fp32" for every bundle, including w8a32/w4a32 quantizations.
  test.each([
    ["ner", "w8a32"],
    ["span", "w4a32"],
    ["fast-decision", "w8a32"],
  ])("%s as %s", async (name, precision) => {
    const engine = await loadVela(editedBundle(name, (m) => (m.precision = precision)));
    try {
      const f = fixtures(name)[0];
      const r = await engine.systemOne({ state: f.state, questions: { q: f.question } });
      expect(r.precision).toBe(precision);
    } finally {
      await engine.dispose();
    }
  });
});

describe("input contract", () => {
  // Requests arrive from JSON and untyped callers; `as never` marks the deliberately invalid inputs below.
  test("text adapters refuse images; vision adapters need exactly one", async () => {
    const text = await loadTask(bundle("ner")),
      vision = await loadTask(bundle("vision-span"));
    try {
      const fq = fixtures("ner")[0].question,
        vq = fixtures("vision-span")[0].question;
      await expect(text.systemOne({ state: "Ada", questions: { q: fq }, images: ["x.png"] })).rejects.toThrow(/does not accept images/);
      await expect(vision.systemOne({ state: "A cat.", questions: { q: vq } })).rejects.toThrow(/exactly one image/);
      await expect(vision.infer({ state: "A cat.", question: vq, images: "x" as never })).rejects.toThrow(/array/);
      await expect(text.systemOne({ questions: [] as never })).rejects.toThrow(/questions/);
    } finally {
      await text.dispose();
      await vision.dispose();
    }
  });
  test("decision bundles validate every question before running any", async () => {
    const fast = await loadDecision(bundle("fast-decision")),
      vision = await loadDecision(bundle("vision-decision"));
    try {
      const ok = { type: "noul", instructions: "x" } as const;
      await expect(fast.systemOne({ questions: { a: ok, b: { type: "span" } } })).rejects.toThrow(/Supported question types/);
      await expect(fast.systemOne({ questions: { a: ok }, images: ["x"] })).rejects.toThrow(/text only/);
      await expect(fast.systemOne({ questions: null as never })).rejects.toThrow(/questions/);
      await expect(fast.systemOne({ questions: { a: ok }, images: "x" as never })).rejects.toThrow(/array/);
      const red = join(bundle("vision-decision"), "red.png");
      await expect(vision.systemOne({ questions: { a: ok }, images: [red, red, red, red] })).rejects.toThrow(/At most 3/);
    } finally {
      await fast.dispose();
      await vision.dispose();
    }
  });
  test("a loaded adapter refuses a question routed to another adapter", async () => {
    const engine = await loadVela(bundle("ner"));
    try {
      await expect(engine.systemOne({ state: "x@y.org", questions: { q: { type: "span", criteria: ["EMAIL_ADDRESS"] } } })).rejects.toThrow(/span_set adapter/);
    } finally {
      await engine.dispose();
    }
  });
  test("an answer without words to score skips the model and returns no spans", async () => {
    const engine = await loadTask(bundle("ner"));
    try {
      const r = await engine.infer({ state: "   ", question: { type: "span", criteria: ["person"] } });
      expect(r.logits).toEqual([]);
      expect(r.answer).toMatchObject({ type: "span", spans: [] });
    } finally {
      await engine.dispose();
    }
  });
});

describe("lifecycle", () => {
  test("calls run in order and dispose waits for them", async () => {
    const engine = await loadTask(bundle("ner"));
    const f = fixtures("ner"),
      order: number[] = [];
    const calls = f.map((x, i) => engine.systemOne({ state: x.state, questions: { q: x.question } }).then(() => order.push(i)));
    await engine.dispose();
    await Promise.all(calls);
    expect(order).toEqual([0, 1]);
    await expect(engine.infer({ state: "Ada", question: f[0].question })).rejects.toThrow(/disposed/);
    await engine.dispose(); // idempotent
  });
  test("a failed call does not block the queue", async () => {
    const engine = await loadVela(bundle("ner"));
    try {
      const bad = engine.systemOne({ state: 3, questions: { q: { type: "span", criteria: ["person"] } } });
      const good = engine.systemOne({ state: "Ada", questions: { q: { type: "span", criteria: ["person"] } } });
      await expect(bad).rejects.toThrow(TypeError);
      expect((await good).answers.q.type).toBe("span");
    } finally {
      await engine.dispose();
    }
  });
  test("a disposed decision engine refuses calls; dispose is idempotent", async () => {
    const engine = await loadVela(bundle("fast-decision"));
    await engine.dispose();
    await engine.dispose();
    await expect(engine.systemOne({ questions: { q: { type: "noul" } } })).rejects.toThrow(/disposed/);
  });
});
