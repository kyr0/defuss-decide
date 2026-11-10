import { describe, expect, test } from "vitest";
import { join } from "node:path";
import sharp from "sharp";
import { loadVela, loadVelaProvider } from "../../src/node.ts";
import { VelaProvider } from "../../src/provider.ts";
import type { BundleMap } from "../../src/types.ts";
import { bundle, editedBundle, expected, fixtures, manifest, sameAnswer } from "./helpers.ts";

const reference = expected();
const all: BundleMap = {
  fast: { ...Object.fromEntries(["span", "span_set", "ner", "relation", "evidence"].map((t) => [t, bundle(t)])), decision: bundle("fast-decision") },
  vision: { span: bundle("vision-span"), ocr: bundle("vision-ocr"), box: bundle("vision-box"), decision: bundle("vision-decision") },
};

/** An image file decoded to RGB bytes, which needs no decoder in the provider. */
async function rgb(file: string) {
  const { data, info } = await sharp(file).removeAlpha().raw().toBuffer({ resolveWithObject: true });
  return { data: new Uint8Array(data), width: info.width, height: info.height };
}

/** A provider over real bundles whose loads and disposals are recorded. */
function observed(bundles: BundleMap = all) {
  const events: string[] = [];
  const provider = new VelaProvider({
    bundles,
    spanConfig: manifest("ner").spanConfig,
    load: async (source) => {
      const name = source.split("/").at(-1);
      events.push("load " + name);
      const engine = await loadVela(source);
      const release = engine.dispose.bind(engine);
      engine.dispose = async () => {
        events.push("dispose " + name);
        await release();
      };
      return engine;
    },
  });
  return { provider, events };
}

describe("routing", () => {
  test("each question reaches its adapter and gets the same answer as the bundle alone", async () => {
    const { provider, events } = observed();
    try {
      const ner = fixtures("ner")[0],
        pii = fixtures("span_set")[0];
      const r = await provider.systemOne({
        state: ner.state,
        questions: { entities: ner.question, pii: pii.question },
      });
      expect(events).toEqual(["load ner", "dispose ner", "load span_set"]);
      expect(sameAnswer(r.answers.entities, reference.bundles.ner[0].answer)).toBe(true);
      expect(r.answers.pii.type).toBe("span");
      expect(r.model).toBe("vllm-sr/vela3-fast");
      expect(r.precision).toBe("fp32");
      const box = fixtures("vision-box")[0];
      const v = await provider.systemOne({
        questions: { where: box.question },
        images: [join(bundle("vision-box"), box.images![0])],
      });
      expect(sameAnswer(v.answers.where, reference.bundles["vision-box"][0].answer)).toBe(true);
      expect(v.model).toBe("vllm-sr/vela3-fast-vision");
      const d = await provider.systemOne({ state: "x", questions: { q: { type: "noul", instructions: "Is it?" } } });
      expect(d.answers.q.type).toBe("noul");
    } finally {
      await provider.dispose();
    }
    expect(events.at(-1)).toBe("dispose fast-decision");
  });
  test("concurrent requests run in call order with one resident adapter", async () => {
    const { provider, events } = observed();
    const img = join(bundle("vision-span"), "sample.png");
    await Promise.all([
      provider.systemOne({ state: "A cat.", questions: { x: { type: "span", criteria: ["unsupported"] } }, images: [img] }),
      provider.systemOne({ state: "Invoice 1", questions: { x: { type: "span", criteria: ["ocr_error"] } }, images: [img] }),
    ]);
    await provider.dispose();
    expect(events).toEqual(["load vision-span", "dispose vision-span", "load vision-ocr", "dispose vision-ocr"]);
    await expect(provider.systemOne({ questions: {} })).rejects.toThrow(/disposed/);
  });
  test("questions for one adapter run together, the resident adapter first, and answers keep the request order", async () => {
    const { provider, events } = observed();
    try {
      const ner = fixtures("ner")[0],
        pii = fixtures("span_set")[0];
      await provider.systemOne({ state: ner.state, questions: { e: ner.question } });
      const r = await provider.systemOne({
        state: ner.state,
        questions: { pii: pii.question, entities: ner.question, pii2: pii.question },
      });
      expect(events).toEqual(["load ner", "dispose ner", "load span_set"]);
      expect(Object.keys(r.answers)).toEqual(["pii", "entities", "pii2"]);
      expect(sameAnswer(r.answers.entities, reference.bundles.ner[0].answer)).toBe(true);
      expect(r.answers.pii2).toEqual(r.answers.pii);
      expect(["webgpu", "cpu"]).toContain(r.backend);
    } finally {
      await provider.dispose();
    }
  });
  test("every question is routed before any bundle loads", async () => {
    const { provider, events } = observed({ fast: { ner: bundle("ner") } });
    const ner = fixtures("ner")[0];
    await expect(
      provider.systemOne({ state: ner.state, questions: { e: ner.question, x: { type: "span", criteria: ["EMAIL_ADDRESS"] } } }),
    ).rejects.toThrow(/Missing bundle: fast\/span_set/);
    expect(events).toEqual([]);
    await provider.dispose();
  });
  test("a decide question runs evidence, then the decision bundle, and returns a typed answer in request order", async () => {
    const { provider, events } = observed();
    try {
      const r = await provider.systemOne({
        state: { order: { id: 17, status: "delivered" } },
        questions: {
          done: { type: "decide", instructions: "Is the order fulfilled?" },
          urgent: { type: "noul", instructions: "Is it urgent?" },
          next: { type: "decide", instructions: "What happens next?", options: { archive: null, ship: "Send the parcel" } },
        },
      });
      expect(Object.keys(r.answers)).toEqual(["done", "urgent", "next"]);
      const done = r.answers.done;
      if (done.type !== "decide") throw new Error(done.type);
      expect(["yes", "no"]).toContain(done.decision);
      expect(done.probabilities.yes + done.probabilities.no).toBeCloseTo(1, 6);
      expect(done.context).toBe("order.id: 17\norder.status: delivered");
      expect(done.observations).toEqual([]);
      // Evidence refers to the state by path instead of copying text; without images there are no observations.
      for (const s of done.evidence) {
        expect(s).not.toHaveProperty("text");
        expect(s.observations_ref).toEqual([]);
        for (const path of s.state_ref) expect(["order.id", "order.status"]).toContain(path);
        expect(s.state_ref.length).toBeGreaterThan(0);
      }
      expect(r.answers.next).toMatchObject({ type: "decide", decision: expect.stringMatching(/^(archive|ship)$/) });
      expect(r.answers.urgent.type).toBe("noul");
      // Each pipeline step loads its bundle once for both decide questions; the plain decision question then runs on
      // the decision bundle the pipeline ended on.
      expect(events).toEqual(["load evidence", "dispose evidence", "load fast-decision"]);
    } finally {
      await provider.dispose();
    }
  });
  test("with an image, the vision findings join the state before evidence and the vision decision", async () => {
    const { provider, events } = observed();
    try {
      const r = await provider.systemOne({
        state: { ocr: "Invoice 128\nTotal: EUR 92.00", customer: "Ada" },
        images: [join(bundle("vision-ocr"), "sample.png")],
        questions: { paid: { type: "decide", instructions: "Is the invoice paid?", region: "the payment stamp" } },
      });
      const paid = r.answers.paid;
      if (paid.type !== "decide") throw new Error(paid.type);
      expect(paid.observations[0]).toMatch(/^the image (shows|does not show) the payment stamp \(confidence \d\.\d\d\)$/);
      expect(paid.observations.length).toBeGreaterThan(1);
      expect(paid.context.split("\n").slice(-paid.observations.length)).toEqual(paid.observations.map((o, i) => `observations.${i}: ${o}`));
      expect(events).toEqual([
        "load vision-box",
        "dispose vision-box",
        "load vision-ocr",
        "dispose vision-ocr",
        "load evidence",
        "dispose evidence",
        "load vision-decision",
      ]);
    } finally {
      await provider.dispose();
    }
  });
  test("with two images, only the vision decision bundle reads them; the image tasks take exactly one", async () => {
    const { provider, events } = observed();
    try {
      const red = join(bundle("vision-decision"), "red.png");
      const r = await provider.systemOne({ state: "Two photos.", images: [red, red], questions: { d: { type: "decide", instructions: "Are both red?" } } });
      expect(r.answers.d).toMatchObject({ type: "decide", observations: [] });
      expect(events).toEqual(["load evidence", "dispose evidence", "load vision-decision"]);
    } finally {
      await provider.dispose();
    }
  });
  test("decide questions are checked and need their decision bundle before anything loads", async () => {
    const { provider, events } = observed({ fast: { evidence: bundle("evidence") } });
    await expect(provider.systemOne({ state: "x", questions: { d: { type: "decide", instructions: "Ok?" } } })).rejects.toThrow(/Missing bundle: fast\/decision/);
    await expect(provider.systemOne({ state: "x", questions: { d: { type: "decide", instructions: "" } } })).rejects.toThrow(TypeError);
    expect(events).toEqual([]);
    await provider.dispose();
  });
  test("without an evidence bundle, decide answers from the decision bundle alone", async () => {
    const { provider, events } = observed({ fast: { decision: bundle("fast-decision") } });
    try {
      const r = await provider.systemOne({ state: "Order 17 was delivered.", questions: { d: { type: "decide", instructions: "Is the order fulfilled?" } } });
      expect(r.answers.d).toMatchObject({ type: "decide", evidence: [], context: "Order 17 was delivered." });
      expect(events).toEqual(["load fast-decision"]);
    } finally {
      await provider.dispose();
    }
  });
  test("without a region, decide skips the region step; the decision bundle reads the state as text lines", async () => {
    const { provider, events } = observed();
    try {
      const red = join(bundle("vision-decision"), "red.png");
      const r = await provider.systemOne({ state: { note: "product photo" }, images: [red], questions: { d: { type: "decide", instructions: "Is the background white?" } } });
      expect(r.answers.d).toMatchObject({ type: "decide", observations: [], context: "note: product photo" });
      expect(events).toEqual(["load evidence", "dispose evidence", "load vision-decision"]);
    } finally {
      await provider.dispose();
    }
  });
  test("decide with lines checks the OCR text line by line before evidence and the decision", async () => {
    const { provider, events } = observed();
    try {
      const scan = await rgb(join(bundle("vision-ocr"), "sample.png"));
      const r = await provider.systemOne({
        state: { ocr: "Invoice 128\nTotal: EUR 92.00" },
        images: [scan],
        questions: { d: { type: "decide", instructions: "Does the text match the scan?", lines: true } },
      });
      const d = r.answers.d;
      if (d.type !== "decide") throw new Error(d.type);
      expect(d.observations.length).toBeGreaterThan(0);
      for (const o of d.observations) expect(o).toMatch(/^(OCR text matches the scan|OCR text not supported by the scan: ".+")$/);
      expect(events).toEqual(["load vision-box", "dispose vision-box", "load vision-ocr", "dispose vision-ocr", "load evidence", "dispose evidence", "load vision-decision"]);
    } finally {
      await provider.dispose();
    }
  });
  test("after an OCR check the decision is the mean of the vision decision with and without the image", async () => {
    const { provider } = observed({ vision: { ocr: bundle("vision-ocr"), decision: bundle("vision-decision") } });
    const scan = join(bundle("vision-ocr"), "sample.png");
    let d;
    try {
      d = (await provider.systemOne({ state: { ocr: "Invoice 128" }, images: [scan], questions: { d: { type: "decide", instructions: "Does the text match?" } } })).answers.d;
    } finally {
      await provider.dispose();
    }
    if (d.type !== "decide") throw new Error(d.type);
    const head = await loadVela(bundle("vision-decision"));
    try {
      const ask = async (images: string[]) => (await head.systemOne({ state: d.context, images, questions: { q: { type: "noul", instructions: "Does the text match?" } } })).answers.q;
      const [withImage, without] = [await ask([scan]), await ask([])];
      if (withImage.type !== "noul" || without.type !== "noul") throw new Error("noul expected");
      expect(d.probabilities.yes).toBeCloseTo((withImage.noul + without.noul) / 2, 6);
    } finally {
      await head.dispose();
    }
  });
  test("an OCR check with lines locates every line, checks each on its strip and returns offsets into the whole text", async () => {
    const { provider, events } = observed();
    try {
      const ocr = "Invoice 128\n\nTotal: EUR 92.00";
      const scan = await rgb(join(bundle("vision-ocr"), "sample.png"));
      const r = await provider.systemOne({ state: { ocr }, images: [scan], questions: { o: { type: "span", task: "ocr_check", criteria: ["ocr_error"], lines: true } } });
      const o = r.answers.o;
      if (o.type !== "span") throw new Error(o.type);
      expect(o).toMatchObject({ target: "ocr", head: "ocr_check" });
      for (const span of o.spans) expect(Array.from(ocr).slice(span.start, span.end).join("")).toBe(span.text);
      // One request locates both lines, then the OCR bundle checks one strip per line.
      expect(events).toEqual(["load vision-box", "dispose vision-box", "load vision-ocr"]);
    } finally {
      await provider.dispose();
    }
  });
  test("a line-by-line OCR check needs one image, both vision task bundles and a decoder, checked before anything loads", async () => {
    const { provider, events } = observed();
    const lines = { o: { type: "span" as const, task: "ocr_check" as const, criteria: ["ocr_error"], lines: true } };
    const scan = join(bundle("vision-ocr"), "sample.png");
    await expect(provider.systemOne({ state: { ocr: "a" }, images: [scan, scan], questions: lines })).rejects.toThrow(/exactly one image/);
    await expect(provider.systemOne({ state: { ocr: "a" }, images: [scan], questions: lines })).rejects.toThrow(/image decoder/);
    await expect(provider.systemOne({ state: { ocr: "a" }, questions: lines })).rejects.toThrow(/applies to OCR checks with an image/);
    await expect(provider.systemOne({ state: "Ada", questions: { s: { type: "span", criteria: ["person"], lines: true } } })).rejects.toThrow(/applies to OCR checks/);
    expect(events).toEqual([]);
    await provider.dispose();
    const only = observed({ vision: { ocr: bundle("vision-ocr") } });
    await expect(only.provider.systemOne({ state: { ocr: "a" }, images: [await rgb(scan)], questions: lines })).rejects.toThrow(/Missing bundle: vision\/box/);
    expect(only.events).toEqual([]);
    await only.provider.dispose();
  });
  test("a single OCR bundle refuses lines instead of ignoring them", async () => {
    const engine = await loadVela(bundle("vision-ocr"));
    try {
      await expect(
        engine.systemOne({ state: { ocr: "a" }, images: [join(bundle("vision-ocr"), "sample.png")], questions: { o: { type: "span", task: "ocr_check", criteria: ["ocr_error"], lines: true } } }),
      ).rejects.toThrow(/needs VelaProvider/);
    } finally {
      await engine.dispose();
    }
  });
  test("a relate question ranks every paragraph on the text decision bundle, sorted by the requested score", async () => {
    const { provider, events } = observed();
    try {
      const paragraphs = { blue: "Photo of a blue sky.", green: "Photo of a green sky.", flour: "Add two cups of flour." };
      const r = await provider.systemOne({
        state: paragraphs,
        questions: { s: { type: "relate", statement: "The sky is blue." }, f: { type: "relate", statement: "The sky is blue.", by: "refutes" } },
      });
      const s = r.answers.s,
        f = r.answers.f;
      if (s.type !== "relate" || f.type !== "relate") throw new Error("relate expected");
      expect(s.by).toBe("supports");
      expect(s.ranking.map((p) => p.key).sort()).toEqual(["blue", "flour", "green"]);
      for (const p of s.ranking) {
        expect(p).not.toHaveProperty("text");
        for (const k of ["supports", "refutes", "relates"] as const) {
          expect(p[k]).toBeGreaterThanOrEqual(0);
          expect(p[k]).toBeLessThanOrEqual(1);
        }
      }
      for (let i = 1; i < s.ranking.length; i++) expect(s.ranking[i - 1].supports).toBeGreaterThanOrEqual(s.ranking[i].supports);
      for (let i = 1; i < f.ranking.length; i++) expect(f.ranking[i - 1].refutes).toBeGreaterThanOrEqual(f.ranking[i].refutes);
      // The same paragraph gets the same scores in both questions.
      expect(Object.fromEntries(f.ranking.map((p) => [p.key, p]))).toEqual(Object.fromEntries(s.ranking.map((p) => [p.key, p])));
      expect(events).toEqual(["load fast-decision"]);
    } finally {
      await provider.dispose();
    }
  });
  test("relate questions are checked and need the text decision bundle before anything loads", async () => {
    const { provider, events } = observed({ vision: { decision: bundle("vision-decision") } });
    await expect(provider.systemOne({ state: ["a"], questions: { r: { type: "relate", statement: "S" } } })).rejects.toThrow(/Missing bundle: fast\/decision/);
    await expect(provider.systemOne({ state: "not paragraphs", questions: { r: { type: "relate", statement: "S" } } })).rejects.toThrow(TypeError);
    expect(events).toEqual([]);
    await provider.dispose();
  });
  test("OCR and region questions without an image never fall back to text", async () => {
    const { provider, events } = observed();
    await expect(provider.systemOne({ state: "x", questions: { x: { type: "span", task: "ocr_check" } } })).rejects.toThrow(/require an image/);
    await expect(provider.systemOne({ questions: { x: { type: "box", instructions: "x" } } })).rejects.toThrow(/require an image/);
    expect(events).toEqual([]);
    await provider.dispose();
  });
  test("a missing bundle is an error, never a substitute", async () => {
    const { provider } = observed({ fast: { ner: bundle("ner") } });
    await expect(provider.systemOne({ state: "x@y.org", questions: { x: { type: "span", criteria: ["EMAIL_ADDRESS"] } } })).rejects.toThrow(/Missing bundle: fast\/span_set/);
    await provider.dispose();
  });
  test("text span questions need the routing configuration of a text bundle", async () => {
    const provider = new VelaProvider({ bundles: all, load: loadVela });
    await expect(provider.systemOne({ state: "x", questions: { x: { type: "span", criteria: ["person"] } } })).rejects.toThrow(/No text span bundle/);
    await expect(provider.systemOne({ questions: [] as never })).rejects.toThrow(/questions object/);
    await provider.dispose();
  });
  test("bundles of different precision report mixed", async () => {
    const quantized = editedBundle("span_set", (m) => (m.precision = "w8a32"));
    const { provider } = observed({ fast: { ner: bundle("ner"), span_set: quantized } });
    try {
      const r = await provider.systemOne({
        state: "Ada at ada@example.org",
        questions: { a: { type: "span", criteria: ["person"] }, b: { type: "span", criteria: ["EMAIL_ADDRESS"] } },
      });
      expect(r.precision).toBe("mixed");
    } finally {
      await provider.dispose();
    }
  });
});

describe("loadVelaProvider (Node.js)", () => {
  test("decodes a file path for the line-by-line OCR check", async () => {
    const provider = await loadVelaProvider({ vision: { box: bundle("vision-box"), ocr: bundle("vision-ocr") } });
    try {
      const r = await provider.systemOne({
        state: { ocr: "Invoice 128" },
        images: [join(bundle("vision-ocr"), "sample.png")],
        questions: { o: { type: "span", task: "ocr_check", criteria: ["ocr_error"], lines: true } },
      });
      expect(r.answers.o).toMatchObject({ type: "span", target: "ocr" });
    } finally {
      await provider.dispose();
    }
  });
  test("reads the routing configuration from the first text bundle and loads lazily", async () => {
    const provider = await loadVelaProvider({ fast: { decision: bundle("fast-decision"), ner: bundle("ner") } }, { threads: 1 });
    try {
      const r = await provider.systemOne({ state: "Ada lives in London.", questions: { e: { type: "span", criteria: ["person"] } } });
      expect(r.answers.e.type).toBe("span");
    } finally {
      await provider.dispose();
    }
  });
  test("without a text bundle, text span questions are refused", async () => {
    const provider = await loadVelaProvider({ vision: { span: bundle("vision-span") } });
    await expect(provider.systemOne({ state: "x", questions: { e: { type: "span", criteria: ["person"] } } })).rejects.toThrow(/No text span bundle/);
    await provider.dispose();
  });
});

describe("the bundles a request loads, before it runs", () => {
  const scan = join(bundle("vision-ocr"), "sample.png");
  const requests = {
    "text spans": { state: "Ada Lovelace lived in London.", questions: { who: { type: "span", criteria: ["person"] } } },
    "decide on text": { state: { order: { status: "delivered" } }, questions: { d: { type: "decide", instructions: "Is the order fulfilled?" } } },
    "decide with a region and an OCR text": {
      state: { ocr: "Invoice 128\nTotal: EUR 92.00" },
      images: [scan],
      questions: { d: { type: "decide", instructions: "Is the invoice paid?", region: "the payment stamp" } },
    },
    "relate": { state: ["Ada works at the lab.", "Ada lives in London."], questions: { r: { type: "relate", statement: "Ada works at the lab." } } },
    "OCR line by line": { state: { ocr: "Invoice 128\nTotal: EUR 92.00" }, images: [scan], questions: { e: { type: "span", task: "ocr_check", criteria: ["ocr_error"], lines: true } } },
  } as const;
  test.each(Object.keys(requests))("%s: sources() names exactly the bundles systemOne then loads", async (name) => {
    const { provider, events } = observed();
    try {
      const request = { ...requests[name as keyof typeof requests] } as unknown as Parameters<VelaProvider["systemOne"]>[0];
      // The line-by-line check crops the image: RGB bytes, as this provider has no image decoder.
      if (name === "OCR line by line") request.images = [await rgb(scan)];
      const planned = provider.sources(request).map((s) => s.split("/").at(-1)).sort();
      await provider.systemOne(request);
      const loaded = [...new Set(events.filter((e) => e.startsWith("load ")).map((e) => e.slice(5)))].sort();
      expect(planned).toEqual(loaded);
      // The same request again needs only the resident bundle when it is the only one: nothing loads.
      if (loaded.length === 1) expect(provider.sources(request)).toEqual([]);
    } finally {
      await provider.dispose();
    }
  });
});
