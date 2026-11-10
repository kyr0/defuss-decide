import { describe, expect, test } from "vitest";
import sharp from "sharp";
import { join } from "node:path";
import {
  formatFastSpan,
  formatVisionTask,
  labels,
  ocrText,
  routeFastSpan,
  taskAnswer,
  visionTask,
} from "../../src/task-format.ts";
import { preprocessImage } from "../../src/image.ts";
import type { Span, SpanQuestion } from "../../src/types.ts";
import type { Fixture } from "./helpers.ts";
import { TEXT_ADAPTERS, VISION_ADAPTERS, bundle, fixtures, manifest, tokenizer } from "./helpers.ts";

const spanConfig = manifest("ner").spanConfig;
const route = (q: Omit<SpanQuestion, "type"> | Record<string, unknown>, n?: number) => routeFastSpan(spanConfig, { type: "span", ...q }, n);
const keys = (spans: Span[]) => spans.map(({ start, end, label, text }) => ({ start, end, label, text }));

async function grid(name: string, file: string) {
  const { data, info } = await sharp(join(bundle(name), file)).removeAlpha().raw().toBuffer({ resolveWithObject: true });
  return preprocessImage({ data, width: info.width, height: info.height }).grid;
}

describe("text span routing reproduces the upstream router", () => {
  test.each([
    [{ criteria: ["person"] }, "ner"],
    [{ criteria: { person: "A named human" } }, "span"],
    [{ criteria: ["EMAIL_ADDRESS"] }, "span_set"],
    [{ criteria: ["unsupported"] }, "span_set"],
    [{ criteria: ["evidence"] }, "evidence"],
    [{ criteria: { support: "x" }, instructions: "Which spans support the claim?" }, "evidence"],
    [{ criteria: ["employer"], instructions: "Find the employer relation." }, "relation"],
    [{ criteria: ["employer"], instructions: "Which spans are the employer of Ada?" }, "relation"],
    [{ criteria: ["Violence"] }, "span"],
  ])("%j → %s", (q, task) => expect(route(q).task).toBe(task));

  test("PII thresholds depend on the target length, with the calibrated temperature", () => {
    const short = route({ criteria: ["EMAIL_ADDRESS"] }, 10),
      long = route({ criteria: ["EMAIL_ADDRESS"] }, 5000);
    expect(short.threshold).toBe(spanConfig.set_adapter.pii_buckets[0].threshold);
    expect(long.threshold).toBe(spanConfig.set_adapter.pii_buckets[1].threshold);
    expect(short.temperature).toBe(spanConfig.set_adapter.router_set_temperature);
  });
  test("toxicity labels use the span adapter with their calibrated threshold and temperature", () => {
    const r = route({ criteria: ["Violence"] });
    expect([r.threshold, r.temperature]).toEqual([spanConfig.thresholds_by_set.toxic, spanConfig.router_set_temperature]);
  });
  test("known label sets fill in descriptions and the default question", () => {
    const r = route({ criteria: ["EMAIL_ADDRESS"] });
    expect(r.pairs[0][1]).toBe(spanConfig.schemas.pii.labels.EMAIL_ADDRESS);
  });
  test("an explicit threshold overrides the route; invalid ones are refused", () => {
    expect(route({ criteria: ["person"], threshold: 0.9 }).threshold).toBe(0.9);
    for (const q of [{ criteria: ["x"], threshold: NaN }, { criteria: ["x"], threshold: 2 }, { criteria: ["x"], threshold: -0.1 }])
      expect(() => route(q)).toThrow(RangeError);
  });
  test("invalid labels and instructions fail explicitly", () => {
    for (const q of [{ criteria: [] }, { criteria: ["x", "x"] }, { criteria: ["1"] }, { criteria: { a: 3 } }, { criteria: [""] }])
      expect(() => route(q)).toThrow(TypeError);
    expect(() => route({ criteria: ["x"], instructions: { a: 1 } })).toThrow(/instructions/);
    expect(() => routeFastSpan(spanConfig, { type: "box" })).toThrow(/span/);
    expect(labels({}, { unsupported: null })).toEqual([["unsupported", ""]]);
  });
});

describe("text span rendering", () => {
  const m = manifest("ner");
  test("a question for another adapter names the adapter it needs", () =>
    expect(() => formatFastSpan(tokenizer("fast"), m, "Ada", { type: "span", criteria: ["EMAIL_ADDRESS"] })).toThrow(/span_set adapter/));
  test("state must be text or {user, context, answer} text fields", () => {
    for (const state of [[], { other: "x" }, { user: 3 }, {}])
      expect(() => formatFastSpan(tokenizer("fast"), m, state, { type: "span", criteria: ["person"] })).toThrow(TypeError);
    expect(() => formatFastSpan(tokenizer("fast"), m, { user: "x" }, { type: "span", target: "answer", criteria: ["person"] })).toThrow(/target/);
  });
  test("an over-long input is refused instead of windowed", () =>
    expect(() => formatFastSpan(tokenizer("fast"), { ...m, maxTokens: 8 }, "Ada Lovelace lived in London.", { type: "span", criteria: ["person"] })).toThrow(/no truncation/));
});

describe("Python parity of every adapter's reference fixtures", () => {
  const cases = [...TEXT_ADAPTERS, ...VISION_ADAPTERS].flatMap((name) => fixtures(name).map((f): [string, string, Fixture] => [name, f.name, f]));
  test.each(cases)("%s/%s: same ids, offsets and positions, same decoded answer from the reference logits", async (name, _, f) => {
    const m = manifest(name);
    const rec =
      m.kind === "fast"
        ? formatFastSpan(tokenizer("fast"), m, f.state, f.question)
        : formatVisionTask(tokenizer("vision"), m, f.state, f.question, await grid(name, f.images[0]));
    expect(rec.ids).toEqual(f.ids);
    expect(rec.offsets).toEqual(f.offsets);
    expect(rec.wordPositions ?? rec.readoutPositions).toEqual(f.wordPositions ?? f.readoutPositions);
    if (f.labelPositions) expect(rec.labelPositions).toEqual(f.labelPositions);
    const a = taskAnswer(rec, f.logits.flat(), m);
    expect(a.type).toBe(f.answer.type);
    if (a.type === "box") {
      expect(a.box).toEqual(f.answer.box);
      expect(a.point).toEqual(f.answer.point);
      expect(a.confidence).toBeCloseTo(f.answer.confidence, 5);
    } else {
      if (a.type !== "span") throw new Error(`unexpected answer type ${a.type}`);
      expect(keys(a.spans)).toEqual(keys(f.answer.spans));
      expect([a.head, a.target, a.threshold]).toEqual([f.answer.head, f.answer.target, f.answer.threshold]);
      a.spans.forEach((s, i) => expect(s.score).toBeCloseTo(f.answer.spans[i].score, 5));
    }
  });
});

describe("vision questions", () => {
  test("visionTask routes box, OCR and image-hallucination questions", () => {
    expect(visionTask({ type: "box", instructions: "x" })).toBe("box");
    expect(visionTask({ type: "span", task: "ocr_check" })).toBe("ocr");
    expect(visionTask({ type: "span", criteria: ["ocr_error"] })).toBe("ocr");
    expect(visionTask({ type: "span" })).toBe("span");
    expect(() => visionTask({ type: "choice" })).toThrow(TypeError);
  });
  test("over-long answers are refused instead of silently clipped at 768 tokens", () =>
    expect(() => formatVisionTask(tokenizer("vision"), manifest("vision-span"), "word ".repeat(800), { type: "span" }, "2x2")).toThrow(/exceeds/));
  test("labels the image adapter cannot predict are refused", () =>
    expect(() => formatVisionTask(tokenizer("vision"), manifest("vision-span"), "A person", { type: "span", criteria: ["person"] }, "2x2")).toThrow(/unsupported/));
  test("an image span question needs an answer to check", () => {
    expect(() => formatVisionTask(tokenizer("vision"), manifest("vision-span"), { user: "x" }, { type: "span" }, "2x2")).toThrow(/answer/);
    expect(() => formatVisionTask(tokenizer("vision"), manifest("vision-span"), { answer: 3 }, { type: "span" }, "2x2")).toThrow(TypeError);
  });
  test("a vision question for another adapter is refused", () =>
    expect(() => formatVisionTask(tokenizer("vision"), manifest("vision-span"), "", { type: "box", instructions: "x" }, "2x2")).toThrow(/requires box/));
  test("box instructions must be text and thresholds valid", () => {
    expect(() => formatVisionTask(tokenizer("vision"), manifest("vision-box"), "", { type: "box", instructions: 3 as never }, "2x2")).toThrow(/text/);
    expect(() => formatVisionTask(tokenizer("vision"), manifest("vision-ocr"), "x", { type: "span", task: "ocr_check", threshold: 3 }, "2x2")).toThrow(RangeError);
  });
});

describe("OCR text", () => {
  test("nested fields follow the Python flattening", () =>
    expect(ocrText({ invoice: { id: 123, paid: true }, items: ["A", null, "B"] })).toBe(
      "invoice.id: 123\ninvoice.paid: True\nitems.0: A\nitems.2: B",
    ));
  test("ocr or answer strings are used verbatim", () => {
    expect(ocrText({ ocr: "Invoice 128" })).toBe("Invoice 128");
    expect(ocrText({ answer: "x" })).toBe("x");
    expect(ocrText("plain")).toBe("plain");
  });
  test("decimal numbers require explicit text, other states are refused", () => {
    expect(() => ocrText({ price: 1.25 })).toThrow(/as strings/);
    expect(() => ocrText([1])).toThrow(TypeError);
    expect(() => ocrText(null)).toThrow(TypeError);
  });
});
