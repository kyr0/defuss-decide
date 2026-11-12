// The helpers behind the recorded numbers of scripts/verify-relate.ts and scripts/verify-ocr-lines.ts, and the scores and
// the rule of the accuracy gate scripts/verify-accuracy.ts.
import { describe, expect, test } from "vitest";
import { entityAgreement, entityScores, inside, passes, pointScores, shaped } from "../../scripts/verify-accuracy.ts";
import { loadPending } from "../../scripts/load-pending.ts";
import { corrupt, flaggedLines } from "../../scripts/verify-ocr-lines.ts";
import { auc } from "../../scripts/verify-relate.ts";
import { loadVela } from "../../src/node.ts";
import type { SpanAnswer } from "../../src/types.ts";
import { editedBundle, fixtures, readJSON } from "./helpers.ts";

describe("auc", () => {
  test("is the share of differently labelled pairs in label order; ties in score count half", () => {
    expect(auc([1, 0, 0], [0.9, 0.2, 0.1])).toBe(1);
    expect(auc([1, 0], [0.1, 0.9])).toBe(0);
    expect(auc([1, 0], [0.5, 0.5])).toBe(0.5);
    expect(auc([1, 0.5, 0], [0.9, 0.1, 0.5])).toBeCloseTo(2 / 3, 12);
  });
  test("is NaN without a pair of different labels", () => {
    expect(auc([1, 1], [0.2, 0.3])).toBeNaN();
  });
});

describe("corrupt", () => {
  test("changes the last digit by 3, else swaps two letters of the longest word", () => {
    expect(corrupt("Ahornweg 27")).toBe("Ahornweg 20");
    expect(corrupt("ATLANTIS 9 km")).toBe("ATLANTIS 2 km");
    expect(corrupt("LIMITED EDITION")).toBe("LMIITED EDITION");
  });
});

describe("flaggedLines", () => {
  test("marks the lines a span overlaps, by code-point offsets", () => {
    const answer: SpanAnswer = { type: "span", target: "ocr", head: "ocr_check", threshold: 0.45, spans: [{ start: 9, end: 12, label: "ocr_error", text: "72", score: 1 }] };
    expect(flaggedLines("Grüße 👋\nWeg 72\n\nEnde", answer)).toEqual([false, true, false]);
  });
});

describe("the accuracy gate", () => {
  const person = { start: 0, end: 4, label: "person" },
    place = { start: 10, end: 16, label: "location" };
  test("an entity counts when start, end and label equal a labelled one", () => {
    const gold = [[person, place], []];
    // One exact span, one with the right boundaries and the wrong label, one in a sentence without entities.
    const scores = entityScores(gold, [[person, { ...place, label: "building" }], [{ start: 2, end: 5, label: "event" }]]);
    expect(scores).toEqual({ sentences: 2, labelled: 2, spans: 3, correct: 1, precision: 0.3333, recall: 0.5, f1: 0.4 });
    expect(entityScores(gold, [[], []])).toMatchObject({ spans: 0, correct: 0, precision: 0, recall: 0, f1: 0 });
  });
  test("agreement counts the sentences with the same spans and the spans only one side has", () => {
    expect(entityAgreement([[person, place], [person], []], [[person], [person], [place]])).toEqual({ sameSentences: 1, spansOnlyFp32: 1, spansOnlyQuantized: 1 });
  });
  test("a point scores when it lies in the labelled region, borders included", () => {
    const region = { left: 0.2, top: 0.1, right: 0.6, bottom: 0.5 };
    expect(inside({ x: 0.2, y: 0.5 }, region)).toBe(true);
    expect(inside({ x: 0.61, y: 0.3 }, region)).toBe(false);
    expect(pointScores([region, region, region], [{ x: 0.4, y: 0.3 }, { x: 0.7, y: 0.3 }, { x: 0.4, y: 0.6 }])).toEqual({ expressions: 3, hits: 1, pointInRegion: 0.3333 });
  });
  test("a bundle passes while its gated figure is at most the allowed drop below FP32's", () => {
    expect(passes("ner", { f1: 0.557 }, { f1: 0.56 })).toBe(true);
    expect(passes("ner", { f1: 0.557 }, { f1: 0.547 })).toBe(true);
    expect(passes("ner", { f1: 0.557 }, { f1: 0.5469 })).toBe(false);
    expect(passes("box", { pointInRegion: 0.708 }, { pointInRegion: 0.7027 })).toBe(true);
    expect(passes("box", { pointInRegion: 0.708 }, { pointInRegion: 0.68 })).toBe(false);
  });
  test("recorded logits keep the fixture's shape: rows of labels for text, one row for an image", () => {
    expect(shaped([1, 2, 3, 4, 5, 6], [[0, 0, 0], [0, 0, 0]])).toEqual([[1, 2, 3], [4, 5, 6]]);
    expect(shaped([1, 2, 3], [0, 0, 0])).toEqual([1, 2, 3]);
  });
});

describe("the gates' loader", () => {
  test("loads a bundle that waits for its gate, which the public loaders refuse, and leaves its manifest alone", async () => {
    const pending = editedBundle("ner", (m) => {
      m.validated = false;
      m.pendingVerification = "node scripts/verify-accuracy.ts models/vela3-fast-onnx/ner/q4";
    });
    await expect(loadVela(pending, { backend: "cpu", acceptUnvalidated: true })).rejects.toThrow(/not verified yet/);
    const engine = await loadPending(pending, "cpu");
    try {
      const f = fixtures("ner")[0];
      const r = await engine.infer({ state: f.state, question: f.question });
      expect(r.record.ids).toEqual(f.ids);
      expect(engine.backend).toBe("cpu");
    } finally {
      await engine.dispose();
    }
    expect(readJSON(`${pending}/manifest.json`).validated).toBe(false);
  });
});
