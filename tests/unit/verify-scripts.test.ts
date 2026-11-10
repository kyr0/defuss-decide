// The helpers behind the recorded numbers of scripts/verify-relate.ts and scripts/verify-ocr-lines.ts.
import { describe, expect, test } from "vitest";
import { corrupt, flaggedLines } from "../../scripts/verify-ocr-lines.ts";
import { auc } from "../../scripts/verify-relate.ts";
import type { SpanAnswer } from "../../src/types.ts";

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
