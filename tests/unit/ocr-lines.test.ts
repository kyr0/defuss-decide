// ocr-lines.ts and cropRGB: the pure steps of the line-by-line OCR check.
import { describe, expect, test } from "vitest";
import { cropRGB } from "../../src/image.ts";
import { STRIP_PADDING, lineRegion, lineStrip, mergeLineAnswers, ocrLines } from "../../src/ocr-lines.ts";
import type { SpanAnswer } from "../../src/types.ts";

describe("ocrLines", () => {
  test("keeps non-blank lines with code-point starts, so offsets stay right after umlauts and emoji", () => {
    const text = "Grüße 👋\n\n  \nTotal: 42";
    const lines = ocrLines(text);
    expect(lines).toEqual([{ text: "Grüße 👋", start: 0 }, { text: "Total: 42", start: 12 }]);
    expect(Array.from(text).slice(12, 21).join("")).toBe("Total: 42");
  });
  test("refuses text without a non-blank line", () => {
    expect(() => ocrLines(" \n\n")).toThrow(/needs OCR text/);
  });
  test("asks vision-box for the quoted line", () => {
    expect(lineRegion("EUR 42.00")).toBe('the text "EUR 42.00"');
  });
});

describe("lineStrip", () => {
  test("spans the full width and the box's rows plus the padding, clamped to the image", () => {
    expect(lineStrip({ top: 0.5, bottom: 0.625, left: 0.25, right: 0.5 }, 800, 1000)).toEqual({ left: 0, right: 800, top: 500 - STRIP_PADDING, bottom: 625 + STRIP_PADDING });
    expect(lineStrip({ top: 0, bottom: 1, left: 0, right: 1 }, 800, 1000)).toEqual({ left: 0, right: 800, top: 0, bottom: 1000 });
    expect(lineStrip({ top: 0.1, bottom: 0.2, left: 0, right: 1 }, 100, 100, 3)).toEqual({ left: 0, right: 100, top: 7, bottom: 23 });
  });
});

describe("mergeLineAnswers", () => {
  const answer = (spans: SpanAnswer["spans"]): SpanAnswer => ({ type: "span", target: "ocr", head: "ocr_check", threshold: 0.45, spans });
  test("moves each line's spans by the line's start", () => {
    const merged = mergeLineAnswers(
      [{ text: "Invoice 128", start: 0 }, { text: "EUR 92.00", start: 12 }],
      [answer([{ start: 8, end: 11, label: "ocr_error", text: "128", score: 0.9 }]), answer([{ start: 4, end: 9, label: "ocr_error", text: "92.00", score: 0.8 }])],
    );
    expect(merged).toMatchObject({ type: "span", target: "ocr", head: "ocr_check", threshold: 0.45 });
    expect(merged.spans.map((s) => [s.start, s.end, s.text])).toEqual([[8, 11, "128"], [16, 21, "92.00"]]);
  });
  test("refuses answers that do not match the lines one to one", () => {
    expect(() => mergeLineAnswers([{ text: "a", start: 0 }], [])).toThrow(TypeError);
  });
});

describe("cropRGB", () => {
  // 3 × 2 image whose red channel holds the pixel index.
  const image = { data: new Uint8Array([0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0, 5, 0, 0]), width: 3, height: 2 };
  test("copies the rectangle's pixels row by row", () => {
    const c = cropRGB(image, 1, 0, 3, 2);
    expect([c.width, c.height]).toEqual([2, 2]);
    expect(Array.from(c.data).filter((_, i) => i % 3 === 0)).toEqual([1, 2, 4, 5]);
  });
  test("clamps to the image and refuses an empty rectangle", () => {
    expect(cropRGB(image, -5, -5, 99, 1).width).toBe(3);
    expect(() => cropRGB(image, 2, 0, 2, 2)).toThrow(RangeError);
  });
});
