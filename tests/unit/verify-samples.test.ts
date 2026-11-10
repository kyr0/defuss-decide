// scripts/verify-samples.ts `matches`: whether a recorded answer on a sample image is what a person reads off the image.
import { describe, expect, test } from "vitest";
import { matches } from "../../scripts/verify-samples.ts";
import type { Answer } from "../../src/types.ts";

const box = (x: number, y: number) => ({ type: "box", box: { top: 0, bottom: 1, left: 0, right: 1 }, point: { x, y }, confidence: 0.9 }) as Answer;
const spans = (...texts: string[]) => ({ type: "span", target: "ocr", head: "ocr_check", threshold: 0.45, spans: texts.map((text) => ({ start: 0, end: 1, label: "ocr_error", score: 1, text })) }) as Answer;
const decision = (d: string) => ({ type: "decide", decision: d, probabilities: {}, evidence: [], context: "", observations: [] }) as Answer;

describe("matches", () => {
  const inside = { left: 0.5, right: 0.8, top: 0.2, bottom: 0.6 };
  test("a box matches when its point lies inside the expected region, edges included", () => {
    expect(matches(box(0.6, 0.4), { inside })).toBe(true);
    expect(matches(box(0.5, 0.6), { inside })).toBe(true);
    expect(matches(box(0.9, 0.4), { inside })).toBe(false);
  });
  test("a span answer must flag every expected text and nothing else; containment either way counts", () => {
    expect(matches(spans("9"), { flag: ["ATLANTIS 9 km"] })).toBe(true);
    expect(matches(spans("Keller\nAhornweg 72"), { flag: ["Ahornweg 72"] })).toBe(true);
    expect(matches(spans(), { flag: ["500 ML"] })).toBe(false);
    expect(matches(spans("500 ML", "ALPENQUELL"), { flag: ["500 ML"] })).toBe(false);
  });
  test("a decision matches the expected option; an expectation of another kind never matches", () => {
    expect(matches(decision("yes"), { decision: "yes" })).toBe(true);
    expect(matches(decision("no"), { decision: "yes" })).toBe(false);
    expect(matches(box(0.6, 0.4), { decision: "yes" })).toBe(false);
  });
});
