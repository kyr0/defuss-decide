// src/decide.ts: the pure steps of the typed decision pipeline (validation, state as text lines, observations, answers).
import { describe, expect, test } from "vitest";
import { decideAnswer, decisionContext, decisionQuestion, evidenceOf, flattenState, ocrObservations, regionObservation, validateDecide, withObservations, meanAnswer } from "../../src/decide.ts";
import type { BoxAnswer, DecideQuestion, Span } from "../../src/types.ts";

describe("decide questions", () => {
  test("need instructions; options need two names with string or null descriptions; a region must be text", () => {
    expect(() => validateDecide({ type: "decide", instructions: "Is the order fulfilled?" })).not.toThrow();
    expect(() => validateDecide({ type: "decide", instructions: "Ship?", options: { ship: "Send it", hold: null } })).not.toThrow();
    for (const q of [
      { type: "decide", instructions: " " },
      { type: "decide", instructions: "Ship?", options: { ship: null } },
      { type: "decide", instructions: "Ship?", options: ["ship", "hold"] },
      { type: "decide", instructions: "Ship?", options: { ship: 1, hold: null } },
      { type: "decide", instructions: "Ship?", region: "" },
    ])
      expect(() => validateDecide(q as never)).toThrow(TypeError);
  });
  test("become noul without options and choice with them", () => {
    expect(decisionQuestion({ type: "decide", instructions: "Paid?" })).toEqual({ type: "noul", instructions: "Paid?" });
    const q: DecideQuestion = { type: "decide", instructions: "Next step?", options: { ship: null, hold: "Wait for payment" } };
    expect(decisionQuestion(q)).toEqual({ type: "choice", instructions: "Next step?", criteria: q.options });
  });
});

describe("state and observations", () => {
  test("JSON state becomes one path: value line per leaf; text stays text", () => {
    expect(flattenState({ order: { status: "delivered", items: [{ sku: "A1", shipped: 2 }], paid: true, note: null } })).toBe(
      "order.status: delivered\norder.items.0.sku: A1\norder.items.0.shipped: 2\norder.paid: true\norder.note: null",
    );
    expect(flattenState("Order 17 was delivered.")).toBe("Order 17 was delivered.");
    expect(flattenState(42)).toBe("42");
  });
  test("observations append to text, join objects as a field and wrap anything else", () => {
    expect(withObservations("Order 17.", ["seen"])).toBe("Order 17.\nseen");
    expect(withObservations({ order: 17 }, ["seen"])).toEqual({ order: 17, observations: ["seen"] });
    expect(withObservations([1, 2], ["seen"])).toEqual({ state: [1, 2], observations: ["seen"] });
    const state = { order: 17 };
    expect(withObservations(state, [])).toBe(state);
  });
  test("vision answers become readable observation lines", () => {
    const box: BoxAnswer = { type: "box", box: { top: 0.191, bottom: 0.75, left: 0.19, right: 0.754 }, point: { y: 0.5, x: 0.5 }, confidence: 0.8312 };
    expect(regionObservation("the parcel", box)).toBe("the image shows the parcel (confidence 0.83)");
    expect(regionObservation("a person", { ...box, confidence: 0.2 })).toBe("the image does not show a person (confidence 0.20)");
    const span: Span = { start: 0, end: 11, label: "ocr_error", text: "Invoice 128", score: 0.9 };
    expect(ocrObservations({ type: "span", target: "ocr", head: "ocr", threshold: 0.5, spans: [span] })).toEqual(['OCR text not supported by the scan: "Invoice 128"']);
    expect(ocrObservations({ type: "span", target: "ocr", head: "ocr", threshold: 0.5, spans: [] })).toEqual(["OCR text matches the scan"]);
  });
});

describe("decision context and evidence references", () => {
  const observations = ["the image shows the label (confidence 0.99)", 'OCR text not supported by the scan: "A\nB"'];
  test("the context is exactly what flattenState(withObservations()) reads, for every state shape", () => {
    for (const state of [{ order: { status: "delivered", items: [{ sku: "A1" }] } }, "Order 17.", [1, 2], 42, "", {}])
      for (const obs of [[], observations]) expect(decisionContext(state, obs).context).toBe(flattenState(withObservations(state, obs)));
  });
  test("each field's range covers its line, values with line breaks included", () => {
    const { context, fields } = decisionContext({ ocr: "ALPEN\nQUELL", n: 1 }, observations);
    const cut = (f: { start: number; end: number }) => Array.from(context).slice(f.start, f.end).join("");
    expect(fields.map((f) => [cut(f), f.path ?? null, f.observation ?? null])).toEqual([
      ["ocr: ALPEN\nQUELL", "ocr", null],
      ["n: 1", "n", null],
      [`observations.0: ${observations[0]}`, null, 0],
      [`observations.1: ${observations[1]}`, null, 1],
    ]);
  });
  test("evidence refers to the observations by index and the state by path, without text", () => {
    const { context, fields } = decisionContext({ order: { status: "delivered" } }, observations);
    const at = (needle: string) => Array.from(context.slice(0, context.indexOf(needle))).length;
    const span = (from: string, to: string) => ({ start: at(from), end: at(to) + Array.from(to).length, label: "evidence", text: "copied", score: 0.8 });
    expect(evidenceOf([span("observations.1", '"A\nB"')], fields)).toEqual([{ start: at("observations.1"), end: Array.from(context).length, label: "evidence", score: 0.8, observations_ref: [1], state_ref: [] }]);
    const [both] = evidenceOf([span("order.status", "the label")], fields);
    expect([both.state_ref, both.observations_ref]).toEqual([["order.status"], [0]]);
  });
  test("a text state is referenced by offsets only; an array state keeps its own paths", () => {
    const text = decisionContext("Order 17 was delivered.", observations);
    expect(evidenceOf([{ start: 0, end: 5, label: "evidence", text: "Order", score: 0.5 }], text.fields)[0]).toMatchObject({ state_ref: [], observations_ref: [] });
    expect(decisionContext([1, 2], observations).fields.slice(0, 2).map((f) => f.path)).toEqual(["0", "1"]);
  });
});

describe("answers", () => {
  test("a noul answer decides yes at p >= 0.5 and reports both probabilities; a choice answer keeps its options", () => {
    expect(decideAnswer({ type: "noul", noul: 0.75 }, [], "ctx", [])).toEqual({ type: "decide", decision: "yes", probabilities: { no: 0.25, yes: 0.75 }, evidence: [], context: "ctx", observations: [] });
    expect(decideAnswer({ type: "noul", noul: 0.2 }, [], "", []).decision).toBe("no");
    expect(decideAnswer({ type: "choice", choice: "hold", probabilities: { ship: 0.3, hold: 0.7 } }, [], "", ["o"])).toMatchObject({ decision: "hold", probabilities: { ship: 0.3, hold: 0.7 }, observations: ["o"] });
    expect(() => decideAnswer({ type: "span", target: "x", head: "x", threshold: 0.5, spans: [] }, [], "", [])).toThrow(TypeError);
  });
});

describe("meanAnswer", () => {
  test("averages p(yes) of two noul answers", () => {
    expect(meanAnswer({ type: "noul", noul: 0.2 }, { type: "noul", noul: 0.6 })).toEqual({ type: "noul", noul: 0.4 });
  });
  test("averages each option of two choice answers and picks the most probable", () => {
    const a = meanAnswer({ type: "choice", choice: "x", probabilities: { x: 0.6, y: 0.4 } }, { type: "choice", choice: "y", probabilities: { x: 0.1, y: 0.9 } });
    expect(a.type === "choice" && a.choice).toBe("y");
    expect(a.type === "choice" && a.probabilities.x).toBeCloseTo(0.35, 12);
  });
  test("refuses answers of different types", () => {
    expect(() => meanAnswer({ type: "noul", noul: 0.5 }, { type: "choice", choice: "x", probabilities: { x: 1 } })).toThrow(TypeError);
  });
});
