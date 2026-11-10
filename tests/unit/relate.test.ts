// relate.ts: the pure steps of paragraph relations (validation, the per-paragraph state and question, the ranking).
import { describe, expect, test } from "vitest";
import { RELATE_CRITERIA, RELATE_QUESTION, relateAnswer, relateParagraphs, relateState, relationScores } from "../../src/relate.ts";
import type { Answer } from "../../src/types.ts";

describe("relateParagraphs", () => {
  test("reads an array (keys are indexes) or an object (keys kept, in order)", () => {
    expect(relateParagraphs({ type: "relate", statement: "S" }, ["a", "b"])).toEqual([["0", "a"], ["1", "b"]]);
    expect(relateParagraphs({ type: "relate", statement: "S", by: "refutes" }, { p7: "x", p2: "y" })).toEqual([["p7", "x"], ["p2", "y"]]);
  });
  test("refuses a missing statement, an unknown relation and paragraphs that are not non-empty strings", () => {
    expect(() => relateParagraphs({ type: "relate", statement: " " }, ["a"])).toThrow(/statement/);
    expect(() => relateParagraphs({ type: "relate", statement: "S", by: "contradicts" as never }, ["a"])).toThrow(/by must be/);
    for (const state of [[], {}, "text", ["a", ""], [1], null]) expect(() => relateParagraphs({ type: "relate", statement: "S" }, state)).toThrow(TypeError);
  });
});

describe("one paragraph's decision", () => {
  test("the state labels statement and paragraph; the question is a four-option choice", () => {
    expect(relateState("The sky is blue.", "Photo of a blue sky.")).toBe("statement: The sky is blue.\nparagraph: Photo of a blue sky.");
    expect(RELATE_QUESTION).toMatchObject({ type: "choice", criteria: RELATE_CRITERIA });
    expect(Object.keys(RELATE_CRITERIA)).toEqual(["supports", "refutes", "relates", "unrelated"]);
  });
  test("relationScores: supports and refutes are their probabilities, relates is everything but unrelated", () => {
    const answer: Answer = { type: "choice", choice: "supports", probabilities: { supports: 0.6, refutes: 0.1, relates: 0.2, unrelated: 0.1 } };
    const s = relationScores(answer);
    expect(s.supports).toBe(0.6);
    expect(s.refutes).toBe(0.1);
    expect(s.relates).toBeCloseTo(0.9, 12);
    expect(() => relationScores({ type: "noul", noul: 0.5 })).toThrow(/answered relate with noul/);
    expect(() => relationScores({ type: "choice", choice: "a", probabilities: { a: 1 } })).toThrow(TypeError);
  });
});

describe("relateAnswer", () => {
  const paragraphs: [string, string][] = [["0", "a"], ["1", "b"], ["2", "c"]];
  const scores = [
    { supports: 0.2, refutes: 0.7, relates: 0.9 },
    { supports: 0.9, refutes: 0.05, relates: 0.95 },
    { supports: 0.2, refutes: 0.1, relates: 0.3 },
  ];
  test("sorts by supports by default, highest first, keeping ties in state order", () => {
    const a = relateAnswer({ type: "relate", statement: "S" }, paragraphs, scores);
    expect(a.by).toBe("supports");
    expect(a.ranking.map((r) => r.key)).toEqual(["1", "0", "2"]);
    expect(a.ranking[0]).toEqual({ key: "1", supports: 0.9, refutes: 0.05, relates: 0.95 });
  });
  test("sorts by the requested relation", () => {
    expect(relateAnswer({ type: "relate", statement: "S", by: "refutes" }, paragraphs, scores).ranking.map((r) => r.key)).toEqual(["0", "2", "1"]);
    expect(relateAnswer({ type: "relate", statement: "S", by: "relates" }, paragraphs, scores).ranking.map((r) => r.key)).toEqual(["1", "0", "2"]);
  });
});
