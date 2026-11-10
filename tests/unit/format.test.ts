import { describe, expect, test } from "vitest";
import sharp from "sharp";
import { join } from "node:path";
import {
  answer,
  checkLength,
  formatFast,
  formatVision,
  jsonText,
  probabilities,
  validateQuestion,
} from "../../src/format.ts";
import { preprocessImage } from "../../src/image.ts";
import { bundle, fixtures, manifest, tokenizer } from "./helpers.ts";

// A code-point tokenizer isolates the layout rules from BPE details.
const chars = { encode: (s: string) => ({ ids: Array.from(s, (c) => c.codePointAt(0)!) }) };
const layout = { markers: { "[Q]": 1, "[O]": 2, "[SEP_SCHEMA]": 3, "[SEG_user]": 4 }, bosId: 0, eosId: 0, maxTokens: 200 };

describe("decision answers", () => {
  test("softmax stays finite for extreme logits and ties pick the first option", () => {
    expect(probabilities([10000, 10000])).toEqual([0.5, 0.5]);
    expect(answer({ type: "choice" }, ["first", "second"], [0, 0])).toMatchObject({ choice: "first" });
  });
  test("empty or non-finite logits are an error, not a decision", () => {
    expect(() => probabilities([NaN])).toThrow(/Non-finite/);
    expect(() => probabilities([])).toThrow(/empty/);
  });
  test("noul returns p(yes) and option-count mismatches fail", () => {
    expect(answer({ type: "noul" }, ["no", "yes"], [0, 0])).toEqual({ type: "noul", noul: 0.5 });
    expect(() => answer({ type: "noul" }, ["no", "yes"], [0, 0, 0])).toThrow(/count mismatch/);
  });
});

describe("question validation", () => {
  test("refuses shapes the heads were not trained on", () => {
    for (const q of [
      { type: "span" },
      { type: "choice", criteria: {} },
      { type: "choice", criteria: ["a"] },
      { type: "choice", criteria: { 1: null, a: null } },
      { type: "noul", criteria: { yes: "x" } },
      { type: "choice", criteria: { a: 2 } },
    ])
      expect(() => validateQuestion(q)).toThrow(TypeError);
  });
  test("returns option names in model order", () => {
    expect(validateQuestion({ type: "noul" })).toEqual(["no", "yes"]);
    expect(validateQuestion({ type: "choice", criteria: { b: null, a: "x" } })).toEqual(["b", "a"]);
  });
  test("jsonText keeps strings verbatim and refuses non-JSON values", () => {
    expect(jsonText("a\"b")).toBe("a\"b");
    expect(jsonText({ a: [1, true, null] })).toBe('{"a":[1,true,null]}');
    expect(() => jsonText(undefined)).toThrow(TypeError);
  });
});

describe("text decision layout", () => {
  test("pooling covers [SEG_user] and the state, never EOS", () => {
    const r = formatFast(chars, layout, "abc", { type: "choice", instructions: "q", criteria: { a: null, b: "desc" } });
    expect(r.ids.slice(...r.pool)).toEqual([4, 97, 98, 99]);
    expect(r.ids[r.pool[1]]).toBe(0);
  });
  test("an over-long input fails instead of being truncated", () => {
    expect(() => formatFast(chars, { ...layout, maxTokens: 4 }, "abc", { type: "noul" })).toThrow(/Nothing was truncated/);
    expect(() => checkLength([1, 2, 3], 2)).toThrow(RangeError);
  });
  test.each(fixtures("fast-decision").map((f) => [f.name, f]))(
    "Python parity: %s renders the reference ids, option positions and pool",
    (_, f) => {
      const r = formatFast(tokenizer("fast"), manifest("fast-decision"), f.state, f.question);
      expect(r.ids).toEqual(f.ids);
      expect(r.optionPositions).toEqual(f.optionPositions);
      expect(r.pool).toEqual(f.pool);
      expect(r.names).toEqual(f.names);
    },
  );
});

describe("vision decision layout", () => {
  const m = manifest("vision-decision");
  const grid = async (file: string) => {
    const { data, info } = await sharp(join(bundle("vision-decision"), file)).removeAlpha().raw().toBuffer({ resolveWithObject: true });
    return preprocessImage({ data, width: info.width, height: info.height }).grid;
  };
  test.each(fixtures("vision-decision").map((f) => [f.name, f]))(
    "Python parity: %s renders the reference ids and option spans",
    async (_, f) => {
      const grids = await Promise.all((f.images ?? []).map(grid));
      const r = formatVision(tokenizer("vision"), m, f.state, f.question, grids);
      expect(r.ids).toEqual(f.ids);
      expect(r.spans).toEqual(f.spans);
    },
  );
  test("refuses more images than the bundle allows and unknown grids", () => {
    const q = { type: "noul", instructions: "x" } as const;
    expect(() => formatVision(tokenizer("vision"), m, "", q, ["1x2", "1x2", "1x2", "1x2"])).toThrow(/At most 3/);
    expect(() => formatVision(tokenizer("vision"), m, "", q, ["3x3"])).toThrow(/Unsupported image grid/);
  });
  test("text ids above the text vocabulary become the unknown id", () => {
    const fake = { encode: () => ({ ids: [49200, 5] }) };
    const r = formatVision(fake, m, "", { type: "noul" }, []);
    expect(r.ids).not.toContain(49200);
    expect(r.ids).toContain(m.unknownTextId);
  });
});
