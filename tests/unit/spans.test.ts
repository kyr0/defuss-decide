import { describe, expect, test } from "vitest";
import {
  CJK,
  CJK_VISION,
  decodeRegion,
  decodeSpans,
  encodeOffsets,
  firstWordTokens,
  splitWords,
} from "../../src/spans.ts";
import { join } from "node:path";
import { Tokenizer } from "@huggingface/tokenizers";
import { BUNDLES, readFixture, readJSON, tokenizer } from "./helpers.ts";
import type { Span } from "../../src/types.ts";

const strip = (spans: Span[]) => spans.map(({ score: _s, probability: _p, ...rest }) => rest);

describe("span decoder parity with the pinned Python decoders", () => {
  test.each(readFixture("decoder-fixtures.json").map((f) => [f.kind, JSON.stringify(f.text), f]))(
    "%s %s",
    (_, __, f) => {
      const a = decodeSpans(f.logits, f.offsets, f.names, f.text, 0.5, { vision: f.kind === "vision" });
      expect(strip(a)).toEqual(strip(f.answer));
      a.forEach((s, i) => expect(Math.abs(s.score - f.answer[i].score)).toBeLessThan(1e-6));
    },
  );
});

describe("tokenizer offsets parity with the Python fast tokenizers", () => {
  test.each(readFixture("offset-fixtures.json").map((f) => [f.kind, JSON.stringify(f.text), f]))(
    "%s %s",
    (kind, _, f) => {
      const e = encodeOffsets(tokenizer(kind), f.text);
      expect(e.ids).toEqual(f.ids);
      expect(e.offsets).toEqual(f.offsets);
      const w = firstWordTokens(f.text, e.offsets);
      expect(w.words).toEqual(f.words);
      expect(w.positions).toEqual(f.positions);
    },
  );
  test("a tokenizer that rewrites text fails instead of returning shifted offsets", () => {
    // The pinned tokenizers have no normalizer; the same tokenizer with one breaks the byte alignment contract.
    const json = readJSON(join(BUNDLES, "tokenizer-fast.json"));
    const lowercase = new Tokenizer({ ...json, normalizer: { type: "Lowercase" } }, readJSON(join(BUNDLES, "tokenizer_config-fast.json")));
    expect(() => encodeOffsets(lowercase, "Ada Lovelace")).toThrow(/byte alignment failed/);
  });
  test("non-string input is refused", () => {
    // Text arrives from JSON; the runtime check refuses anything but a string.
    expect(() => encodeOffsets(tokenizer("fast"), 42 as never)).toThrow(TypeError);
  });
});

describe("word units", () => {
  test("URLs, e-mail addresses and hyphenated words stay one unit; CJK splits per character", () => {
    const text = "see https://a.io/x, mail ada@ex.org re-run 東京";
    const words = splitWords(text).map(([a, b]) => Array.from(text).slice(a, b).join(""));
    expect(words).toEqual(["see", "https://a.io/x", ",", "mail", "ada@ex.org", "re-run", "東", "京"]);
  });
  test("the text and vision decoders keep their different compatibility-ideograph ranges", () => {
    // Upstream text range starts at U+8C48 (the NFC form of U+F900), vision at U+F900; see src/spans.ts.
    expect(CJK).toContain("\u8C48-\uFAFF");
    expect(CJK_VISION).toContain("\uF900-\uFAFF");
    expect(CJK_VISION).not.toContain("\u8C48");
    // U+A500 (Vai) is a letter inside U+8C48-U+FAFF only: a unit of its own for text, part of the word for vision.
    const text = "a\uA500b";
    const spans = (vision: boolean) =>
      decodeSpans([5, -5, 5], [[0, 1], [1, 2], [2, 3]], ["x"], text, 0.5, { vision }).map((s) => s.text);
    expect(spans(false)).toEqual(["a", "b"]);
    expect(spans(true)).toEqual([text]);
  });
});

describe("URL spans", () => {
  // decodeSpans documents step 4: text URLs lose trailing punctuation and anything after a CJK character; the vision
  // decoder keeps them, as span_decode_v2.py does.
  const url = (text: string, offsets: [number, number][], vision = false) =>
    decodeSpans(Array(offsets.length).fill(5), offsets, ["link"], text, 0.5, { vision }).map((s) => s.text);
  test("a text URL span drops trailing punctuation the run included", () => {
    expect(url("https://a.io/x.", [[0, 14], [14, 15]])).toEqual(["https://a.io/x"]);
    expect(url("www.a.io!?", [[0, 8], [8, 9], [9, 10]])).toEqual(["www.a.io"]);
  });
  test("a text URL span ends before the first CJK character", () => {
    const text = "https://a.io/" + String.fromCodePoint(0x6771, 0x4eac);
    expect(url(text, [[0, 13], [13, 14], [14, 15]])).toEqual(["https://a.io/"]);
  });
  test("the vision decoder keeps the URL as decoded", () => {
    expect(url("https://a.io/x.", [[0, 15]], true)).toEqual(["https://a.io/x."]);
  });
});

describe("region decoder", () => {
  test("chooses the four-connected peak component, not other islands", () => {
    const lg = Array(320).fill(-20);
    lg[0] = lg[1] = 20;
    lg[255] = 20;
    const a = decodeRegion(lg, "2x2", 0.7);
    expect(a.box).toEqual({ top: 0, bottom: 1 / 16, left: 0, right: 2 / 16 });
    expect(a.point).toEqual({ x: 1 / 32, y: 1 / 32 });
    // Peak cell: local p~1 averaged with its global cell p~0.
    expect(a.confidence).toBeCloseTo(0.5, 6);
  });
  test("a logit count that does not fit the grid is an error", () => {
    expect(() => decodeRegion(Array(64).fill(0), "2x2")).toThrow(/grid/);
  });
});
