// Portions derived from vela3-fast (Apache-2.0) and vela3-fast-vision (MIT).
// See THIRD_PARTY.md and documentation/licenses for attribution and license texts.
/**
 * Word alignment and span/region decoding, ported from the pinned `vela3_schema.py` (text), `probe_lib.py`
 * and `span_decode_v2.py` (vision).
 *
 * Offsets are Unicode code points with an exclusive end, as in Python, not UTF-16 units. WHY: the reference
 * fixtures and every consumer that compares against the Python implementation count code points; using
 * UTF-16 would shift every span after an emoji by one.
 *
 * The regular expressions emulate Python's `re` semantics where they differ from JavaScript: `\s` covers the
 * characters Python treats as whitespace (including U+001C–U+001F and U+0085, which JavaScript's `\s` does
 * not), and `[^\W_]` is spelled out as `[\p{L}\p{N}]`.
 * VERIFIED: tests/unit/spans.test.ts checks the decoders and the tokenizer offsets against the pinned Python outputs.
 * @module
 */

import type { BoxAnswer, ByteTokenizer, Span } from "./types.ts";

/**
 * CJK, kana, hangul and full-width ranges whose characters each form their own word unit (text adapters).
 *
 * Written as escapes on purpose. The upstream text decoder starts its compatibility-ideograph range at
 * U+8C48 (豈), the canonical twin of U+F900 (豈): the two glyphs look identical, and Unicode normalization
 * (NFC) turns U+F900 into U+8C48. So the upstream range really is U+8C48–U+FAFF. The vision decoder uses
 * U+F900–U+FAFF (see {@link CJK_VISION}). Spelled as literal characters, the difference is invisible and an
 * editor that normalizes on save would silently change the decoder. `tests/unit/spans.test.ts` pins both.
 */
export const CJK =
  "\u2E80-\u2FFF\u3000-\u303F\u3040-\u30FF\u3100-\u31FF\u3400-\u4DBF\u4E00-\u9FFF" +
  "\uA000-\uA4CF\uAC00-\uD7AF\u8C48-\uFAFF\uFE30-\uFE4F\uFF00-\uFFEF";
/** The vision decoder's CJK ranges (`span_decode_v2.py`): compatibility ideographs start at U+F900. */
export const CJK_VISION = CJK.replace("\u8C48-\uFAFF", "\uF900-\uFAFF");
const TRAIL = String.raw`.,;:!?)\\\]}>"'”’»`;
// Python's str.isspace() set, as a character class body.
const WS = String.raw`\t-\r \u001c-\u001f\u0085\u00A0\u1680\u2000-\u200A\u2028\u2029\u202F\u205F\u3000`;
/** Rewrite `\s`/`\S` in a pattern to Python's whitespace set. */
const pythonSpace = (s: string): string =>
  s.replaceAll("\\s", WS).replaceAll("\\S", `[^${WS}]`);
// Word units of the text schema (GLiNER2-style splitter): URLs, e-mail addresses, @handles, single CJK
// characters, letter/digit runs joined by - or _, and any other single non-space character.
const WORD = new RegExp(
  pythonSpace(
    String.raw`(?:https?://|www\.)[^\s${CJK}]+?(?=[${TRAIL}]*(?![^\s${CJK}]))|[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}|@[a-z0-9_]+|[${CJK}]|(?:(?![${CJK}])[\p{L}\p{N}_])+(?:[-_](?:(?![${CJK}])[\p{L}\p{N}_])+)*|\S`,
  ),
  "giu",
);
// Brackets and quotes trimmed from both ends of a decoded span.
const EDGE = new Set(Array.from("[](){}<>「」『』【】《》〈〉\"'“”‘’«»‹›"));
const space = (c: string | undefined): boolean => c !== undefined && new RegExp(`[${WS}]`, "u").test(c);

/**
 * All matches of `re` as code-point ranges.
 * @param re A global regular expression.
 * @returns `[start, end)` in code points.
 */
function matches(re: RegExp, text: string): [number, number][] {
  const cp = new Map<number, number>();
  let n = 0;
  for (let i = 0; i < text.length; ) {
    cp.set(i, n++);
    i += text.codePointAt(i)! > 65535 ? 2 : 1;
  }
  cp.set(text.length, n);
  return [...text.matchAll(re)].map((m) => [cp.get(m.index)!, cp.get(m.index + m[0].length)!]);
}

/**
 * Split text into the schema's word units.
 * @returns Code-point ranges of each word.
 */
export function splitWords(text: string): [number, number][] {
  return matches(WORD, text);
}

// ByteLevel BPE is reversible. Mapping bytes to source code points preserves partial
// UTF-8 tokens (emoji/CJK) and avoids quadratic tokenize-every-prefix alignment.
// This is GPT-2's bytes_to_unicode table, inverted: printable bytes map to themselves, the rest to U+0100+.
const byteMap = new Map<string, number>();
{
  const bytes = [
    ...Array.from({ length: 94 }, (_, i) => 33 + i),
    ...Array.from({ length: 12 }, (_, i) => 161 + i),
    ...Array.from({ length: 82 }, (_, i) => 174 + i),
  ];
  let n = 0;
  const have = new Set(bytes);
  for (let b = 0; b < 256; b++) if (!have.has(b)) bytes.push(b);
  for (const b of bytes)
    byteMap.set(String.fromCodePoint(have.has(b) ? b : 256 + n++), b);
}

/**
 * Tokenize `text` and return each token's code-point range in `text`.
 *
 * Tokenizers.js returns no offsets, so they are reconstructed from the token bytes: every ByteLevel token
 * maps back to the exact UTF-8 bytes it covers, which are then matched against the source text. WHY not
 * re-tokenize prefixes: that is quadratic and breaks on tokens that split a multi-byte character (emoji,
 * CJK), where a prefix is not valid UTF-8. Added (special) tokens are matched by their literal text.
 * @param tokenizer A Tokenizers.js tokenizer of the bundle.
 * @throws {TypeError} When `text` is not a string (callers pass parsed JSON).
 * @throws {Error} When token bytes do not line up with the text. The pinned tokenizers have no normalizer;
 *   a tokenizer that rewrites text would break the offset contract, so this fails instead of guessing.
 */
export function encodeOffsets(tokenizer: ByteTokenizer, text: string): { ids: number[]; offsets: [number, number][] } {
  if (typeof text !== "string") throw new TypeError("Text must be a string");
  const ids = tokenizer.encode(text, { add_special_tokens: false }).ids,
    enc = new TextEncoder(),
    bytes = enc.encode(text),
    byteCP: number[] = [];
  Array.from(text).forEach((c, i) => {
    for (let j = 0; j < enc.encode(c).length; j++) byteCP.push(i);
  });
  const added = tokenizer.get_added_tokens_decoder(),
    offsets: [number, number][] = [];
  let at = 0;
  for (const id of ids) {
    const token = tokenizer.id_to_token(id);
    // An id without a token string cannot be aligned; [undefined] fails the alignment check below.
    const raw: (number | undefined)[] =
      token === undefined
        ? [undefined]
        : added.has(id)
          ? Array.from(enc.encode(token))
          : Array.from(token, (c) => byteMap.get(c));
    if (raw.some((b, j) => b === undefined || bytes[at + j] !== b))
      throw new Error(
        "Tokenizer byte alignment failed; unsupported tokenizer normalization",
      );
    offsets.push([
      byteCP[at] ?? Array.from(text).length,
      (byteCP[at + raw.length - 1] ?? -1) + 1,
    ]);
    at += raw.length;
  }
  if (at !== bytes.length)
    throw new Error("Tokenizer did not cover the complete source text");
  return { ids, offsets };
}

/**
 * The first token of every word: the positions the span heads read.
 *
 * Words that no token overlaps (impossible for the pinned tokenizers, but cheap to guard) are skipped, so
 * `words[i]` and `positions[i]` always belong together.
 * @param offsets Token offsets from {@link encodeOffsets}.
 * @returns Word ranges and their first-token indices.
 */
export function firstWordTokens(text: string, offsets: readonly [number, number][]): { words: [number, number][]; positions: number[] } {
  const words: [number, number][] = [],
    positions: number[] = [];
  let t = 0;
  for (const [a, b] of splitWords(text)) {
    while (t < offsets.length && offsets[t][1] <= a) t++;
    if (t < offsets.length && offsets[t][0] < b) {
      words.push([a, b]);
      positions.push(t);
    }
  }
  return { words, positions };
}

/**
 * Decode per-word logits into labelled character spans.
 *
 * Mirrors the upstream decoders step by step:
 * 1. Sigmoid with temperature per word and label; the best label wins when it clears `threshold`.
 * 2. Words are grouped into "units" (a URL, an e-mail address, a letter/digit run); within a unit the label
 *    with the highest summed probability wins, and gaps inside the unit are filled with it. WHY: the heads
 *    score the first token of each schema word, and one unit (`ada@example.org`) can span several words,
 *    so without voting a single e-mail address could come back as three spans with two labels.
 * 3. Consecutive words with the same label merge into one span.
 * 4. Whitespace, brackets and quotes are trimmed from the span ends; text URLs also lose trailing
 *    punctuation and anything after a CJK character.
 *
 * Text spans carry `probability` (mean over the run's words) and `score` (mean over every word overlapping
 * the trimmed span). Vision spans carry only `score`, as in `span_decode_v2.py`.
 * @param logits `words × labels` values, row-major.
 * @param offsets Word ranges in code points.
 * @param names Label names in head order.
 * @param text The text the offsets index into.
 * @param threshold Minimum sigmoid probability in [0, 1].
 * @param options `temperature` divides logits before the sigmoid (the PII and toxicity routes use calibrated
 *   temperatures); `vision` selects the vision decoder's rules.
 */
export function decodeSpans(
  logits: ArrayLike<number>,
  offsets: readonly [number, number][],
  names: readonly string[],
  text: string,
  threshold: number,
  { temperature = 1, vision = false }: { temperature?: number; vision?: boolean } = {},
): Span[] {
  const chars = Array.from(text),
    K = names.length,
    T = offsets.length,
    p = Float32Array.from(logits, (x) => 1 / (1 + Math.exp(-x / temperature))),
    lab: number[] = [],
    pk: number[] = [];
  for (let i = 0; i < T; i++) {
    let k = 0;
    for (let j = 1; j < K; j++) if (p[i * K + j] > p[i * K + k]) k = j;
    pk.push(p[i * K + k]);
    lab.push(pk[i] > threshold ? k : -1);
  }
  const cj = vision ? CJK_VISION : CJK;
  const body = String.raw`[^\s\]\)}>「」『』"'“”${cj}]`;
  // Text URLs stop before trailing punctuation (lazy body + lookahead); vision URLs are greedy upstream.
  const url = vision
    ? String.raw`(?:https?://|www\.)${body}+`
    : String.raw`(?:https?://|www\.)${body}+?(?=[.,;:!?]*(?!${body}))`;
  const unitRE = new RegExp(
    pythonSpace(String.raw`${url}|(?:(?![${cj}])[\p{L}\p{N}]|[@._\-+])+|\S`),
    "gu",
  );
  const u = new Int32Array(chars.length + 1).fill(-1);
  matches(unitRE, text).forEach(([a, b], k) => u.fill(k, a, b));
  const tu = offsets.map(([a, b]) => {
    while (a < b && space(chars[a])) a++;
    return a < b ? u[a] : -1;
  });
  for (let t = 0; t < T; ) {
    if (tu[t] < 0) {
      t++;
      continue;
    }
    let e = t;
    while (e + 1 < T && tu[e + 1] === tu[t]) e++;
    const selected: number[] = [];
    for (let i = t; i <= e; i++) if (lab[i] >= 0) selected.push(i);
    if (selected.length) {
      const scores = new Map<number, number>();
      for (const i of selected)
        scores.set(lab[i], (scores.get(lab[i]) ?? 0) + pk[i]);
      let winner = lab[selected[0]],
        best = -1;
      for (const [k, s] of scores)
        if (s > best) {
          winner = k;
          best = s;
        }
      for (const i of selected) lab[i] = winner;
      for (let i = selected[0]; i <= selected.at(-1)!; i++)
        if (lab[i] < 0) lab[i] = winner;
    }
    t = e + 1;
  }
  type Run = { k: number; start: number; end: number; indices: number[] };
  const runs: Run[] = [];
  let cur: Run | null = null;
  for (let i = 0; i < T; i++) {
    if (cur && lab[i] === cur.k) {
      cur.end = offsets[i][1];
      cur.indices.push(i);
      continue;
    }
    if (cur) runs.push(cur);
    cur =
      lab[i] >= 0
        ? { k: lab[i], start: offsets[i][0], end: offsets[i][1], indices: [i] }
        : null;
  }
  if (cur) runs.push(cur);
  return runs.flatMap((r): Span[] => {
    let { start: s, end: e, k } = r;
    while (s < e && (space(chars[s]) || EDGE.has(chars[s]))) s++;
    while (e > s && (space(chars[e - 1]) || EDGE.has(chars[e - 1]))) e--;
    if (!vision && /^(https?:\/\/|www\.)/i.test(chars.slice(s, e).join(""))) {
      for (let i = s; i < e; i++)
        if (new RegExp(`[${CJK}]`, "u").test(chars[i])) {
          e = i;
          break;
        }
      while (
        e > s &&
        (".,;:!?)".includes(chars[e - 1]) ||
          EDGE.has(chars[e - 1]) ||
          space(chars[e - 1]))
      )
        e--;
    }
    if (e <= s) return [];
    const overlap = offsets
      .map(([a, b], i) => (a < e && b > s ? i : -1))
      .filter((i) => i >= 0);
    const mean = (is: number[]) =>
      is.reduce((sum, i) => sum + p[i * K + k], 0) / is.length;
    return [
      {
        start: s,
        end: e,
        label: names[k],
        ...(vision ? {} : { probability: mean(r.indices) }),
        text: chars.slice(s, e).join(""),
        score: mean(overlap),
      },
    ];
  });
}

/**
 * Decode the region adapter's per-cell logits into a normalized box, a point and a confidence.
 *
 * The adapter scores 64 cells (8 × 8) per 512 px tile plus 64 cells of the global view. Each cell of the
 * `rows·8 × cols·8` heat map averages its local tile score with the global score covering the same area;
 * the box is the four-connected component around the peak whose cells reach `frac × peak`. WHY the
 * connected component and not every cell above the cut: a second, separate bright island elsewhere in the
 * image must not stretch the box across both.
 * @param logits `(rows·cols + 1) × 64` cell logits, tiles first, global view last.
 * @param grid `"rows x cols"` from preprocessing.
 * @param frac Fraction of the peak a cell needs to join the box (the adapter config pins 0.7).
 * @returns Coordinates in [0, 1], origin top left; `point` is the peak cell's center.
 * @throws {Error} When the logit count does not match the grid.
 */
export function decodeRegion(logits: ArrayLike<number>, grid: string, frac = 0.7): BoxAnswer {
  const [R, C] = grid.split("x").map(Number),
    H = R * 8,
    W = C * 8;
  if (logits.length !== (R * C + 1) * 64)
    throw new Error("Invalid region feature grid");
  const p = Float32Array.from(logits, (x) => 1 / (1 + Math.exp(-x))),
    h = new Float32Array(H * W);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const local =
        (Math.floor(y / 8) * C + Math.floor(x / 8)) * 64 +
        (y % 8) * 8 +
        (x % 8);
      const global = R * C * 64 + Math.floor(y / R) * 8 + Math.floor(x / C);
      h[y * W + x] = (p[local] + p[global]) / 2;
    }
  let peak = 0;
  for (let i = 1; i < h.length; i++) if (h[i] > h[peak]) peak = i;
  const seen = new Set([peak]),
    queue = [peak],
    cut = frac * h[peak];
  let ymin = H,
    ymax = 0,
    xmin = W,
    xmax = 0;
  for (let j = 0; j < queue.length; j++) {
    const i = queue[j],
      y = Math.floor(i / W),
      x = i % W;
    ymin = Math.min(ymin, y);
    ymax = Math.max(ymax, y);
    xmin = Math.min(xmin, x);
    xmax = Math.max(xmax, x);
    for (const [a, b] of [
      [y - 1, x],
      [y + 1, x],
      [y, x - 1],
      [y, x + 1],
    ]) {
      const n = a * W + b;
      if (a >= 0 && a < H && b >= 0 && b < W && !seen.has(n) && h[n] >= cut) {
        seen.add(n);
        queue.push(n);
      }
    }
  }
  return {
    type: "box",
    box: {
      top: ymin / H,
      bottom: (ymax + 1) / H,
      left: xmin / W,
      right: (xmax + 1) / W,
    },
    point: { y: (Math.floor(peak / W) + 0.5) / H, x: ((peak % W) + 0.5) / W },
    confidence: h[peak],
  };
}
