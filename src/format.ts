/**
 * Input rendering and answer decoding for the v0.1 decision heads (`choice` and `noul`).
 *
 * The token layout reproduces the pinned Python renderers id for id, because the trained heads read fixed
 * positions (the `[O]` markers, the pooled `[SEG_user]` span, the per-option masks). A layout that is only
 * "close" would still run and still return probabilities, just wrong ones, so every helper here either
 * produces the exact upstream sequence or throws. VERIFIED: tests/unit/format.test.ts renders every recorded Python
 * fixture to its reference ids, option positions and pool.
 *
 * Sources: `vela3_encoder.py` (vela3-fast) and `uvb_core.py` (vela3-fast-vision) at the revisions pinned in
 * `documentation/ADAPTER_VALIDATION.md`.
 * @module
 */

import type { Answer, ChoiceAnswer, DecisionQuestion, FastManifest, NoulAnswer, TokenizerLike, VisionManifest } from "./types.ts";

/**
 * Serialize a state or instruction value the way the Python renderer does.
 *
 * Strings pass through untouched (Python inserts them verbatim); every other JSON value is encoded with
 * `JSON.stringify`. WHY: the models were trained on the Python rendering, and a string that went through
 * `JSON.stringify` would gain quotes and escapes the model never saw.
 *
 * Caveat: JavaScript and Python format some floats differently (`1e21` vs `1e+21`), so pass an already
 * serialized string when byte-exact parity matters.
 * @param value Text or any JSON-serializable value.
 * @throws {TypeError} When `value` has no JSON representation (`undefined`, a function, a symbol).
 */
export function jsonText(value: unknown): string {
  if (typeof value === "string") return value;
  const encoded: string | undefined = JSON.stringify(value);
  if (encoded === undefined) throw new TypeError("Expected a JSON value");
  return encoded;
}

/**
 * Check that a decision question is one the exported heads can answer, and return its option names.
 *
 * WHY so strict: the heads were trained on exactly these shapes. A question outside them would still produce
 * logits, and nothing downstream could tell that the numbers are meaningless.
 *
 * - `noul` (yes/no) has no criteria: its two options are fixed as `["no", "yes"]`.
 * - `choice` needs a non-empty object of option name → description (`string` or `null`).
 * - Integer-like option names (`"0"`, `"12"`) are refused: JavaScript orders integer keys numerically while
 *   Python keeps insertion order, so the option order, and therefore the answer, would differ between the
 *   exporter and this runtime.
 *
 * The parameter is `unknown` because questions arrive from JSON and user code; this is the runtime check.
 * @returns Option names in model order.
 * @throws {TypeError} For any unsupported or ambiguous question.
 */
export function validateQuestion(q: unknown): string[] {
  const question = q as { type?: unknown; criteria?: unknown } | null | undefined;
  if (!question || (question.type !== "choice" && question.type !== "noul"))
    throw new TypeError("Supported question types: choice, noul");
  if (question.type === "noul") {
    if (question.criteria != null) throw new TypeError("noul criteria are unsupported by these Vela decision heads");
    return ["no", "yes"];
  }
  const criteria = question.criteria;
  if (!criteria || Array.isArray(criteria) || typeof criteria !== "object" || !Object.keys(criteria).length)
    throw new TypeError("choice requires a nonempty criteria object");
  if (Object.keys(criteria).some((k) => /^(0|[1-9]\d*)$/.test(k)))
    throw new TypeError("Use non-integer option keys to preserve cross-language order");
  if (Object.values(criteria).some((v) => v != null && typeof v !== "string"))
    throw new TypeError("Option descriptions must be strings or null");
  return Object.keys(criteria);
}

/**
 * Token ids of `text` without BOS/EOS.
 *
 * WHY no special tokens: the renderers place BOS, EOS and the marker tokens themselves by id; letting the
 * tokenizer add them would duplicate or misplace them.
 */
export function encodeText(tokenizer: TokenizerLike, text: string): number[] {
  return tokenizer.encode(text, { add_special_tokens: false }).ids;
}

/** Description of option `name`, or `null` (always `null` for `noul`). */
const description = (q: DecisionQuestion, name: string): string | null => (q.type === "choice" ? (q.criteria[name] ?? null) : null);

/**
 * Render a text decision for the vela3-fast decision graph.
 *
 * Layout: `BOS [Q] instructions ([O] option[: description])* [SEP_SCHEMA] [SEG_user] state EOS`.
 * `optionPositions` are the `[O]` marker positions the option head reads; `pool` is the half-open token
 * range the state is mean-pooled over. It includes `[SEG_user]` and excludes EOS, as upstream does.
 * @param m Bundle manifest (marker ids, BOS/EOS, token limit).
 * @param state Text or JSON value describing the situation to decide on.
 * @throws {RangeError} When the rendered input exceeds `m.maxTokens`. Nothing is truncated.
 */
export function formatFast(
  tokenizer: TokenizerLike,
  m: FastManifest,
  state: unknown,
  q: DecisionQuestion,
): { ids: number[]; names: string[]; optionPositions: number[]; pool: [number, number] } {
  const names = validateQuestion(q),
    mk = m.markers;
  const ids = [m.bosId, mk["[Q]"], ...encodeText(tokenizer, jsonText(q.instructions ?? ""))],
    optionPositions: number[] = [];
  for (const name of names) {
    optionPositions.push(ids.length);
    const desc = description(q, name);
    ids.push(mk["[O]"], ...encodeText(tokenizer, desc ? `${name}: ${desc}` : name));
  }
  ids.push(mk["[SEP_SCHEMA]"]);
  const start = ids.length;
  ids.push(mk["[SEG_user]"], ...encodeText(tokenizer, jsonText(state ?? "")));
  const end = ids.length;
  ids.push(m.eosId);
  checkLength(ids, m.maxTokens);
  return { ids, names, optionPositions, pool: [start, end] };
}

/**
 * Render an image (or image-free) decision for the vela3-fast-vision decision graph.
 *
 * Layout: one image prefix per image (the processor's BOS only once per request), then
 * `Input: state`, ` Question: instructions`, and each option between `sep` tokens. `spans` are the
 * half-open token ranges of each option; the head mean-pools each range.
 *
 * Text token ids at or above 49152 (the text vocabulary size) are mapped to `m.unknownTextId`. WHY: the
 * vision tokenizer appends image tokens above the text vocabulary, and the upstream processor maps any text
 * token in that range to its unknown id, so a user typing an image-token string cannot inject image slots.
 * @param grids Tile grid per image (`"1x2"`, `"2x1"`, `"2x2"`) from `preprocessImage` (`image.ts`).
 * @throws {RangeError} For too many images, an unknown grid or an over-long input.
 * @throws {TypeError} When an option renders to zero tokens (the head would average nothing).
 */
export function formatVision(
  tokenizer: TokenizerLike,
  m: VisionManifest,
  state: unknown,
  q: DecisionQuestion,
  grids: string[] = [],
): { ids: number[]; names: string[]; spans: [number, number][] } {
  const names = validateQuestion(q);
  if (grids.length > m.maxImages) throw new RangeError(`At most ${m.maxImages} images`);
  const ids: number[] = [];
  // Without images the processor still starts the request with BOS (id 0).
  if (!grids.length) ids.push(0);
  for (const [i, grid] of grids.entries()) {
    const prefix = m.imagePrefixes[grid];
    if (!prefix) throw new RangeError(`Unsupported image grid: ${grid}`);
    ids.push(...(i ? prefix.slice(1) : prefix)); // official processor emits one BOS per request
  }
  const textIds = (text: string) => encodeText(tokenizer, text).map((i) => (i >= 49152 ? m.unknownTextId : i));
  const st = state == null ? "" : jsonText(state);
  if (st.trim()) ids.push(...textIds(`Input: ${st.trim()}`));
  ids.push(...textIds(` Question: ${jsonText(q.instructions ?? "").trim()}`));
  const spans: [number, number][] = [];
  for (const name of names) {
    ids.push(m.sepId);
    const start = ids.length;
    const desc = description(q, name);
    const text = q.type === "noul" ? (name === "no" ? "No / false" : "Yes / true") : desc ? `${name}: ${desc}` : name;
    ids.push(...textIds(` ${text}`));
    if (ids.length === start) throw new TypeError("Option must tokenize to at least one token");
    spans.push([start, ids.length]);
  }
  ids.push(m.sepId);
  checkLength(ids, m.maxTokens);
  return { ids, names, spans };
}

/**
 * Refuse inputs longer than the bundle's token budget.
 *
 * WHY refuse instead of truncate: the head reads positions at the end of the sequence (options, pooled
 * state). Cutting the tail would drop exactly those, and the answer would look valid while being wrong.
 * @throws {RangeError}
 */
export function checkLength(ids: readonly number[], maxTokens: number): void {
  if (ids.length > maxTokens) throw new RangeError(`Input has ${ids.length} tokens; limit ${maxTokens}. Nothing was truncated.`);
}

/**
 * Numerically stable softmax.
 *
 * WHY subtract the maximum: decision logits can be large (the tests use 10 000), and `Math.exp` of those
 * overflows to `Infinity`, turning every probability into `NaN`.
 * @returns Probabilities summing to 1.
 * @throws {Error} For empty or non-finite logits, which signal a broken graph or input, not a decision.
 */
export function probabilities(logits: ArrayLike<number>): number[] {
  const values = Array.from(logits);
  if (!values.length || values.some((x) => !Number.isFinite(x))) throw new Error("Non-finite or empty model output");
  const max = Math.max(...values),
    exp = values.map((x) => Math.exp(x - max)),
    total = exp.reduce((a, b) => a + b, 0);
  return exp.map((x) => x / total);
}

/**
 * Turn decision logits into the public answer object.
 *
 * `noul` returns p(yes). `choice` returns the most probable option and all probabilities; ties resolve to
 * the first option, which matches `numpy.argmax` in the reference implementation.
 * @param names Option names from {@link validateQuestion}.
 * @param logits One logit per option.
 * @throws {Error} When the graph returned a different number of options than were asked.
 */
export function answer(q: Pick<DecisionQuestion, "type">, names: string[], logits: ArrayLike<number>): Answer {
  const p = probabilities(logits);
  if (p.length !== names.length) throw new Error("Output option count mismatch");
  if (q.type === "noul") return { type: "noul", noul: p[1] } satisfies NoulAnswer;
  const best = p.reduce((j, x, i) => (x > p[j] ? i : j), 0);
  return {
    type: "choice",
    choice: names[best],
    probabilities: Object.fromEntries(names.map((n, i) => [n, p[i]])),
  } satisfies ChoiceAnswer;
}
